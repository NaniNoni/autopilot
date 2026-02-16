#include "orchestrator.hpp"
#include "llama.h"
#include "state_request.hpp"
#include "window_state_provider.hpp"

#include <expected>
#include <print>
#include <iostream>

#include <spdlog/spdlog.h>
#include <nlohmann/json.hpp>

static void llama_log_callback(enum ggml_log_level level, const char* text, [[maybe_unused]] void* user_data) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        spdlog::debug("{}", text);
    }
}

std::expected<void, OrchestratorError> Orchestrator::init() noexcept {
    llama_log_set(llama_log_callback, this);

    const char* orchestrator_path_env = std::getenv("ORCHESTRATOR_MODEL_PATH");
    std::string orchestrator_path = orchestrator_path_env ? orchestrator_path_env : DEFAULT_ORCHESTRATOR_PATH;
    if (orchestrator_path.empty()) {
        spdlog::error("Error: Orchestrator path is empty");
        return std::unexpected(OrchestratorError::MODEL_BAD_PATH);
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = N_GPU_LAYERS;
    model = llama_model_load_from_file(orchestrator_path.c_str(), model_params);
    if (model == nullptr) {
        spdlog::error("Error: unable to load model {}", orchestrator_path);
        return std::unexpected(OrchestratorError::MODEL_LOAD_FAILED);
    }

    auto window_state_provider = std::make_unique<WindowStateProvider>();
    if (!window_state_provider->init()) {
        spdlog::error("Failed to initialize window state provider");
        return std::unexpected(OrchestratorError::STATE_PROVIDER_ERROR);
    }
    m_state_providers.insert({StateProviderKind::WINDOW, std::move(window_state_provider)});

    vocab = llama_model_get_vocab(model);
    while (true) {
        std::print("> ");
        std::string input;
        std::getline(std::cin, input);

        process_prompt(input);
    }
}

Orchestrator::~Orchestrator() noexcept {
    llama_sampler_free(smpl);
    llama_free(ctx);
    llama_model_free(model);
}

static std::string_view messager_role_to_string(MessagerRole role) noexcept {
    switch (role) {
        case MessagerRole::System: return "system";
        case MessagerRole::User: return "user";
        case MessagerRole::Assistant: return "assistant";
    };

    return "INVALID_ROLE";
}

static constexpr std::string_view MESSAGE_HEADER_START = "<|start_header_id|>";
static constexpr std::string_view MESSAGE_HEADER_END = "<|end_header_id|>";
static constexpr std::string_view MESSAGE_EOT = "<|eot_id|>";
static constexpr std::string_view BEGIN_OF_TEXT = "<|begin_of_text|>";

std::string Message::to_string() const noexcept {
    std::string out;

    out += MESSAGE_HEADER_START;
    out += messager_role_to_string(role);
    out += MESSAGE_HEADER_END;
    out += '\n';
    out += content;
    out += MESSAGE_EOT;

    return out;
}

const Message IDENTITY_MESSAGE = Message {
    .role = MessagerRole::System,
    .content = SYSTEM_PROMPT
};

std::string Orchestrator::build_history() noexcept {
    std::string out;

    out += BEGIN_OF_TEXT;
    out += '\n';
    out += IDENTITY_MESSAGE.to_string();

    for (Message message : m_history) {
        out += message.to_string();
    }

    // open the assistant turn
    out += MESSAGE_HEADER_START;
    out += messager_role_to_string(MessagerRole::Assistant);
    out += MESSAGE_HEADER_END;
    out += '\n';

    return out;
}

int Orchestrator::process_prompt(const std::string& user_prompt) {
    m_history.push_back(Message {
        .role = MessagerRole::User,
        .content = user_prompt
    });

    std::expected<std::string, LLMError> llm_out = run_llm();
    if (!llm_out) {
        spdlog::error("Error occurred while running LLM, exiting.");
        return 1;
    }

    // determine if the LLM output is a state-fetch instruction
    // for now, any valid json is considered a state-fetch instruction
    // this should probably be fixed for security purposes
    std::expected<StateRequest, StateRequestError> req = StateRequest::from_json(*llm_out);
    if (req) {
        StateProviderKind kind = (*req).kind;
        std::unique_ptr<StateProvider>& provider = m_state_providers[kind];
        nlohmann::json out = provider->processRequest(*req);

        m_history.push_back(Message {
            .role = MessagerRole::System,
            .content = out.dump(4)
        });

        std::println();

        std::expected<std::string, LLMError> llm_response = run_llm();
        if (!llm_response) {
            spdlog::error("Error occurred while running LLM, exiting.");
            return 1;
        }
    }
    else {
        spdlog::debug("Assitant output is not a valid JSON command. Continuing.");
    }

    std::println();

    return 0;
}

std::expected<std::string, Orchestrator::LLMError> Orchestrator::run_llm() noexcept {
    std::string full_prompt = build_history();

    // tokenize the prompt
    // find the number of tokens in the prompt
    const i32 n_prompt = -llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), nullptr, 0, false, true);
    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), false, true) < 0) {
        spdlog::error("Failed to tokenize the prompt - {}", full_prompt);
        return std::unexpected(LLMError::TOKENIZE_FAILED);
    }

    llama_context_params ctx_params = llama_context_default_params();
    // n_ctx is the context size
    ctx_params.n_ctx = n_prompt + N_PREDICT - 1;
    // n_batch is the maximum number of tokens that can be processed in a single call to llama_decode
    ctx_params.n_batch = n_prompt;
    // enable performance counters
    ctx_params.no_perf = false;

    ctx = llama_init_from_model(model, ctx_params);
    if (ctx == nullptr) {
        spdlog::error("Failed to create llama_context");
        return std::unexpected(LLMError::CONTEXT_CREATION_FAILED);
    }

    auto sparams = llama_sampler_chain_default_params();
    sparams.no_perf = false;
    smpl = llama_sampler_chain_init(sparams);

    llama_sampler_chain_add(smpl, llama_sampler_init_greedy());

    // print the prompt token-by-token
    for (auto id : prompt_tokens) {
        char buf[128];
        int n = llama_token_to_piece(vocab, id, buf, sizeof(buf), 0, true);
        if (n < 0) {
            std::println("Failed to convert token to piece");
            return std::unexpected(LLMError::TOKEN_TO_PIECE_CONVERSION_FAILED);
        }
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            std::println("Failed to eval");
            return std::unexpected(LLMError::EVALUATION_FAILED);
        }

        llama_token decoder_start_token_id = llama_model_decoder_start_token(model);
        if (decoder_start_token_id == LLAMA_TOKEN_NULL) {
            decoder_start_token_id = llama_vocab_bos(vocab);
        }

        batch = llama_batch_get_one(&decoder_start_token_id, 1);
    }

    const auto t_main_start = ggml_time_us();
    int n_decode = 0;
    llama_token new_token_id;
    std::string assistant_text;

    for (int n_pos = 0; n_pos + batch.n_tokens < n_prompt + N_PREDICT; ) {
        // evaluate the current batch with the transformer model
        if (llama_decode(ctx, batch)) {
            spdlog::error("Failed to eval");
            return std::unexpected(LLMError::EVALUATION_FAILED);
        }

        n_pos += batch.n_tokens;

        // sample the next token
        {
            new_token_id = llama_sampler_sample(smpl, ctx, -1);

            // is it an end of generation?
            if (llama_vocab_is_eog(vocab, new_token_id)) {
                break;
            }

            char buf[128];
            int n = llama_token_to_piece(vocab, new_token_id, buf, sizeof(buf), 0, false);
            if (n < 0) {
                fprintf(stderr, "%s: error: failed to convert token to piece\n", __func__);
                return std::unexpected(LLMError::TOKEN_TO_PIECE_CONVERSION_FAILED);
            }

            std::string_view piece(buf, n);
            std::print("{}", piece);
            std::fflush(stdout);

            assistant_text.append(piece);

            // prepare the next batch with the sampled token
            batch = llama_batch_get_one(&new_token_id, 1);

            n_decode += 1;
        }
    }

    m_history.push_back(Message {
        .role = MessagerRole::Assistant,
        .content = assistant_text
    });

    return assistant_text;
}
