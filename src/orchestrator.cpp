#include "orchestrator.hpp"
#include "llama.h"

#include <cstdio>
#include <expected>
#include <print>
#include <iostream>

#include <nlohmann/json.hpp>

static void llama_log_callback(enum ggml_log_level level, const char* text, [[maybe_unused]] void* user_data) {
    if (level >= GGML_LOG_LEVEL_WARN) {
        std::print("{}", text);
    }
}

std::expected<void, OrchestratorError> Orchestrator::init() noexcept {
    llama_log_set(llama_log_callback, this);

    const char* orchestrator_path_env = std::getenv("ORCHESTRATOR_MODEL_PATH");
    std::string orchestrator_path = orchestrator_path_env ? orchestrator_path_env : DEFAULT_ORCHESTRATOR_PATH;
    if (orchestrator_path.empty()) {
        std::println("Error: Orchestrator path is empty");
        return std::unexpected(OrchestratorError::MODEL_BAD_PATH);
    }

    ggml_backend_load_all();
    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = N_GPU_LAYERS;
    model = llama_model_load_from_file(orchestrator_path.c_str(), model_params);
    if (model == nullptr) {
        std::println("Error: unable to load model {}", orchestrator_path);
        return std::unexpected(OrchestratorError::MODEL_LOAD_FAILED);
    }

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

    for (Message message : history) {
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
    history.push_back(Message {
        .role = MessagerRole::User,
        .content = user_prompt
    });

    std::string full_prompt = build_history();

    // tokenize the prompt
    // find the number of tokens in the prompt
    const i32 n_prompt = -llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), nullptr, 0, false, true);
    // allocate space for the tokens and tokenize the prompt
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, full_prompt.c_str(), full_prompt.size(), prompt_tokens.data(), prompt_tokens.size(), false, true) < 0) {
        std::println("Error: failed to tokenize the prompt - {}", full_prompt);
        return 1;
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
        std::println("Error: failed to create llama_context");
        return 1;
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
            std::println("Error: failed to convert token to piece");
            return 1;
        }
    }

    // prepare a batch for the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());

    if (llama_model_has_encoder(model)) {
        if (llama_encode(ctx, batch)) {
            std::println("Error : failed to eval");
            return 1;
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
            fprintf(stderr, "%s : failed to eval, return code %d\n", __func__, 1);
            return 1;
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
                return 1;
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

    history.push_back(Message {
        .role = MessagerRole::Assistant,
        .content = assistant_text
    });

    printf("\n");

    const auto t_main_end = ggml_time_us();

    fprintf(stderr, "%s: decoded %d tokens in %.2f s, speed: %.2f t/s\n",
            __func__, n_decode, (t_main_end - t_main_start) / 1000000.0f, n_decode / ((t_main_end - t_main_start) / 1000000.0f));


    // fprintf(stderr, "\n");
    // llama_perf_sampler_print(smpl);
    // llama_perf_context_print(ctx);
    fprintf(stderr, "\n");

    return 0;
}
