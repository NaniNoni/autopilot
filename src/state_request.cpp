#include "state_request.hpp"

#include <expected>
#include <optional>
#include <spdlog/spdlog.h>

std::optional<StateProviderKind> state_provider_kind_from_string(std::string_view str) noexcept {
    if (str == "window") {
        return std::make_optional(StateProviderKind::WINDOW);
    }

    return std::nullopt;
}

std::expected<StateRequest, StateRequestError> StateRequest::from_json(std::string_view str) noexcept {
    std::optional<StateProviderKind> kind;
    nlohmann::json args;

    try {
        nlohmann::json obj = nlohmann::json::parse(str);

        std::string kind_str = obj["request_kind"].get<std::string>();
        kind = state_provider_kind_from_string(kind_str);
        if (!kind) {
            return std::unexpected(StateRequestError::INVALID);
        }

        args = obj["args"];
    }
    catch (const nlohmann::json::exception& err) {
        return std::unexpected(StateRequestError::INVALID);
    }

    return StateRequest {
        *kind, std::move(args)
    };
}
