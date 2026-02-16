#pragma once

#include <expected>

#include <nlohmann/json.hpp>

enum class StateRequestError {
    INVALID
};

enum class StateProviderKind {
    WINDOW
};

std::optional<StateProviderKind> state_provider_kind_from_string(std::string_view str) noexcept;

struct StateRequest {
    [[nodiscard]] static std::expected<StateRequest, StateRequestError> from_json(std::string_view str) noexcept;
    StateProviderKind kind;
    nlohmann::json args;
};
