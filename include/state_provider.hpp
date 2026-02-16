#pragma once

#include <expected>
#include <variant>

#include "state_request.hpp"

enum class WindowStateProviderError {
    WL_DISPLAY_CONNECT_ERROR,
    WL_UNSUPPORTED_COMPOSITOR
};

using StateProviderError = std::variant<WindowStateProviderError>;

class StateProvider {
public:
    virtual std::expected<void, StateProviderError> init() noexcept = 0;
    virtual ~StateProvider() noexcept = default;
    virtual nlohmann::json processRequest(StateRequest req) noexcept = 0;
};
