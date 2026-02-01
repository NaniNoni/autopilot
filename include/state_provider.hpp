#pragma once

#include <expected>
#include <variant>

enum class WindowStateProviderError {
    WL_DISPLAY_CONNECT_ERROR,
    WL_UNSUPPORTED_COMPOSITOR
};

using StateProviderError = std::variant<WindowStateProviderError>;

class StateProvider {
public:
    virtual std::expected<void, StateProviderError> init() noexcept = 0;
};
