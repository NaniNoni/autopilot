#include "window_state_provider.hpp"

#include <spdlog/spdlog.h>
#include <wayland-client-core.h>

std::expected<void, StateProviderError> WindowStateProvider::init() noexcept {
    m_display = wl_display_connect(nullptr);
    spdlog::debug("Connecting to Wayland display");

    if (!m_display) {
        spdlog::error("Could not connect to Wayland display");
        return std::unexpected(WindowStateProviderError::WL_DISPLAY_CONNECT_ERROR);
    }

    return {};
}

WindowStateProvider::~WindowStateProvider() noexcept {
    wl_display_disconnect(m_display);
}
