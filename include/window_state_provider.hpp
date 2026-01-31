#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <expected>

#include <wayland-client.h>
#include "state_provider.hpp"

struct WindowInfo {
    std::string window_id;
    std::string title;
    std::string app_id;
};

class WindowStateProvider : public StateProvider {
public:
    ~WindowStateProvider() noexcept;
    std::expected<void, StateProviderError> init() noexcept;

    [[nodiscard]] std::vector<WindowInfo> get_open_windows() noexcept;
    [[nodiscard]] std::optional<WindowInfo> get_window_state(std::string_view window_id) noexcept;

private:
    // std::optional<CCZcosmicToplevelInfoV1> m_toplevel_info { std::nullopt };
    wl_display* m_display { nullptr };
};
