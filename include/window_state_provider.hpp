#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <vector>
#include <wayland-client.h>

#include "cosmic-toplevel-info-unstable-v1.hpp"
#include "state_provider.hpp"

struct WindowInfo {
    std::string window_id;
    std::string title;
    std::string app_id;
};

class WindowStateProvider : public StateProvider {
public:
    WindowStateProvider() noexcept;
    ~WindowStateProvider() noexcept;

    [[nodiscard]] std::vector<WindowInfo> get_open_windows() noexcept;
    [[nodiscard]] std::optional<WindowInfo> get_window_state(std::string_view window_id) noexcept;

private:
    bool m_ready { false };
    wl_display* m_display { nullptr };
    wl_registry* m_registry { nullptr };
    std::optional<CCZcosmicToplevelInfoV1> m_toplevel_info { std::nullopt };

    void ensure_connected() noexcept;
    void pump_events_once() noexcept;
    void pump_until_roundtrip() noexcept;
};
