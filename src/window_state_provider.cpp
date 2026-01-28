#include "window_state_provider.hpp"

#include <spdlog/spdlog.h>
#include <wayland-client.h>

#include "int_types.hpp"

static void registry_global(void* data, wl_registry* registry, u32 name, const char* interface, u32 version);
static void registry_global_remove(void* data, wl_registry* registry, u32 name);

static const wl_registry_listener REGISTRY_LISTENER = {
    .global = registry_global,
    .global_remove = registry_global_remove,
};

WindowStateProvider::WindowStateProvider() noexcept {
    ensure_connected();
}

void WindowStateProvider::ensure_connected() noexcept {
    if (m_ready) return;

    m_display = wl_display_connect(nullptr);
    if (!m_display) {
        spdlog::error("wl_display_connect failed");
        return;
    }

    m_registry = wl_display_get_registry(m_display);
    if (!m_registry) {
        spdlog::error("wl_display_get_registry failed");
        return;
    }

    wl_registry_add_listener(m_registry, &REGISTRY_LISTENER, this);

    if (!m_toplevel_info) {
        spdlog::error("COSMIC toplevel info not available");
        return;
    }

    m_ready = true;
}

void WindowStateProvider::pump_events_once() noexcept {
    if (!m_display) return;

    wl_display_dispatch_pending(m_display);
    wl_display_flush(m_display);
}

void WindowStateProvider::pump_until_roundtrip() noexcept {
    if (!m_display) return;

    wl_display_roundtrip(m_display);
}
