#include "window_state_provider.hpp"
#include "ext-foreign-toplevel-list-v1-client-protocol.h"
#include "int_types.hpp"
#include "state_provider.hpp"

#include <spdlog/spdlog.h>
#include <wayland-client-core.h>
#include <wayland-client-protocol.h>

std::expected<void, StateProviderError> WindowStateProvider::init() noexcept {
    spdlog::debug("Connecting to Wayland display");
    m_display = wl_display_connect(nullptr);
    if (!m_display) {
        spdlog::error("Could not connect to Wayland display");
        return std::unexpected(WindowStateProviderError::WL_DISPLAY_CONNECT_ERROR);
    }

    m_registry = wl_display_get_registry(m_display);
    wl_registry_add_listener(m_registry, &REGISTRY_LISTENER, this);
    wl_display_roundtrip(m_display);

    if (!m_ext_list) {
        spdlog::error("Compositor does not support ext_foreign_toplevel_list_v1");
        return std::unexpected(WindowStateProviderError::WL_UNSUPPORTED_COMPOSITOR);
    }

    ext_foreign_toplevel_list_v1_add_listener(m_ext_list, &EXT_LIST_LISTENER, this);
    wl_display_roundtrip(m_display);

    return {};
}

WindowStateProvider::~WindowStateProvider() noexcept {
    wl_display_disconnect(m_display);
}


void WindowStateProvider::on_registry_global(
    void* data, wl_registry* registry, u32 name, const char* interface, u32 version)
{
    auto* self = static_cast<WindowStateProvider*>(data);

    if (std::string_view(interface) == "ext_foreign_toplevel_list_v1") {
        const u32 v = std::min<u32>(version, 1);
        self->m_ext_list = static_cast<ext_foreign_toplevel_list_v1*>(
            wl_registry_bind(registry, name, &ext_foreign_toplevel_list_v1_interface, v));
    }
}

void WindowStateProvider::on_registry_global_remove(void* /*data*/, wl_registry* /*registry*/, uint32_t /*name*/) {}

// ext_foreign_toplevel_list_v1 callbacks

void WindowStateProvider::on_list_toplevel(
    void* data, ext_foreign_toplevel_list_v1* /*list*/, ext_foreign_toplevel_handle_v1* handle)
{
    auto* self = static_cast<WindowStateProvider*>(data);

    CachedWindow cw;
    cw.ext_handle = handle;

    auto [it, inserted] = self->m_by_handle.emplace(handle, std::move(cw));
    if (!inserted) return;

    ext_foreign_toplevel_handle_v1_add_listener(handle, &EXT_HANDLE_LISTENER, self);
}

void WindowStateProvider::on_list_finished(
    void* data, ext_foreign_toplevel_list_v1* /*list*/)
{
    auto* self = static_cast<WindowStateProvider*>(data);
    self->m_initial_done = true;
}

// ext_foreign_toplevel_handle_v1 callbacks

void WindowStateProvider::on_handle_identifier(
    void* data, ext_foreign_toplevel_handle_v1* handle, const char* id)
{
    auto* self = static_cast<WindowStateProvider*>(data);
    auto it = self->m_by_handle.find(handle);
    if (it == self->m_by_handle.end()) return;

    it->second.info.window_id = id ? id : "";
    if (!it->second.info.window_id.empty()) {
        self->m_handle_by_id[it->second.info.window_id] = handle;
    }
}

void WindowStateProvider::on_handle_title(
    void* data, ext_foreign_toplevel_handle_v1* handle, const char* title)
{
    auto* self = static_cast<WindowStateProvider*>(data);
    auto it = self->m_by_handle.find(handle);
    if (it == self->m_by_handle.end()) return;

    it->second.info.title = title ? title : "";
}

void WindowStateProvider::on_handle_app_id(
    void* data, ext_foreign_toplevel_handle_v1* handle, const char* app_id)
{
    auto* self = static_cast<WindowStateProvider*>(data);
    auto it = self->m_by_handle.find(handle);
    if (it == self->m_by_handle.end()) return;

    it->second.info.app_id = app_id ? app_id : "";
}

void WindowStateProvider::on_handle_done(
    void* data, ext_foreign_toplevel_handle_v1* handle)
{
    auto* self = static_cast<WindowStateProvider*>(data);
    auto it = self->m_by_handle.find(handle);
    if (it == self->m_by_handle.end()) return;

    it->second.seen_done = true;
}

void WindowStateProvider::on_handle_closed(
    void* data, ext_foreign_toplevel_handle_v1* handle)
{
    auto* self = static_cast<WindowStateProvider*>(data);
    auto it = self->m_by_handle.find(handle);
    if (it == self->m_by_handle.end()) return;

    if (!it->second.info.window_id.empty()) {
        self->m_handle_by_id.erase(it->second.info.window_id);
    }
    self->m_by_handle.erase(it);

    ext_foreign_toplevel_handle_v1_destroy(handle);
}

void WindowStateProvider::pump_events() noexcept {
    if (!m_display) return;
    wl_display_dispatch_pending(m_display);
    // wl_display_roundtrip(m_display);
}

std::vector<WindowInfo> WindowStateProvider::get_open_windows() noexcept {
    pump_events();

    std::vector<WindowInfo> out;
    out.reserve(m_by_handle.size());
    for (auto& [_, cw] : m_by_handle) {
        out.push_back(cw.info);
    }
    return out;
}

std::optional<WindowInfo> WindowStateProvider::get_window_state(std::string_view window_id) noexcept {
    pump_events();

    auto it_handle = m_handle_by_id.find(std::string(window_id));
    if (it_handle == m_handle_by_id.end()) return std::nullopt;

    auto it = m_by_handle.find(it_handle->second);
    if (it == m_by_handle.end()) return std::nullopt;

    return it->second.info;
}
