#include "window_state_provider.hpp"
#include "ext-foreign-toplevel-list-v1-client-protocol.h"
#include "int_types.hpp"
#include "state_provider.hpp"
#include "state_request.hpp"

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

nlohmann::json WindowStateProvider::processRequest(StateRequest req) noexcept {
    nlohmann::json out;

    if (req.kind != StateProviderKind::WINDOW) {
        out["ok"] = false;
        out["error"] = "wrong_provider_kind";
        return out;
    }

    try {
        if (!req.args.is_object()) {
            out["ok"] = false;
            out["error"] = "args_not_object";
            return out;
        }

        if (!req.args.contains("action") || !req.args["action"].is_string()) {
            out["ok"] = false;
            out["error"] = "missing_or_invalid_action";
            return out;
        }

        const std::string action = req.args["action"].get<std::string>();

        const nlohmann::json params =
            (req.args.contains("params") && req.args["params"].is_object())
                ? req.args["params"]
                : nlohmann::json::object();

        if (action == "get_open_windows") {
            const auto windows = get_open_windows();

            nlohmann::json arr = nlohmann::json::array();
            for (const auto& w : windows) {
                arr.push_back({
                    {"window_id", w.window_id},
                    {"title",     w.title},
                    {"app_id",    w.app_id},
                });
            }

            out["ok"] = true;
            out["action"] = action;
            out["windows"] = std::move(arr);
            return out;
        }

        if (action == "get_window_state") {
            if (!params.contains("window_id") || !params["window_id"].is_string()) {
                out["ok"] = false;
                out["action"] = action;
                out["error"] = "missing_or_invalid_window_id";
                return out;
            }

            const std::string window_id = params["window_id"].get<std::string>();
            const auto info = get_window_state(window_id);

            if (!info) {
                out["ok"] = false;
                out["action"] = action;
                out["error"] = "not_found";
                out["window_id"] = window_id;
                return out;
            }

            out["ok"] = true;
            out["action"] = action;
            out["window"] = {
                {"window_id", info->window_id},
                {"title",     info->title},
                {"app_id",    info->app_id},
            };
            return out;
        }

        // Unknown action
        out["ok"] = false;
        out["error"] = "unknown_action";
        out["action"] = action;
        return out;
    }
    catch (const std::exception& e) {
        spdlog::error("WindowStateProvider::processRequest error: {}", e.what());
        out["ok"] = false;
        out["error"] = "exception";
        return out;
    }
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
