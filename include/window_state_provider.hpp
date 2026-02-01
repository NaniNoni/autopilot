#pragma once

#include <optional>
#include <string>
#include <string_view>
#include <unordered_map>
#include <vector>
#include <expected>

#include <wayland-client-protocol.h>
#include <wayland-client.h>
#include <ext-foreign-toplevel-list-v1-client-protocol.h>

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
    wl_display* m_display { nullptr };
    wl_registry* m_registry { nullptr };
    void pump_events() noexcept;

    ext_foreign_toplevel_list_v1* m_ext_list { nullptr };
    bool m_initial_done { false };

    struct CachedWindow {
        WindowInfo info {};
        ext_foreign_toplevel_handle_v1* ext_handle { nullptr };
        bool seen_done { false };
    };

    std::unordered_map<ext_foreign_toplevel_handle_v1*, CachedWindow> m_by_handle;
    std::unordered_map<std::string, ext_foreign_toplevel_handle_v1*> m_handle_by_id;

    // Wayland registry callbacks
    static void on_registry_global(
        void* data, wl_registry* registry, uint32_t name, const char* interface, uint32_t version);
    static void on_registry_global_remove(
        void* data, wl_registry* registry, uint32_t name);

    static constexpr wl_registry_listener REGISTRY_LISTENER = {
        .global = on_registry_global,
        .global_remove = on_registry_global_remove
    };

    // ext_foreign_toplevel_list_v1 callbacks
    static void on_list_toplevel(
        void* data, ext_foreign_toplevel_list_v1* list, ext_foreign_toplevel_handle_v1* handle);
    static void on_list_finished(
        void* data, ext_foreign_toplevel_list_v1* list);

    static constexpr ext_foreign_toplevel_list_v1_listener EXT_LIST_LISTENER = {
        .toplevel = on_list_toplevel,
        .finished = on_list_finished
    };

    // ext_foreign_toplevel_handle_v1 callbacks
    static void on_handle_identifier(
        void* data, ext_foreign_toplevel_handle_v1* handle, const char* id);
    static void on_handle_title(
        void* data, ext_foreign_toplevel_handle_v1* handle, const char* title);
    static void on_handle_app_id(
        void* data, ext_foreign_toplevel_handle_v1* handle, const char* app_id);
    static void on_handle_done(
        void* data, ext_foreign_toplevel_handle_v1* handle);
    static void on_handle_closed(
        void* data, ext_foreign_toplevel_handle_v1* handle);

    static constexpr ext_foreign_toplevel_handle_v1_listener EXT_HANDLE_LISTENER = {
        .closed = on_handle_closed,
        .done = on_handle_done,
        .title = on_handle_title,
        .app_id = on_handle_app_id,
        .identifier = on_handle_identifier,
    };
};
