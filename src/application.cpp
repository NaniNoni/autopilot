#include "application.hpp"
#include <print>

Application::Application() noexcept {
    if (!orchestrator.init()) {
        std::println("Failed to initialize orchestrator");
    }
}

Application::~Application() noexcept {}
