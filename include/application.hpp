#pragma once

#include "orchestrator.hpp"

class Application {
public:
    Application() noexcept;
    ~Application() noexcept;

private:
    Orchestrator orchestrator;
};
