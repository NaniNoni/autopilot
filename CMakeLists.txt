cmake_minimum_required(VERSION 3.20)
project(autosktop LANGUAGES CXX)
include(FetchContent)

set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

FetchContent_Declare(llama_cpp
    GIT_REPOSITORY https://github.com/ggml-org/llama.cpp
    GIT_TAG b7789
)

FetchContent_Declare(nlohmann_json
    GIT_REPOSITORY https://github.com/nlohmann/json
    GIT_TAG v3.12.0
)

FetchContent_Declare(spdlog
    GIT_REPOSITORY https://github.com/gabime/spdlog
    GIT_TAG v1.17.0
)

add_executable(${CMAKE_PROJECT_NAME}
    src/main.cpp
    src/application.cpp
    src/orchestrator.cpp
)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

# TODO: set this per architecture??
set(GGML_CUDA ON)
FetchContent_MakeAvailable(llama_cpp nlohmann_json spdlog)

file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models)
include(cmake/download_model.cmake)
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    DEFAULT_ORCHESTRATOR_PATH="${MODEL_PATH}"
)

find_program(HYPRWAYLAND_SCANNER hyprwayland-scanner REQUIRED)
set(PROTOCOLS_DIR ${CMAKE_SOURCE_DIR}/protocols)
set(GEN_DIR ${CMAKE_BINARY_DIR}/generated/wayland)
file(MAKE_DIRECTORY ${GEN_DIR})

function(generate_protocol_bindings PROTO_NAME)
    set(${PROTO_NAME}_XML ${PROTOCOLS_DIR}/${PROTO_NAME}.xml)
    set(${PROTO_NAME}_GEN_CPP ${GEN_DIR}/${PROTO_NAME}.cpp)
    set(${PROTO_NAME}_GEN_HPP ${GEN_DIR}/${PROTO_NAME}.hpp)

    add_custom_command(
        OUTPUT ${${PROTO_NAME}_GEN_CPP} ${${PROTO_NAME}_GEN_HPP}
        COMMAND ${HYPRWAYLAND_SCANNER} --client --wayland-enums
            ${${PROTO_NAME}_XML} ${GEN_DIR}
        DEPENDS ${${PROTO_NAME}_XML}
        COMMENT "Generating protocol bindings for ${PROTO_NAME} (hyprwayland-scanner)"
        VERBATIM
    )

    add_library(${PROTO_NAME} STATIC
        ${${PROTO_NAME}_GEN_CPP} ${${PROTO_NAME}_GEN_HPP}
    )
    target_include_directories(${PROTO_NAME} PUBLIC ${GEN_DIR})
endfunction()

generate_protocol_bindings(cosmic-toplevel-management-unstable-v1)
generate_protocol_bindings(cosmic-toplevel-info-unstable-v1)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    llama
    nlohmann_json::nlohmann_json
    spdlog::spdlog

    # Wayland protocols
    cosmic-toplevel-management-unstable-v1
    cosmic-toplevel-info-unstable-v1
)
