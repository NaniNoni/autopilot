cmake_minimum_required(VERSION 3.20)
project(autosktop LANGUAGES CXX C)
include(FetchContent)

find_package(PkgConfig REQUIRED)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# --- deps ---
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

# TODO: set this per architecture??
set(GGML_CUDA ON)
FetchContent_MakeAvailable(llama_cpp nlohmann_json spdlog)

add_executable(${CMAKE_PROJECT_NAME}
    src/main.cpp
    src/application.cpp
    src/orchestrator.cpp
    src/window_state_provider.cpp
    src/state_request.cpp
)
target_include_directories(${CMAKE_PROJECT_NAME} PRIVATE include)

set_target_properties(${CMAKE_PROJECT_NAME} PROPERTIES
  CXX_STANDARD 23
  CXX_STANDARD_REQUIRED YES
  CXX_EXTENSIONS NO
)

# model path define
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/models)
include(cmake/download_model.cmake)
download_model(${MODEL_PATH} ${MODEL_URL})
target_compile_definitions(${CMAKE_PROJECT_NAME} PRIVATE
    DEFAULT_ORCHESTRATOR_PATH="${MODEL_PATH}"
)

pkg_check_modules(WAYLAND REQUIRED IMPORTED_TARGET wayland-client)
find_program(WAYLAND_SCANNER wayland-scanner REQUIRED)

set(PROTOCOLS_DIR ${CMAKE_SOURCE_DIR}/protocols)
set(GEN_DIR ${CMAKE_BINARY_DIR}/generated/wayland)
file(MAKE_DIRECTORY ${GEN_DIR})

# Generate protocol C + headers using wayland-scanner
function(generate_protocol_bindings PROTO_NAME)
    set(XML_FILE "${PROTOCOLS_DIR}/${PROTO_NAME}.xml")

    set(CLIENT_HDR "${GEN_DIR}/${PROTO_NAME}-client-protocol.h")
    set(ENUM_HDR   "${GEN_DIR}/${PROTO_NAME}-protocol-enum.h")
    set(C_CODE     "${GEN_DIR}/${PROTO_NAME}-protocol.c")

    add_custom_command(
        OUTPUT "${CLIENT_HDR}"
        COMMAND "${WAYLAND_SCANNER}" client-header "${XML_FILE}" "${CLIENT_HDR}"
        DEPENDS "${XML_FILE}"
        COMMENT "wayland-scanner client-header ${PROTO_NAME}"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${ENUM_HDR}"
        COMMAND "${WAYLAND_SCANNER}" enum-header "${XML_FILE}" "${ENUM_HDR}"
        DEPENDS "${XML_FILE}"
        COMMENT "wayland-scanner enum-header ${PROTO_NAME}"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${C_CODE}"
        COMMAND "${WAYLAND_SCANNER}" private-code "${XML_FILE}" "${C_CODE}"
        DEPENDS "${XML_FILE}"
        COMMENT "wayland-scanner private-code ${PROTO_NAME}"
        VERBATIM
    )

    add_library(${PROTO_NAME} STATIC
        "${C_CODE}"
        "${CLIENT_HDR}"
        "${ENUM_HDR}"
    )

    target_include_directories(${PROTO_NAME} PUBLIC "${GEN_DIR}")
    target_link_libraries(${PROTO_NAME} PUBLIC PkgConfig::WAYLAND)
endfunction()

generate_protocol_bindings(cosmic-toplevel-management-unstable-v1)
generate_protocol_bindings(cosmic-toplevel-info-unstable-v1)
generate_protocol_bindings(ext-foreign-toplevel-list-v1)
generate_protocol_bindings(ext-workspace-v1)

target_link_libraries(${CMAKE_PROJECT_NAME} PRIVATE
    llama
    nlohmann_json::nlohmann_json
    spdlog::spdlog

    PkgConfig::WAYLAND

    cosmic-toplevel-management-unstable-v1
    cosmic-toplevel-info-unstable-v1
    ext-foreign-toplevel-list-v1
    ext-workspace-v1
)
