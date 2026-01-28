set(MODEL_URL "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
set(MODEL_PATH "${CMAKE_SOURCE_DIR}/models/llama-3.1-8b-instruct.Q4_K_M.gguf")

set(MODEL_SHA256 "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c")

function(verify_model_sha256 path expected)
    if (EXISTS "${path}")
        message(STATUS "Checking model checksum")
        file(SHA256 "${path}" actual)
        if (NOT actual STREQUAL expected)
            message(FATAL_ERROR
                "Model checksum mismatch!\n"
                "  path: ${path}\n"
                "  expected: ${expected}\n"
                "  actual:   ${actual}\n"
                "Delete the file and reconfigure to re-download."
            )
        endif()
    endif()
endfunction()

if (NOT EXISTS "${MODEL_PATH}")
    message(STATUS "Downloading model...")
    file(DOWNLOAD
        "${MODEL_URL}"
        "${MODEL_PATH}"
        SHOW_PROGRESS
        STATUS status
        LOG log
    )
    list(GET status 0 code)
    if (NOT code EQUAL 0)
        message(FATAL_ERROR "Model download failed: ${log}")
    endif()
endif()

verify_model_sha256("${MODEL_PATH}" "${MODEL_SHA256}")
