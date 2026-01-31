set(MODEL_URL "https://huggingface.co/bartowski/Meta-Llama-3.1-8B-Instruct-GGUF/resolve/main/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf")
set(MODEL_PATH "${CMAKE_SOURCE_DIR}/models/llama-3.1-8b-instruct.Q4_K_M.gguf")
set(MODEL_SHA256 "7b064f5842bf9532c91456deda288a1b672397a54fa729aa665952863033557c")

# record that a given model file has been verified successfully.
set(MODEL_SHA_STAMP "${CMAKE_BINARY_DIR}/.model_sha256_ok.stamp")

function(download_model model_path model_url)
    if (NOT EXISTS "${model_path}")
        message(STATUS "Downloading model...")
        file(DOWNLOAD
            "${model_url}"
            "${model_path}"
            SHOW_PROGRESS
            STATUS status
            LOG log
        )
        list(GET status 0 code)
        if (NOT code EQUAL 0)
            message(FATAL_ERROR "Model download failed: ${log}")
        endif()

        # Model just changed, so verify
        if (EXISTS "${MODEL_SHA_STAMP}")
            file(REMOVE "${MODEL_SHA_STAMP}")
        endif()
    endif()

    verify_model_sha256_once("${model_path}" "${MODEL_SHA256}" "${MODEL_SHA_STAMP}")
endfunction()

# Verify checksum only when needed
function(verify_model_sha256_once path expected stamp)
    if (NOT EXISTS "${path}")
        message(FATAL_ERROR "Model file not found: ${path}")
    endif()

    set(need_check TRUE)
    if (EXISTS "${stamp}")
        file(TIMESTAMP "${path}" model_ts "%s")
        file(TIMESTAMP "${stamp}" stamp_ts "%s")
        if (model_ts LESS_EQUAL stamp_ts)
            set(need_check FALSE)
        endif()
    endif()

    if (need_check)
        message(STATUS "Checking model checksum (SHA256)")
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

        # Record successful verification
        file(WRITE "${stamp}" "ok\n")
        message(STATUS "Model checksum OK (stamp updated)")
    else()
        message(STATUS "Skipping model checksum (already verified)")
    endif()
endfunction()
