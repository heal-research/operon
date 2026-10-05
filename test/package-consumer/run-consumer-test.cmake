# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#
# CTest driver (`cmake -P`) for the installed-package-consumer contract.
#
# Runs, in order, the four steps an out-of-tree consumer of operon actually
# performs: install the package this build just produced into a staging
# prefix, relocate that prefix to an unrelated path (proving nothing in the
# installed tree hard-codes the staging location), configure the standalone
# consumer project against ONLY the relocated prefix via CMAKE_PREFIX_PATH,
# then build and run it. Any step failing aborts immediately with the
# offending command's captured output.
cmake_minimum_required(VERSION 3.25)

foreach(var
    CMAKE_COMMAND
    OPERON_MAIN_BUILD_DIR
    OPERON_STAGE_DIR
    OPERON_RELOCATED_PREFIX
    OPERON_CONSUMER_SOURCE_DIR
    OPERON_CONSUMER_BUILD_DIR
)
    if(NOT DEFINED ${var} OR "${${var}}" STREQUAL "")
        message(FATAL_ERROR "run-consumer-test.cmake: ${var} must be passed via -D")
    endif()
endforeach()

function(run_step description)
    execute_process(
        COMMAND ${ARGN}
        RESULT_VARIABLE step_result
        OUTPUT_VARIABLE step_output
        ERROR_VARIABLE step_error
    )
    if(NOT step_result EQUAL 0)
        message(FATAL_ERROR
            "package-consumer: ${description} failed (exit ${step_result})\n"
            "--- stdout ---\n${step_output}\n"
            "--- stderr ---\n${step_error}"
        )
    endif()
    message(STATUS "package-consumer: ${description} OK")
endfunction()

# Start from a clean slate: re-running ctest must not observe stale state
# from a previous run (a previously relocated prefix, a previous consumer
# build cache pointing at now-invalid paths, ...).
file(REMOVE_RECURSE "${OPERON_STAGE_DIR}")
file(REMOVE_RECURSE "${OPERON_RELOCATED_PREFIX}")
file(REMOVE_RECURSE "${OPERON_CONSUMER_BUILD_DIR}")

# A multi-config parent build must install and exercise the same configuration
# CTest selected. Single-config generators expand $<CONFIG> to an empty string,
# for which omitting --config preserves CMake's normal behavior. CMake uses
# --config for installation/building; CTest spells the equivalent option -C.
set(operon_cmake_config_args)
set(operon_ctest_config_args)
if(DEFINED OPERON_CONFIGURATION AND NOT "${OPERON_CONFIGURATION}" STREQUAL "")
    list(APPEND operon_cmake_config_args --config "${OPERON_CONFIGURATION}")
    list(APPEND operon_ctest_config_args -C "${OPERON_CONFIGURATION}")
endif()

# Preserve the parent's toolchain, build mode, and flags when required. In
# particular, a statically linked sanitizer build must link its consumer with
# the same sanitizer runtime; a fresh consumer configuration does not inherit
# those settings automatically.
set(operon_consumer_config_args)
foreach(var
    OPERON_TOOLCHAIN_FILE
    OPERON_VCPKG_TARGET_TRIPLET
    OPERON_VCPKG_INSTALLED_DIR
    OPERON_MSVC_RUNTIME_LIBRARY
    OPERON_CMAKE_C_COMPILER
    OPERON_CMAKE_CXX_COMPILER
    OPERON_CMAKE_C_FLAGS
    OPERON_CMAKE_CXX_FLAGS
    OPERON_CMAKE_EXE_LINKER_FLAGS
    OPERON_CMAKE_SHARED_LINKER_FLAGS
    OPERON_CMAKE_BUILD_TYPE
)
    if(DEFINED ${var} AND NOT "${${var}}" STREQUAL "")
        if(var STREQUAL "OPERON_TOOLCHAIN_FILE")
            set(cmake_var CMAKE_TOOLCHAIN_FILE)
        elseif(var STREQUAL "OPERON_MSVC_RUNTIME_LIBRARY")
            set(cmake_var CMAKE_MSVC_RUNTIME_LIBRARY)
        else()
            string(REPLACE "OPERON_" "" cmake_var "${var}")
        endif()
        list(APPEND operon_consumer_config_args "-D${cmake_var}=${${var}}")
    endif()
endforeach()

# 1. Install only the library package components. CI builds operon_test, not
# the optional CLI executables, whose independent install component would
# otherwise make this contract test fail before reaching the consumer.
run_step("install runtime"
    "${CMAKE_COMMAND}" --install "${OPERON_MAIN_BUILD_DIR}" ${operon_cmake_config_args} --component operon_Runtime --prefix "${OPERON_STAGE_DIR}"
)
run_step("install development"
    "${CMAKE_COMMAND}" --install "${OPERON_MAIN_BUILD_DIR}" ${operon_cmake_config_args} --component operon_Development --prefix "${OPERON_STAGE_DIR}"
)

# 2. Relocate the staged prefix. A consumer never sees OPERON_STAGE_DIR;
# only the moved-to path is handed to the fixture below. If any exported
# file baked an absolute path back to the staging (or, worse, the operon
# source/build tree) location, this rename makes that surface as a
# find_package or link failure instead of silently passing.
file(RENAME "${OPERON_STAGE_DIR}" "${OPERON_RELOCATED_PREFIX}")

# 3. Configure the fixture with the relocated prefix as its only package
# location. Forwarded parent settings preserve dependency discovery,
# compiler/runtime ABI, and instrumentation without exposing the build tree.
run_step("configure"
    "${CMAKE_COMMAND}"
        -S "${OPERON_CONSUMER_SOURCE_DIR}"
        -B "${OPERON_CONSUMER_BUILD_DIR}"
        "-DCMAKE_PREFIX_PATH=${OPERON_RELOCATED_PREFIX}"
        ${operon_consumer_config_args}
)

# 4. Build it in the selected configuration, when applicable.
run_step("build"
    "${CMAKE_COMMAND}" --build "${OPERON_CONSUMER_BUILD_DIR}" ${operon_cmake_config_args}
)

# 5. Run it via CTest. `-C`, rather than CMake's `--config`, selects a
# multi-config CTest configuration.
run_step("run"
    "${CMAKE_COMMAND}" -E env --unset=CTEST_OUTPUT_ON_FAILURE
    ctest --test-dir "${OPERON_CONSUMER_BUILD_DIR}" ${operon_ctest_config_args} --output-on-failure
)
