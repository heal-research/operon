# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#
# --capabilities must report the configuration that was built and its flags: build_type must equal the built
# configuration, the flags that only the Debug configuration sets must be absent from any other configuration, and the
# flags that do apply (the global and per-configuration CMake flags followed by the compile options of the operon_operon
# target) must be reported exactly.
#   -DCLI=<executable> -DCONFIG=<built configuration> -DDEBUG_FLAGS=<CMAKE_CXX_FLAGS_DEBUG> -DACTIVE_FLAGS=<flags
#   applied to the built configuration: CMAKE_CXX_FLAGS plus CMAKE_CXX_FLAGS_<CONFIG>>
#   -DTARGET_OPTIONS_FILE=<file holding the COMPILE_OPTIONS of operon_operon joined with '|'>
execute_process(
    COMMAND "${CLI}" --capabilities
    RESULT_VARIABLE status
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
)
if(NOT status STREQUAL "0")
    message(FATAL_ERROR "--capabilities exited with '${status}'\n${err}")
endif()
string(JSON reported ERROR_VARIABLE jsonError GET "${out}" compiler flags)
if(jsonError)
    message(FATAL_ERROR "no compiler.flags in the capabilities output: ${jsonError}\n${out}")
endif()
string(JSON buildType ERROR_VARIABLE jsonError GET "${out}" build_type)
if(jsonError)
    message(FATAL_ERROR "no build_type in the capabilities output: ${jsonError}\n${out}")
endif()
if(NOT "${buildType}" STREQUAL "${CONFIG}")
    message(FATAL_ERROR "build_type is '${buildType}' but the ${CONFIG} configuration was built")
endif()

file(READ "${TARGET_OPTIONS_FILE}" targetOptions)
string(REPLACE "|" " " targetOptions "${targetOptions}")
string(REGEX REPLACE "[ \t]+" ";" targetTokens "${targetOptions}")
list(FILTER targetTokens EXCLUDE REGEX "^$")
if(NOT targetTokens)
    message(FATAL_ERROR "the operon_operon target has no compile options to check (${TARGET_OPTIONS_FILE})")
endif()
string(REGEX REPLACE "[ \t]+" ";" reportedTokens "${reported}")
string(REGEX REPLACE "[ \t]+" ";" activeTokens "${ACTIVE_FLAGS}")
string(REGEX REPLACE "[ \t]+" ";" debugTokens "${DEBUG_FLAGS}")

# The reported flags are exactly CMAKE_CXX_FLAGS, the flags of the built configuration and the compile options of
# operon_operon, in that order. Comparing the whole sequence (not only membership) also catches a target option that is
# dropped but happens to be repeated in CMAKE_CXX_FLAGS.
set(expectedTokens ${activeTokens} ${targetTokens})
list(FILTER expectedTokens EXCLUDE REGEX "^$")
list(FILTER reportedTokens EXCLUDE REGEX "^$")
if(NOT expectedTokens STREQUAL reportedTokens)
    message(FATAL_ERROR "the ${CONFIG} flags are not CMAKE_CXX_FLAGS + CMAKE_CXX_FLAGS_<CONFIG> + the compile options "
        "of operon_operon\nexpected: ${expectedTokens}\nreported: ${reportedTokens}")
endif()

if(NOT CONFIG STREQUAL "Debug")
    foreach(token IN LISTS debugTokens)
        if(NOT token STREQUAL "" AND NOT token IN_LIST activeTokens AND token IN_LIST reportedTokens)
            message(FATAL_ERROR "Debug-only flag '${token}' is reported for the ${CONFIG} build: '${reported}'")
        endif()
    endforeach()
endif()
message(STATUS "compiler flags of the ${CONFIG} build: ${reported}")
