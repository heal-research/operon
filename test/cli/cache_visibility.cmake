# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

# Command-line contract of --cache-visibility for operon_gp and operon_nsgp:
#   - listed in --help, default committed;
#   - immediate/committed accepted with and without --transposition-cache;
#   - any other value is rejected with exit status 1 and a message naming the supported values;
#   - the committed default makes the final model and the evaluation counters independent of --threads.
#
# Usage: cmake -DOPERON_GP=<exe> -DOPERON_NSGP=<exe> -DDATA_DIR=<dir> [-DWITH_JIT=ON] -P cache_visibility.cmake

foreach(var OPERON_GP OPERON_NSGP DATA_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not set")
    endif()
endforeach()

set(common_args
    --dataset "${DATA_DIR}/Poly-10.csv" --target Y --train 0:100 --test 100:200
    --generations 4 --population-size 20 --pool-size 20 --seed 7 --evaluations 100000000)

# Larger and with brood selection, for the thread-count check: in immediate mode both programs then differ between
# thread counts in every run (a population of 20 with the default generator did so only in some runs of operon_gp).
set(determinism_args
    --dataset "${DATA_DIR}/Poly-10.csv" --target Y --train 0:100 --test 100:200
    --generations 6 --population-size 100 --pool-size 100 --offspring-generator brood:3
    --seed 7 --evaluations 100000000)

include("${CMAKE_CURRENT_LIST_DIR}/cli_helpers.cmake")

function(expect_rejected exe value)
    run_cli(run "${exe}" ${ARGN})
    if(NOT run_RC EQUAL 1)
        message(FATAL_ERROR "${exe} --cache-visibility=${value}\nexpected exit status 1, got ${run_RC}\n${run_OUT}")
    endif()
    if(NOT run_ERR MATCHES "cache-visibility" OR NOT run_ERR MATCHES "'${value}'"
            OR NOT run_ERR MATCHES "immediate, committed")
        message(FATAL_ERROR "${exe} --cache-visibility=${value}\nerror message does not name the option, the value "
            "and the supported values:\n${run_ERR}")
    endif()
endfunction()

foreach(exe IN ITEMS "${OPERON_GP}" "${OPERON_NSGP}")
    get_filename_component(name "${exe}" NAME)
    message(STATUS "${name}: --help lists the option and its default")
    run_cli(help "${exe}" --help)
    if(NOT help_RC EQUAL 0 OR NOT help_OUT MATCHES "--cache-visibility" OR NOT help_OUT MATCHES "default: committed")
        message(FATAL_ERROR "${name} --help must list --cache-visibility with default committed:\n${help_OUT}")
    endif()

    message(STATUS "${name}: accepted values")
    expect_success("${exe}" ${common_args} --threads 1 --transposition-cache)
    expect_success("${exe}" ${common_args} --threads 1 --transposition-cache --cache-visibility=committed)
    expect_success("${exe}" ${common_args} --threads 1 --transposition-cache --cache-visibility=immediate)
    expect_success("${exe}" ${common_args} --threads 1 --transposition-cache --cache-visibility=committed
        --cache-max-age 2)
    expect_success("${exe}" ${common_args} --threads 1 --cache-visibility=immediate) # accepted, no cache to apply it to
    if(WITH_JIT)
        expect_success("${exe}" ${common_args} --threads 1 --jit=all --transposition-cache
            --cache-visibility=immediate)
        expect_success("${exe}" ${common_args} --threads 1 --jit=all --transposition-cache
            --cache-visibility=committed)
    endif()

    message(STATUS "${name}: rejected values")
    foreach(bad IN ITEMS bogus Committed "")
        expect_rejected("${exe}" "${bad}" ${common_args} --threads 1 --transposition-cache "--cache-visibility=${bad}")
    endforeach()
    expect_rejected("${exe}" bogus ${common_args} --threads 1 --cache-visibility bogus) # validated without the cache
    if(WITH_JIT)
        expect_rejected("${exe}" bogus ${common_args} --threads 1 --jit=all --transposition-cache
            --cache-visibility=bogus)
    endif()

    message(STATUS "${name}: committed result does not depend on --threads")
    final_state(one "${exe}" ${determinism_args} --threads 1 --transposition-cache --cache-visibility=committed)
    final_state(four "${exe}" ${determinism_args} --threads 4 --transposition-cache --cache-visibility=committed)
    final_state(default_four "${exe}" ${determinism_args} --threads 4 --transposition-cache)
    if(NOT one STREQUAL four)
        message(FATAL_ERROR "${name}: committed runs differ between 1 and 4 threads:\n${one}\n${four}")
    endif()
    if(NOT one STREQUAL default_four)
        message(FATAL_ERROR "${name}: the default is not committed:\n${one}\n${default_four}")
    endif()
endforeach()
