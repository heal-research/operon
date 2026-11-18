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

# Runs `exe` with `args`; sets <prefix>_RC, <prefix>_OUT and <prefix>_ERR in the caller.
function(run_cli prefix exe)
    execute_process(COMMAND "${exe}" ${ARGN}
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err
        TIMEOUT 300)
    set(${prefix}_RC "${rc}" PARENT_SCOPE)
    set(${prefix}_OUT "${out}" PARENT_SCOPE)
    set(${prefix}_ERR "${err}" PARENT_SCOPE)
endfunction()

function(expect_success exe)
    run_cli(run "${exe}" ${ARGN})
    if(NOT run_RC EQUAL 0)
        message(FATAL_ERROR "${exe} ${ARGN}\nexpected exit status 0, got ${run_RC}\n${run_ERR}")
    endif()
endfunction()

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

# Report columns that hold wall-clock time; every other column of the final report row must match.
set(timing_columns opt_time sort_ms elapsed)

# The model printed last, plus every report column of the final row except the timing ones, keyed by the name in the
# header row.
function(final_state out exe)
    run_cli(run "${exe}" ${ARGN})
    if(NOT run_RC EQUAL 0)
        message(FATAL_ERROR "${exe} ${ARGN}\nexpected exit status 0, got ${run_RC}\n${run_ERR}")
    endif()
    string(STRIP "${run_OUT}" stripped)
    string(REPLACE "\n" ";" lines "${stripped}")
    list(LENGTH lines n)
    math(EXPR last_row "${n} - 2")
    list(GET lines 0 header)
    list(GET lines ${last_row} row)
    list(GET lines -1 model)
    string(REGEX MATCHALL "[^ ]+" names "${header}")
    string(REGEX MATCHALL "[^ ]+" columns "${row}")
    list(LENGTH names n_names)
    list(LENGTH columns n_columns)
    if(NOT n_names EQUAL n_columns OR NOT "eval_cnt" IN_LIST names OR NOT "best_fit" IN_LIST names)
        message(FATAL_ERROR "${exe}: cannot match the report columns to the header:\n${header}\n${row}")
    endif()
    set(state "${model}")
    foreach(name column IN ZIP_LISTS names columns)
        if(NOT name IN_LIST timing_columns)
            string(APPEND state "|${name}=${column}")
        endif()
    endforeach()
    set(${out} "${state}" PARENT_SCOPE)
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
