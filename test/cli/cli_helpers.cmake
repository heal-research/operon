# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

# Helpers shared by the command-line test scripts that run operon_gp / operon_nsgp.

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
