# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

# `--resume` continues a run exactly like the uninterrupted run: stop at generation G (checkpoint), resume to
# generation 6, and compare with a run that went to 6 without interruption. The final checkpoints (parent population,
# master and worker random states, transposition cache entries and clock) must be byte for byte the same file, and so
# must the final model and the population columns of the last report row. Runs without a cache, with the cache in
# both visibility modes and with a maximum entry age are covered, for GP and NSGA-II. Also: a mismatch between the
# cache options of the checkpoint and of the resumed run is rejected, as is a cache-less checkpoint resumed with a
# cache; a finite --generations below the checkpoint's generation is rejected (equal, greater and unlimited are
# accepted); the content of --shape-constraints-config (not its path) is part of the fingerprint.
#
# Usage: cmake -DOPERON_GP=<exe> -DOPERON_NSGP=<exe> -DDATA_DIR=<dir> -DWORK_DIR=<dir> -P checkpoint_resume.cmake

foreach(var OPERON_GP OPERON_NSGP DATA_DIR WORK_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not set")
    endif()
endforeach()
file(MAKE_DIRECTORY "${WORK_DIR}")

set(base_args
    --dataset "${DATA_DIR}/Poly-10.csv" --target Y --train 0:100 --test 100:200
    --population-size 20 --pool-size 20 --seed 7 --evaluations 100000000 --iterations 3)
set(total_generations 6)

# Report columns that depend on how the process was started (evaluation counters restart at zero when a run is
# resumed, timings are wall clock); every other column of the last row is compared.
set(process_columns eval_cnt res_eval jac_eval opt_time sort_ms elapsed)

function(run_cli prefix exe)
    execute_process(COMMAND "${exe}" ${ARGN}
        RESULT_VARIABLE rc OUTPUT_VARIABLE out ERROR_VARIABLE err
        TIMEOUT 300)
    set(${prefix}_RC "${rc}" PARENT_SCOPE)
    set(${prefix}_OUT "${out}" PARENT_SCOPE)
    set(${prefix}_ERR "${err}" PARENT_SCOPE)
endfunction()

# The model printed last plus every non-process column of the last report row.
# (A resumed run does not print the header row, so the column names come from `header_text`, the output of a run
# that did.)
function(final_state out text header_text)
    string(STRIP "${header_text}" header_stripped)
    string(REPLACE "\n" ";" header_lines "${header_stripped}")
    list(GET header_lines 0 header)
    string(STRIP "${text}" stripped)
    string(REPLACE "\n" ";" lines "${stripped}")
    list(LENGTH lines n)
    math(EXPR last_row "${n} - 2")
    list(GET lines ${last_row} row)
    list(GET lines -1 model)
    string(REGEX MATCHALL "[^ ]+" names "${header}")
    string(REGEX MATCHALL "[^ ]+" columns "${row}")
    set(state "${model}")
    foreach(name column IN ZIP_LISTS names columns)
        if(NOT name IN_LIST process_columns)
            string(APPEND state "|${name}=${column}")
        endif()
    endforeach()
    set(${out} "${state}" PARENT_SCOPE)
endfunction()

function(run_ok prefix exe)
    run_cli(${prefix} "${exe}" ${ARGN})
    if(NOT ${prefix}_RC EQUAL 0)
        message(FATAL_ERROR "${exe} ${ARGN}\nexpected exit status 0, got ${${prefix}_RC}\n${${prefix}_ERR}")
    endif()
    set(${prefix}_OUT "${${prefix}_OUT}" PARENT_SCOPE)
    set(${prefix}_ERR "${${prefix}_ERR}" PARENT_SCOPE)
endfunction()

# case: <label>;<threads>;<extra args joined with |>
function(check_resume exe label threads extra)
    string(REPLACE "|" ";" extra "${extra}")
    get_filename_component(name "${exe}" NAME)
    foreach(checkpoint_at IN ITEMS 2 3)
        set(tag "${name}-${label}-t${threads}-g${checkpoint_at}")
        message(STATUS "${tag}")
        set(full "${WORK_DIR}/${tag}-full.beve")
        set(part "${WORK_DIR}/${tag}-part.beve")
        set(resumed "${WORK_DIR}/${tag}-resumed.beve")
        file(REMOVE "${full}" "${part}" "${resumed}")
        set(args ${base_args} --threads ${threads} ${extra})

        run_ok(whole "${exe}" ${args} --generations ${total_generations}
            --checkpoint-interval ${total_generations} --checkpoint-file "${full}")
        run_ok(first "${exe}" ${args} --generations ${checkpoint_at}
            --checkpoint-interval ${checkpoint_at} --checkpoint-file "${part}")
        run_ok(second "${exe}" ${args} --generations ${total_generations} --resume "${part}"
            --checkpoint-interval ${total_generations} --checkpoint-file "${resumed}")

        file(SHA256 "${full}" full_hash)
        file(SHA256 "${resumed}" resumed_hash)
        if(NOT full_hash STREQUAL resumed_hash)
            message(FATAL_ERROR "${tag}: the final checkpoint of the resumed run differs from the uninterrupted one")
        endif()
        final_state(whole_state "${whole_OUT}" "${whole_OUT}")
        final_state(second_state "${second_OUT}" "${whole_OUT}")
        if(NOT whole_state STREQUAL second_state)
            message(FATAL_ERROR "${tag}: the resumed run ends differently:\n${whole_state}\n${second_state}")
        endif()
    endforeach()
endfunction()

foreach(exe IN ITEMS "${OPERON_GP}" "${OPERON_NSGP}")
    check_resume("${exe}" nocache 1 "")
    check_resume("${exe}" nocache 4 "")
    check_resume("${exe}" committed 1 "--transposition-cache")
    check_resume("${exe}" committed 4 "--transposition-cache")
    check_resume("${exe}" committed-brood 4 "--transposition-cache|--offspring-generator|brood:3")
    check_resume("${exe}" committed-age 4 "--transposition-cache|--cache-max-age|2")
    check_resume("${exe}" immediate 1 "--transposition-cache|--cache-visibility|immediate")
endforeach()

# ---- the checkpoint carries the cache: a cache that has entries makes the file larger than a cache-less one ----
set(plain "${WORK_DIR}/plain.beve")
set(cached "${WORK_DIR}/cached.beve")
run_ok(plain_run "${OPERON_GP}" ${base_args} --threads 1 --generations 4 --checkpoint-interval 4 --checkpoint-file "${plain}")
run_ok(cached_run "${OPERON_GP}" ${base_args} --threads 1 --generations 4 --transposition-cache
    --checkpoint-interval 4 --checkpoint-file "${cached}")
file(SIZE "${plain}" plain_size)
file(SIZE "${cached}" cached_size)
if(NOT cached_size GREATER plain_size)
    message(FATAL_ERROR "the checkpoint of a run with a cache (${cached_size} bytes) is not larger than without "
        "(${plain_size} bytes)")
endif()

# ---- mismatched options are rejected ----
function(expect_resume_rejected exe checkpoint pattern)
    run_cli(rejected "${exe}" ${base_args} --threads 1 --generations ${total_generations} --resume "${checkpoint}" ${ARGN})
    if(NOT rejected_RC EQUAL 1 OR NOT rejected_ERR MATCHES "${pattern}")
        message(FATAL_ERROR "${exe} ${ARGN}: expected exit status 1 and /${pattern}/, got ${rejected_RC}\n"
            "${rejected_ERR}")
    endif()
endfunction()

set(committed_checkpoint "${WORK_DIR}/committed-checkpoint.beve")
run_ok(make "${OPERON_GP}" ${base_args} --threads 1 --generations 3 --transposition-cache --cache-max-age 2
    --checkpoint-interval 3 --checkpoint-file "${committed_checkpoint}")
expect_resume_rejected("${OPERON_GP}" "${committed_checkpoint}" "--cache-visibility=committed but this run uses --cache-visibility=immediate"
    --transposition-cache --cache-max-age 2 --cache-visibility immediate)
expect_resume_rejected("${OPERON_GP}" "${committed_checkpoint}" "--cache-max-age=2 but this run uses --cache-max-age=5"
    --transposition-cache --cache-max-age 5)
expect_resume_rejected("${OPERON_GP}" "${committed_checkpoint}" "carries a transposition cache: resume with --transposition-cache"
    --cache-max-age 2)
# another seed builds another random table: the keys would mean something else
set(nsgp_checkpoint "${WORK_DIR}/nsgp-checkpoint.beve")
run_ok(make_nsgp "${OPERON_NSGP}" ${base_args} --threads 1 --generations 3 --transposition-cache --cache-max-age 2
    --checkpoint-interval 3 --checkpoint-file "${nsgp_checkpoint}")
expect_resume_rejected("${OPERON_NSGP}" "${nsgp_checkpoint}" "different random table"
    --transposition-cache --cache-max-age 2 --seed 8)

# ---- the run fingerprint: a checkpoint is only resumed by a run that computes what it computed ----
# (checked before the cache is imported or any fitness is restored; the message names what differs)
expect_resume_rejected("${OPERON_GP}" "${committed_checkpoint}" "differing: .*--objective"
    --transposition-cache --cache-max-age 2 --objective mse)
expect_resume_rejected("${OPERON_GP}" "${committed_checkpoint}" "differing: .*--offspring-generator"
    --transposition-cache --cache-max-age 2 --offspring-generator brood:3)
expect_resume_rejected("${OPERON_GP}" "${committed_checkpoint}" "differing: .*dataset.train"
    --transposition-cache --cache-max-age 2 --train 0:90)
expect_resume_rejected("${OPERON_GP}" "${committed_checkpoint}" "differing: .*--iterations"
    --transposition-cache --cache-max-age 2 --iterations 5)
# the same options with a different dataset content (a digit inserted into the first value of the first row, which
# starts with a minus sign: -0.99... becomes -70.99...)
file(READ "${DATA_DIR}/Poly-10.csv" csv)
string(FIND "${csv}" "\n" first_newline)
math(EXPR after_newline "${first_newline} + 2")
string(SUBSTRING "${csv}" 0 ${after_newline} csv_head)
string(SUBSTRING "${csv}" ${after_newline} -1 csv_tail)
file(WRITE "${WORK_DIR}/modified.csv" "${csv_head}7${csv_tail}")
expect_resume_rejected("${OPERON_GP}" "${committed_checkpoint}" "differing: .*dataset.content"
    --transposition-cache --cache-max-age 2 --dataset "${WORK_DIR}/modified.csv")
# nsgp checkpoints differ from gp ones in the options of the program
expect_resume_rejected("${OPERON_NSGP}" "${committed_checkpoint}" "differing: .*--sorter"
    --transposition-cache --cache-max-age 2)
# options that may change on resume do not matter: a larger thread count, other generation limit, other seed without
# a cache (the plain checkpoint has no cache)
run_ok(resumed_ok "${OPERON_GP}" ${base_args} --threads 4 --generations 6 --resume "${plain}" --seed 99 --timelimit 1000)
# the dataset may come from another path (its content is what is compared)
file(WRITE "${WORK_DIR}/copy-of-poly-10.csv" "${csv}")
run_ok(other_path "${OPERON_GP}" ${base_args} --dataset "${WORK_DIR}/copy-of-poly-10.csv" --threads 1
    --generations 6 --resume "${plain}")

# ---- --generations against the generation of the checkpoint (the plain checkpoints are at generation 4) ----
# ---- and a cache-less checkpoint is not resumed with a cache (the cache-bearing one is not resumed without, above) ----
foreach(exe IN ITEMS "${OPERON_GP}" "${OPERON_NSGP}")
    get_filename_component(name "${exe}" NAME)
    set(own_plain "${WORK_DIR}/${name}-plain.beve")
    run_ok(make_plain "${exe}" ${base_args} --threads 1 --generations 4 --checkpoint-interval 4
        --checkpoint-file "${own_plain}")
    run_ok(limit_equal "${exe}" ${base_args} --threads 1 --generations 4 --resume "${own_plain}")
    run_ok(limit_greater "${exe}" ${base_args} --threads 1 --generations 7 --resume "${own_plain}")
    run_ok(limit_unlimited "${exe}" ${base_args} --threads 1 --generations unlimited --timelimit 1
        --resume "${own_plain}")
    run_cli(limit_lower "${exe}" ${base_args} --threads 1 --generations 3 --resume "${own_plain}")
    if(NOT limit_lower_RC EQUAL 1 OR NOT limit_lower_ERR MATCHES "is at generation 4 but --generations is 3")
        message(FATAL_ERROR "${exe}: a --generations below the checkpoint's generation must be rejected: "
            "${limit_lower_RC}\n${limit_lower_ERR}")
    endif()
    run_cli(no_cache_checkpoint "${exe}" ${base_args} --threads 1 --generations 6 --transposition-cache
        --resume "${own_plain}")
    if(NOT no_cache_checkpoint_RC EQUAL 1 OR NOT no_cache_checkpoint_ERR MATCHES "has no transposition cache")
        message(FATAL_ERROR "${exe}: a cache-less checkpoint resumed with --transposition-cache must be rejected: "
            "${no_cache_checkpoint_RC}\n${no_cache_checkpoint_ERR}")
    endif()
endforeach()

# ---- the content of --shape-constraints-config is part of the fingerprint, its path is not ----
set(shape_a "${WORK_DIR}/shape-a.json")
set(shape_b "${WORK_DIR}/shape-b.json")
set(shape_reformatted "${WORK_DIR}/shape-reformatted.json")
file(WRITE "${shape_a}" [[{"domains":{"X1":[-1,1]},"constraints":[{"op":"derivative","variable":"X1","order":1,"sign":1}]}]])
file(WRITE "${shape_b}" [[{"domains":{"X1":[-1,1]},"constraints":[{"op":"derivative","variable":"X1","order":1,"sign":-1}]}]])
file(WRITE "${shape_reformatted}" [[{ "constraints" : [ { "sign" : 1, "order" : 1, "variable" : "X1", "op" : "derivative" } ],
  "domains" : { "X1" : [ -1, 1 ] } }]])
set(shape_checkpoint "${WORK_DIR}/shape-checkpoint.beve")
run_ok(make_shape "${OPERON_GP}" ${base_args} --inputs X1 --threads 1 --generations 3
    --shape-constraints-config "${shape_a}" --checkpoint-interval 3 --checkpoint-file "${shape_checkpoint}")
run_ok(shape_same "${OPERON_GP}" ${base_args} --inputs X1 --threads 1 --generations 5
    --shape-constraints-config "${shape_a}" --resume "${shape_checkpoint}")
run_ok(shape_reformatted "${OPERON_GP}" ${base_args} --inputs X1 --threads 1 --generations 5
    --shape-constraints-config "${shape_reformatted}" --resume "${shape_checkpoint}")
expect_resume_rejected("${OPERON_GP}" "${shape_checkpoint}" "differing: .*shape.constraints"
    --inputs X1 --shape-constraints-config "${shape_b}")
expect_resume_rejected("${OPERON_GP}" "${shape_checkpoint}" "differing: .*shape.constraints" --inputs X1)

# ---- a --resume that cannot be used is an error, except for a file that does not exist ----
run_cli(missing "${OPERON_GP}" ${base_args} --threads 1 --generations 2 --resume "${WORK_DIR}/no-such-checkpoint.beve")
if(NOT missing_RC EQUAL 0 OR NOT missing_ERR MATCHES "does not exist, starting fresh")
    message(FATAL_ERROR "a missing checkpoint must start fresh with a warning: ${missing_RC}\n${missing_ERR}")
endif()
set(garbage "${WORK_DIR}/garbage.beve")
file(WRITE "${garbage}" "this is not a checkpoint")
expect_resume_rejected("${OPERON_GP}" "${garbage}" "cannot resume")
expect_resume_rejected("${OPERON_NSGP}" "${garbage}" "cannot resume")
set(empty "${WORK_DIR}/empty.beve")
file(WRITE "${empty}" "")
expect_resume_rejected("${OPERON_GP}" "${empty}" "cannot resume")
find_program(HEAD_PROGRAM head)
if(HEAD_PROGRAM)
    set(truncated "${WORK_DIR}/truncated.beve")
    execute_process(COMMAND "${HEAD_PROGRAM}" -c 300 "${committed_checkpoint}" OUTPUT_FILE "${truncated}")
    expect_resume_rejected("${OPERON_GP}" "${truncated}" "cannot resume")
endif()
