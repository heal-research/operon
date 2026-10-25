# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

# Command-line contract of --evaluations for operon_gp and operon_nsgp:
#   - a zero --population-size or --pool-size, and a budget below --population-size (initialization costs at least
#     one evaluation per individual), are rejected with exit status 1; a budget equal to the population is accepted;
#   - the budget limits committed work: one evaluation less than init + k generations commits k - 1 generations,
#     exactly init + k commits k, and the final state is the same for every --threads;
#   - a binding offspring-selection pressure (os:2:1, with and without the transposition cache) ends the run at the
#     same generation for every --threads.
#
# Usage: cmake -DOPERON_GP=<exe> -DOPERON_NSGP=<exe> -DDATA_DIR=<dir> -P budget.cmake

foreach(var OPERON_GP OPERON_NSGP DATA_DIR)
    if(NOT DEFINED ${var})
        message(FATAL_ERROR "${var} is not set")
    endif()
endforeach()

include("${CMAKE_CURRENT_LIST_DIR}/cli_helpers.cmake")

set(base_args
    --dataset "${DATA_DIR}/Poly-10.csv" --target Y --train 0:100 --test 100:200
    --population-size 20 --pool-size 20 --seed 7)

# The budget unit of a final_state() string: residual + Jacobian evaluations (not the evaluator call count).
function(committed out state)
    if(NOT state MATCHES "[|]res_eval=([0-9]+)[|]jac_eval=([0-9]+)")
        message(FATAL_ERROR "no res_eval/jac_eval in: ${state}")
    endif()
    math(EXPR cost "${CMAKE_MATCH_1} + ${CMAKE_MATCH_2}")
    set(${out} "${cost}" PARENT_SCOPE)
endfunction()

function(expect_same what left right)
    if(NOT left STREQUAL right)
        message(FATAL_ERROR "${what}:\n${left}\n${right}")
    endif()
endfunction()

foreach(exe IN ITEMS "${OPERON_GP}" "${OPERON_NSGP}")
    get_filename_component(name "${exe}" NAME)

    message(STATUS "${name}: a budget below the population size is rejected")
    run_cli(low "${exe}" ${base_args} --threads 1 --evaluations 19)
    if(NOT low_RC EQUAL 1 OR NOT low_ERR MATCHES "error: --evaluations: 19 is smaller than --population-size 20")
        message(FATAL_ERROR "${name}: expected exit status 1 and a budget error, got ${low_RC}\n${low_ERR}")
    endif()
    expect_success("${exe}" ${base_args} --threads 1 --evaluations 20 --generations 3)

    message(STATUS "${name}: zero population or pool is rejected")
    foreach(option IN ITEMS population-size pool-size)
        run_cli(zero "${exe}" ${base_args} --threads 1 --evaluations 100 --${option} 0)
        if(NOT zero_RC EQUAL 1 OR NOT zero_ERR MATCHES "error: --${option}: must be at least 1")
            message(FATAL_ERROR "${name} --${option} 0: expected exit status 1 and a size error, got ${zero_RC}\n${zero_ERR}")
        endif()
    endforeach()
    run_cli(zero "${exe}" ${base_args} --threads 1 --population-size 0 --pool-size 0 --evaluations 0)
    if(NOT zero_RC EQUAL 1)
        message(FATAL_ERROR "${name}: --population-size 0 --evaluations 0 must be rejected, got ${zero_RC}\n${zero_ERR}")
    endif()

    foreach(generator IN ITEMS basic os:10 brood:3 poly:3)
      foreach(iterations IN ITEMS 0 2)
        foreach(cache IN ITEMS off on)
            # local search (--iterations 2) makes a generation cost more than one evaluation per offspring
            set(variant ${base_args} --offspring-generator ${generator} --iterations ${iterations})
            if(cache STREQUAL "on")
                list(APPEND variant --transposition-cache --cache-visibility committed)
            endif()
            message(STATUS "${name}: budget boundary, ${generator}, iterations ${iterations}, cache ${cache}")

            # What an unbounded run commits after 0, 1 and 2 generations.
            foreach(k IN ITEMS 0 1 2)
                final_state(unbounded${k} "${exe}" ${variant} --generations ${k} --threads 1 --evaluations unlimited)
                committed(cost${k} "${unbounded${k}}")
            endforeach()

            foreach(threads IN ITEMS 1 4 16)
                # init + 1 generation - 1: nothing but the initial population
                math(EXPR below "${cost1} - 1")
                final_state(s "${exe}" ${variant} --generations 3 --threads ${threads} --evaluations ${below})
                expect_same("${name} ${generator} cache ${cache}: budget ${below} at ${threads} threads: initial population"
                    "${s}" "${unbounded0}")
                # init + 1 generation: exactly one generation commits
                final_state(s "${exe}" ${variant} --generations 3 --threads ${threads} --evaluations ${cost1})
                expect_same("${name} ${generator} cache ${cache}: budget ${cost1} at ${threads} threads: one generation"
                    "${s}" "${unbounded1}")
                # init + 2 generations - 1: the second generation is discarded. (If the first generation already ended
                # the unbounded run, e.g. by crossing the selection-pressure cap on this platform's random draws,
                # there is no second generation to discard.)
                if(cost2 GREATER cost1)
                    math(EXPR below "${cost2} - 1")
                    final_state(s "${exe}" ${variant} --generations 3 --threads ${threads} --evaluations ${below})
                    expect_same("${name} ${generator} cache ${cache}: budget ${below} at ${threads} threads: one generation"
                        "${s}" "${unbounded1}")
                endif()
            endforeach()
        endforeach()
      endforeach()
    endforeach()

    message(STATUS "${name}: a binding selection pressure does not depend on --threads")
    # Which seeds let the pressure cap end the run depends on the platform's random distributions: scan seeds until
    # some end the run early, and require that at least one did.
    foreach(cache IN ITEMS off on)
        set(bound 0)
        foreach(seed RANGE 13 76)
            if(bound GREATER_EQUAL 2)
                break()
            endif()
            set(cached_args
                --dataset "${DATA_DIR}/Poly-10.csv" --target Y --train 0:100 --test 100:200
                --population-size 8 --pool-size 8 --seed ${seed} --iterations 0 --generations 100
                --evaluations unlimited --offspring-generator os:2:1)
            if(cache STREQUAL "on")
                list(APPEND cached_args --transposition-cache --cache-visibility committed)
            endif()
            final_state(reference "${exe}" ${cached_args} --threads 1)
            if(NOT reference MATCHES "[|]iteration=([0-9]+)" OR NOT CMAKE_MATCH_1 LESS 100)
                continue() # the cap did not end the run for this seed
            endif()
            math(EXPR bound "${bound} + 1")
            foreach(threads IN ITEMS 4 16 16)
                final_state(s "${exe}" ${cached_args} --threads ${threads})
                expect_same("${name}: os:2:1, seed ${seed}, cache ${cache}, at ${threads} threads" "${s}" "${reference}")
            endforeach()
        endforeach()
        if(bound EQUAL 0)
            message(FATAL_ERROR "${name}: no seed in 13..76 let the selection pressure end the run (cache ${cache})")
        endif()
    endforeach()
endforeach()
