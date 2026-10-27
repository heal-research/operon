# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
cmake_minimum_required(VERSION 3.20)

project(operonTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)

if(PROJECT_IS_TOP_LEVEL)
  find_package(operon REQUIRED)
  enable_testing()
endif()

find_package(Catch2 3 REQUIRED)
find_package(scn REQUIRED)
find_package(nanobench REQUIRED)
# vdt is optional: if found, enables the "Backend vs vdt ULP accuracy" test case
# (gated by OPERON_HAVE_VDT in backend.cpp). Not in vcpkg.json; provide via overlay or nix.
find_package(vdt QUIET)

add_executable(operon_test
    source/operon_test.cpp
    source/implementation/ga_base.cpp
    source/implementation/backend.cpp
    source/implementation/autodiff.cpp
    source/implementation/cli_args.cpp
    source/implementation/cli_error.cpp
    source/implementation/pareto_front.cpp
    ../cli/source/pareto_front.cpp
    source/implementation/tree_diff.cpp
    source/implementation/range_oracles.cpp
    source/implementation/crossover.cpp
    source/implementation/crowding_distance.cpp
    source/implementation/details.cpp
    source/implementation/dispatch_table.cpp
    source/implementation/evaluation.cpp
    source/implementation/error_metrics.cpp
    source/implementation/information_criteria.cpp
    source/implementation/content_hash.cpp
    source/implementation/enumeration.cpp
    source/implementation/grammar.cpp
    source/implementation/hashing.cpp
    source/implementation/zobrist.cpp
    source/implementation/infix_parser.cpp
    source/implementation/initialization.cpp
    source/implementation/mutation.cpp
    source/implementation/node.cpp
    source/implementation/node_impact.cpp
    source/implementation/nondominatedsort.cpp
    source/implementation/permutation_importance.cpp
    source/implementation/evaluator.cpp
    source/implementation/optimizer.cpp
    source/implementation/probes.cpp
    source/implementation/random.cpp
    source/implementation/selection.cpp
    source/implementation/serialization.cpp
    source/implementation/shape_constrained_evaluator.cpp
    source/implementation/shape_constrained_value_evaluator.cpp
    source/implementation/shape_bound_correctness.cpp
    ../cli/source/shape_constraints_config.cpp
    source/implementation/standard_library.cpp
    source/performance/autodiff.cpp
    source/performance/creator.cpp
    source/performance/distance.cpp
    source/performance/evaluation.cpp
    source/performance/infix_parser.cpp
    source/performance/nondominatedsort.cpp
    source/performance/primitives.cpp
    source/performance/jit_breakeven.cpp
    source/performance/shape_bisection.cpp
    source/performance/bisection_strategies.cpp
    source/implementation/jet.cpp
    source/implementation/interval_jet.cpp
    source/implementation/second_order_jet.cpp
    source/implementation/interval_jet_compare.cpp
    source/implementation/jit.cpp
    source/implementation/pappus_backend.cpp
    source/implementation/pappus_nsgp.cpp
    source/implementation/registry_coverage.cpp
    source/implementation/user_defined_registries.cpp
    source/implementation/memory_view.cpp
    source/implementation/view_descriptor.cpp
    source/implementation/least_squares.cpp
    source/implementation/least_squares_lm_adapter.cpp
    source/implementation/least_squares_fit.cpp
    source/implementation/gradient_cost.cpp
    source/implementation/least_squares_gradient_adapter.cpp
    source/implementation/subtree.cpp
    source/implementation/interpreter_least_squares.cpp
    source/implementation/composed_function.cpp
    source/implementation/gaussian_gradient_cost.cpp
    source/implementation/poisson_gradient_cost.cpp
    )
target_link_libraries(operon_test PRIVATE operon::operon Catch2::Catch2WithMain scn::scn nanobench::nanobench)
target_compile_definitions(operon_test PRIVATE OPERON_TEST_SOURCE_DIR="${CMAKE_CURRENT_SOURCE_DIR}/..")
if(vdt_FOUND)
    target_link_libraries(operon_test PRIVATE vdt::vdt)
    target_compile_definitions(operon_test PRIVATE OPERON_HAVE_VDT)
endif()
target_compile_features(operon_test PRIVATE cxx_std_23)
target_include_directories(operon_test PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/../source/thirdparty ${CMAKE_CURRENT_SOURCE_DIR}/../cli/source)
set_target_properties(operon_test PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

# Hidden interval-jet comparison/benchmark harness ("[.interval-jet-compare]", "[.interval-jet-bench]") as a
# stand-alone executable so it can be built on its own, in Release, with heap-allocation counting enabled:
#   cmake --build <build-dir> --target operon_interval_jet_compare
add_executable(operon_interval_jet_compare EXCLUDE_FROM_ALL source/implementation/interval_jet_compare.cpp)
target_link_libraries(operon_interval_jet_compare PRIVATE operon::operon Catch2::Catch2WithMain)
target_compile_definitions(operon_interval_jet_compare PRIVATE OPERON_JET_COUNT_ALLOCATIONS)
target_compile_features(operon_interval_jet_compare PRIVATE cxx_std_23)
set_target_properties(operon_interval_jet_compare PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
)

if(MSVC)
    target_compile_options(operon_test PUBLIC "/std:c++latest")
else()
    if (UNIX AND NOT APPLE)
        target_link_options(operon_test PUBLIC "-Wl,--no-undefined")
    endif()
endif()

add_test(NAME operon_test COMMAND operon_test "~[performance]" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
windows_set_path(operon_test operon::operon)

# ---- Stage 1 memory-layout-plan consumer fixtures ----
# Standalone smoke executables (not linked into operon_test) proving the
# canonical view headers are usable in isolation by a minimal external
# consumer. See OPERON_MEMORY_LAYOUT_PLAN.md, Stage 1.
add_executable(memory_view_consumer source/memory_view_consumer.cpp)
target_link_libraries(memory_view_consumer PRIVATE operon::operon)
target_compile_features(memory_view_consumer PRIVATE cxx_std_23)
add_test(NAME memory_view_consumer COMMAND memory_view_consumer)
windows_set_path(memory_view_consumer operon::operon)
add_dependencies(operon_test memory_view_consumer)

# view_descriptor.h is deliberately plain C11 with no C++ dependency, so its
# consumer fixture is compiled as C and does not link operon::operon -- it
# only needs the header's own include path. C is enabled here, scoped to
# this directory, rather than on the root project: enabling it globally
# would impose a C-compiler prerequisite on every consumer that only wants
# the (C++-only) library. `enable_language()` works whether this file is
# add_subdirectory()'d from the root build or configured as its own
# standalone top-level project (PROJECT_IS_TOP_LEVEL above), so both paths
# get a C compiler for this one fixture.
enable_language(C)
set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED TRUE)
add_executable(view_descriptor_consumer source/view_descriptor_consumer.c)
target_include_directories(view_descriptor_consumer PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/../include")
set_target_properties(view_descriptor_consumer PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED YES)
add_test(NAME view_descriptor_consumer COMMAND view_descriptor_consumer)
add_dependencies(operon_test view_descriptor_consumer)

# Building this target guarantees every ctest fixture registered above is
# present, independent of which specific target a build invocation names
# (e.g. CI's `--target operon_test` alone already covers it transitively via
# the add_dependencies() calls above; this target additionally documents
# and covers a plain `cmake --build . --target build_all_tests` invocation).
add_custom_target(build_all_tests)
add_dependencies(build_all_tests operon_test memory_view_consumer view_descriptor_consumer)

# ---- Installed-package-consumer contract (package-consumer) ----
# Exercises the install -> relocate -> find_package(operon CONFIG) -> build
# -> run path a real external consumer takes, using the package this build
# just produced. Every step below runs at `ctest` time, not at configure
# time: only the actual build tree produced by *this* invocation is a valid
# thing to install from.
#
# Only meaningful when operon is being built directly (PROJECT_IS_TOP_LEVEL
# is FALSE here precisely when test/ has been add_subdirectory()'d from the
# main operon build -- see cmake/project-is-top-level.cmake). When test/ is
# instead configured standalone against some pre-existing operon
# installation, there is no operon build tree in this invocation to install
# from, so the fixture is skipped.
if(NOT PROJECT_IS_TOP_LEVEL)
    set(operon_package_consumer_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/package-consumer")
    set(operon_package_consumer_work_dir "${CMAKE_BINARY_DIR}/package-consumer")
    # The consumer uses a fresh CMake configuration. Preserve the parent
    # compiler and flags so an instrumented or ABI-specialized static package
    # is linked with its required runtime as well.
    set(operon_package_consumer_parent_context_args
        "-DOPERON_CMAKE_C_COMPILER=${CMAKE_C_COMPILER}"
        "-DOPERON_CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
        "-DOPERON_CMAKE_C_FLAGS=${CMAKE_C_FLAGS}"
        "-DOPERON_CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS}"
        "-DOPERON_CMAKE_EXE_LINKER_FLAGS=${CMAKE_EXE_LINKER_FLAGS}"
        "-DOPERON_CMAKE_SHARED_LINKER_FLAGS=${CMAKE_SHARED_LINKER_FLAGS}"
        "-DOPERON_CMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
    )


    add_test(
        NAME package_consumer
        COMMAND "${CMAKE_COMMAND}"
                "-DCMAKE_COMMAND=${CMAKE_COMMAND}"
                "-DOPERON_MAIN_BUILD_DIR=${CMAKE_BINARY_DIR}"
                "-DOPERON_CONFIGURATION=$<CONFIG>"
                "-DOPERON_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
                "-DOPERON_VCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}"
                "-DOPERON_VCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR}"
                "-DOPERON_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
                ${operon_package_consumer_parent_context_args}
                "-DOPERON_STAGE_DIR=${operon_package_consumer_work_dir}/stage"
                "-DOPERON_RELOCATED_PREFIX=${operon_package_consumer_work_dir}/relocated-prefix"
                "-DOPERON_CONSUMER_SOURCE_DIR=${operon_package_consumer_source_dir}"
                "-DOPERON_CONSUMER_BUILD_DIR=${operon_package_consumer_work_dir}/build"
                -P "${operon_package_consumer_source_dir}/run-consumer-test.cmake"
    )

    # Same install -> relocate -> find_package -> build -> run path, but the
    # consumer here links only operon::core (see
    # test/package-consumer/core/CMakeLists.txt), proving that
    # contract holds independently of the full operon::operon aggregate.
    # Reuses the same generic run-consumer-test.cmake driver above.
    set(operon_package_consumer_core_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/package-consumer/core")
    set(operon_package_consumer_core_work_dir "${CMAKE_BINARY_DIR}/package-consumer-core")

    add_test(
        NAME package_consumer_core
        COMMAND "${CMAKE_COMMAND}"
                "-DCMAKE_COMMAND=${CMAKE_COMMAND}"
                "-DOPERON_MAIN_BUILD_DIR=${CMAKE_BINARY_DIR}"
                "-DOPERON_CONFIGURATION=$<CONFIG>"
                "-DOPERON_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
                "-DOPERON_VCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}"
                "-DOPERON_VCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR}"
                "-DOPERON_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
                ${operon_package_consumer_parent_context_args}
                "-DOPERON_STAGE_DIR=${operon_package_consumer_core_work_dir}/stage"
                "-DOPERON_RELOCATED_PREFIX=${operon_package_consumer_core_work_dir}/relocated-prefix"
                "-DOPERON_CONSUMER_SOURCE_DIR=${operon_package_consumer_core_source_dir}"
                "-DOPERON_CONSUMER_BUILD_DIR=${operon_package_consumer_core_work_dir}/build"
                -P "${operon_package_consumer_source_dir}/run-consumer-test.cmake"
    )

    # Optional backend request with the backend's pappus dependency made
    # unavailable: core must still configure/build/run and the package must
    # report the backend components as not found.
    set(operon_package_consumer_optional_backend_source_dir "${CMAKE_CURRENT_SOURCE_DIR}/package-consumer/optional-backend")
    set(operon_package_consumer_optional_backend_work_dir "${CMAKE_BINARY_DIR}/package-consumer-optional-backend")

    add_test(
        NAME package_consumer_optional_backend
        COMMAND "${CMAKE_COMMAND}"
                "-DCMAKE_COMMAND=${CMAKE_COMMAND}"
                "-DOPERON_MAIN_BUILD_DIR=${CMAKE_BINARY_DIR}"
                "-DOPERON_CONFIGURATION=$<CONFIG>"
                "-DOPERON_TOOLCHAIN_FILE=${CMAKE_TOOLCHAIN_FILE}"
                "-DOPERON_VCPKG_TARGET_TRIPLET=${VCPKG_TARGET_TRIPLET}"
                "-DOPERON_VCPKG_INSTALLED_DIR=${VCPKG_INSTALLED_DIR}"
                "-DOPERON_MSVC_RUNTIME_LIBRARY=${CMAKE_MSVC_RUNTIME_LIBRARY}"
                ${operon_package_consumer_parent_context_args}
                "-DOPERON_STAGE_DIR=${operon_package_consumer_optional_backend_work_dir}/stage"
                "-DOPERON_RELOCATED_PREFIX=${operon_package_consumer_optional_backend_work_dir}/relocated-prefix"
                "-DOPERON_CONSUMER_SOURCE_DIR=${operon_package_consumer_optional_backend_source_dir}"
                "-DOPERON_CONSUMER_BUILD_DIR=${operon_package_consumer_optional_backend_work_dir}/build"
                -P "${operon_package_consumer_source_dir}/run-consumer-test.cmake"
    )
endif()

# ---- operon_nsgp: Pareto front export ----
# A budget stop inside a generation leaves unfilled offspring with default rank 0 and empty trees; they are not front
# members. The run must exit 0 and write the front.
if(TARGET operon_nsgp)
    # ctest runs operon_nsgp directly, so building operon_test must build it too.
    add_dependencies(operon_test operon_nsgp)
    add_test(
        NAME cli_nsgp_partial_generation_front
        COMMAND "${CMAKE_COMMAND}"
                "-DNSGP=$<TARGET_FILE:operon_nsgp>"
                "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/partial-front.json"
                "-DWORKDIR=${CMAKE_SOURCE_DIR}"
                "-DARGS=--dataset;data/Poly-10.csv;--target;Y;--train;0:100;--test;100:200;--population-size;128;--pool-size;128;--threads;1;--seed;7;--generations;30;--evaluations;220;--iterations;0;--objective;nmse"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/check_pareto_front.cmake"
    )
    # --epsilon reaches the front export: with a tolerance larger than every objective difference all individuals are
    # equivalent, so exactly one is exported (an exact-dominance export would list the whole trade-off front).
    add_test(
        NAME cli_nsgp_epsilon_front
        COMMAND "${CMAKE_COMMAND}"
                "-DNSGP=$<TARGET_FILE:operon_nsgp>"
                "-DOUT=${CMAKE_CURRENT_BINARY_DIR}/epsilon-front.json"
                "-DWORKDIR=${CMAKE_SOURCE_DIR}"
                "-DEXPECT_COUNT=1"
                "-DARGS=--dataset;data/Poly-10.csv;--target;Y;--train;0:100;--test;100:200;--population-size;50;--pool-size;50;--threads;1;--seed;7;--generations;5;--iterations;0;--objective;nmse;--epsilon;1e9"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/check_pareto_front.cmake"
    )
endif()

# ---- Command-line programs: option validation and error reporting ----
# Each case runs the real executable through cli/expect.cmake: `fail` cases need exit status 1 and a stderr match
# (a crash or abort does not count), `ok` cases need exit status 0 and a stdout match.
if(TARGET operon_gp AND TARGET operon_nsgp)
    # CI builds only operon_test; the CLI cases below run the real executables.
    add_dependencies(operon_test operon_gp operon_nsgp)
    set(operon_cli_data "data/Poly-10.csv")

    function(add_cli_test NAME PROGRAM EXPECT REGEX)
        string(REPLACE ";" "|" joined "${ARGN}")
        add_test(
            NAME cli_${NAME}
            COMMAND "${CMAKE_COMMAND}"
                    "-DCLI=$<TARGET_FILE:${PROGRAM}>"
                    "-DARGS=${joined}"
                    "-DEXPECT=${EXPECT}"
                    "-DREGEX=${REGEX}"
                    -P "${CMAKE_CURRENT_SOURCE_DIR}/cli_expect.cmake"
            WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
        )
    endfunction()

    # dead options
    add_cli_test(gp_rejects_selection_pressure operon_gp fail
        "error: --selection-pressure: is not supported.*--offspring-generator os:"
        "--dataset;${operon_cli_data};--target;Y;--selection-pressure;5")
    add_cli_test(nsgp_rejects_selection_pressure operon_nsgp fail
        "error: --selection-pressure: is not supported.*--offspring-generator os:"
        "--dataset;${operon_cli_data};--target;Y;--selection-pressure;5")
    add_cli_test(gp_rejects_epsilon operon_gp fail
        "error: --epsilon: is only used by operon_nsgp"
        "--dataset;${operon_cli_data};--target;Y;--epsilon;1e-3")
    add_cli_test(nsgp_accepts_epsilon operon_nsgp ok "iteration"
        "--dataset;${operon_cli_data};--target;Y;--population-size;10;--pool-size;10;--threads;1;--seed;1;--generations;1;--epsilon;1e-3")

    # build description
    add_cli_test(gp_capabilities operon_gp ok "\"schema\":\"operon.capabilities\".*\"program\":\"operon_gp\".*\"report-json\"" "--capabilities")
    set(operon_active_flags "${CMAKE_CXX_FLAGS}")
    foreach(operon_cfg IN LISTS operon_flag_configs)
        string(TOUPPER "${operon_cfg}" operon_cfg_upper)
        string(APPEND operon_active_flags "$<$<CONFIG:${operon_cfg}>: ${CMAKE_CXX_FLAGS_${operon_cfg_upper}}>")
    endforeach()
    # The compile options of operon_operon are resolved by file(GENERATE), the same way as for the reported flags (an
    # add_test() argument is evaluated in a context that omits options inherited from linked targets).
    file(GENERATE
        OUTPUT "${CMAKE_CURRENT_BINARY_DIR}/capabilities-target-options-$<CONFIG>.txt"
        CONTENT "$<JOIN:$<TARGET_PROPERTY:COMPILE_OPTIONS>,|>"
        TARGET operon_operon
    )
    add_test(
        NAME cli_capabilities_compiler_flags
        COMMAND "${CMAKE_COMMAND}"
                "-DCLI=$<TARGET_FILE:operon_gp>"
                "-DCONFIG=$<CONFIG>"
                "-DTARGET_OPTIONS_FILE=${CMAKE_CURRENT_BINARY_DIR}/capabilities-target-options-$<CONFIG>.txt"
                "-DDEBUG_FLAGS=${CMAKE_CXX_FLAGS_DEBUG}"
                "-DACTIVE_FLAGS=${operon_active_flags}"
                -P "${CMAKE_CURRENT_SOURCE_DIR}/check_capabilities_flags.cmake"
        WORKING_DIRECTORY "${CMAKE_SOURCE_DIR}"
    )
    add_cli_test(nsgp_capabilities operon_nsgp ok "\"program\":\"operon_nsgp\".*\"machine_report_schema_versions\":\\[\\].*\"sorter\"" "--capabilities")

    # threads
    add_cli_test(gp_rejects_zero_threads operon_gp fail
        "error: --threads: must be at least 1 .omit the option to use all hardware threads."
        "--dataset;${operon_cli_data};--target;Y;--threads;0")
    add_cli_test(nsgp_rejects_zero_threads operon_nsgp fail
        "error: --threads: must be at least 1 .omit the option to use all hardware threads."
        "--dataset;${operon_cli_data};--target;Y;--threads;0")

    # budgets
    add_cli_test(gp_rejects_malformed_evaluations operon_gp fail
        "error: --evaluations: 'lots' is not a non-negative integer or 'unlimited'"
        "--dataset;${operon_cli_data};--target;Y;--evaluations;lots")
    add_cli_test(gp_rejects_malformed_generations operon_gp fail
        "error: --generations: '-3' is not a non-negative integer or 'unlimited'"
        "--dataset;${operon_cli_data};--target;Y;--generations;-3")
    add_cli_test(gp_unlimited_generations_stop_on_budget operon_gp ok "iteration"
        "--dataset;${operon_cli_data};--target;Y;--population-size;10;--pool-size;10;--threads;1;--seed;1;--generations;unlimited;--evaluations;60")
    add_cli_test(gp_unlimited_evaluations_stop_on_generations operon_gp ok "iteration[^\n]*\n[^\n]*\n[^\n]*\n"
        "--dataset;${operon_cli_data};--target;Y;--population-size;10;--pool-size;10;--threads;1;--seed;1;--generations;2;--evaluations;unlimited")
    add_cli_test(nsgp_unlimited_budgets operon_nsgp ok "iteration"
        "--dataset;${operon_cli_data};--target;Y;--population-size;10;--pool-size;10;--threads;1;--seed;1;--generations;2;--evaluations;18446744073709551615")

    # offspring-generator spec parsing
    foreach(spec "os:abc" "os:10:x" "os:10:nan" "os::5" "os:" "brood:x" "brood:5x" "brood:-1" "brood:0" "brood:" "poly:x" "poly::" "poly:0")
        string(REGEX REPLACE "[^a-z0-9]" "_" tag "${spec}")
        add_cli_test(gp_rejects_generator_${tag} operon_gp fail
            "error: unable to parse generator argument '${spec}': '[^']*' (is not a valid|takes at most)"
            "--dataset;${operon_cli_data};--target;Y;--population-size;10;--pool-size;10;--threads;1;--offspring-generator;${spec}")
    endforeach()
    foreach(spec "basic:x" "os:10:0.5:x" "brood:5:x" "poly:5:x")
        string(REGEX REPLACE "[^a-z0-9]" "_" tag "${spec}")
        add_cli_test(gp_rejects_generator_fields_${tag} operon_gp fail
            "error: unable to parse generator argument '${spec}': '[a-z]*' takes at most"
            "--dataset;${operon_cli_data};--target;Y;--population-size;10;--pool-size;10;--threads;1;--offspring-generator;${spec}")
    endforeach()
    add_cli_test(gp_rejects_empty_generator operon_gp fail "error: unable to parse generator argument ''"
        "--dataset;${operon_cli_data};--target;Y;--population-size;10;--pool-size;10;--threads;1;--offspring-generator=")
    add_cli_test(gp_accepts_valid_generator_specs operon_gp ok "iteration"
        "--dataset;${operon_cli_data};--target;Y;--population-size;10;--pool-size;10;--threads;1;--seed;1;--generations;1;--offspring-generator;os:50:0.5")

    # operon_enum: seed/threads may be omitted (random seed, all hardware threads); GP-only options are rejected
    if(TARGET operon_enum)
        add_dependencies(operon_test operon_enum)
        add_cli_test(enum_capabilities operon_enum ok "\"program\":\"operon_enum\"" "--capabilities")
        add_cli_test(enum_default_seed_and_threads operon_enum ok "mdl_bits="
            "--dataset;${operon_cli_data};--target;Y;--train;0:50;--max-complexity;4;--top-k;1")
        add_cli_test(enum_rejects_epsilon operon_enum fail
            "error: --epsilon: is only used by operon_nsgp"
            "--dataset;${operon_cli_data};--target;Y;--epsilon;1e-3")
        add_cli_test(enum_rejects_selection_pressure operon_enum fail
            "error: --selection-pressure: is not supported and has no effect"
            "--dataset;${operon_cli_data};--target;Y;--selection-pressure;5")
    endif()
endif()
