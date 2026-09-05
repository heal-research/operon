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
    source/implementation/tree_diff.cpp
    source/implementation/range_tightening.cpp
    source/implementation/feynman_benchmarks.cpp
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
    source/implementation/jit.cpp
    source/implementation/pappus_backend.cpp
    source/implementation/pappus_nsgp.cpp
    source/implementation/registry_coverage.cpp
    source/implementation/user_defined_registries.cpp
    source/implementation/subtree.cpp
    source/implementation/composed_function.cpp
    )
target_link_libraries(operon_test PRIVATE operon::operon Catch2::Catch2WithMain scn::scn nanobench::nanobench)
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

if(MSVC)
    target_compile_options(operon_test PUBLIC "/std:c++latest")
else()
    if (UNIX AND NOT APPLE)
        target_link_options(operon_test PUBLIC "-Wl,--no-undefined")
    endif()
endif()

add_test(NAME operon_test COMMAND operon_test "~[performance]" WORKING_DIRECTORY ${CMAKE_SOURCE_DIR})
windows_set_path(operon_test operon::operon)

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
                "-DOPERON_STAGE_DIR=${operon_package_consumer_work_dir}/stage"
                "-DOPERON_RELOCATED_PREFIX=${operon_package_consumer_work_dir}/relocated-prefix"
                "-DOPERON_CONSUMER_SOURCE_DIR=${operon_package_consumer_source_dir}"
                "-DOPERON_CONSUMER_BUILD_DIR=${operon_package_consumer_work_dir}/build"
                -P "${operon_package_consumer_source_dir}/run-consumer-test.cmake"
    )
endif()
