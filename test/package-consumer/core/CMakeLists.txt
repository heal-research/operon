# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#
# Standalone consumer project for the operon::core contract. NEVER
# add_subdirectory()'d into the main operon build -- run-consumer-test.cmake
# configures/builds/runs it as an entirely separate `cmake` invocation,
# against an installed and relocated `operon` package, to prove that a
# consumer wanting only the compiled, Eigen-free core contracts
# (tree/grammar/primitive-set/hash implementation plus header-only
# numerical/view contracts) can link operon::core alone: no Eigen,
# no Ceres, no L-BFGS, and no dependency on liboperon's compiled
# backend-adapter.
cmake_minimum_required(VERSION 3.25)

project(operon_package_consumer_core LANGUAGES CXX)

# A core-only request must not attempt backend dependency discovery: no
# Eigen (interpreter/optimizer numerics), no L-BFGS (CoefficientOptimizer),
# no vstat/infix-parser/pappus/eve/asmjit (there is no separate "Ceres"
# CMake package to check -- the vendored include/operon/ceres/ headers are
# autodiff-only and never reach find_package()). Nothing is disabled here on
# purpose: when the build environment can discover these packages, the
# assertions below prove operonConfig.cmake never even probed for them.
find_package(operon CONFIG REQUIRED COMPONENTS core)

if(NOT operon_core_FOUND)
    message(FATAL_ERROR "package-consumer(core): operon_core_FOUND is not set")
endif()
foreach(backend_target IN ITEMS
    Eigen3::Eigen lbfgs::lbfgs infix-parser::infix-parser vstat::vstat
    pappus::pappus eve::eve asmjit::asmjit
)
    if(TARGET ${backend_target})
        message(FATAL_ERROR
            "package-consumer(core): find_package(operon COMPONENTS core) "
            "discovered backend dependency ${backend_target}")
    endif()
endforeach()

# ---- Structural guard: operon::core must not reach any backend dependency ----
# Complements the executable below (which proves the headers actually
# compile and run standalone): this walks the target's transitive interface
# link libraries (breadth-first, plain queue -- no function/macro recursion,
# which would need error-prone scope juggling for a value this simple) and
# fails loudly if Eigen, L-BFGS, vstat, infix-parser, pappus, eve, asmjit or
# the backend-adapter/aggregate targets appear anywhere, rather than relying
# solely on them happening not to be `#include`d by the handful of headers
# exercised at runtime.
set(operon_core_queue "operon::core")
set(operon_core_seen "")
while(operon_core_queue)
    list(POP_FRONT operon_core_queue operon_core_current)
    if(operon_core_current IN_LIST operon_core_seen)
        continue()
    endif()
    list(APPEND operon_core_seen "${operon_core_current}")

    if(operon_core_current MATCHES "[Ee]igen|^(eve|lbfgs|vstat|pappus|infix-parser|asmjit)::|^operon::(backend_adapter|operon)$")
        message(FATAL_ERROR
            "package-consumer(core): '${operon_core_current}' is "
            "transitively reachable from operon::core, which must stay backend-free")
    endif()

    if(TARGET ${operon_core_current})
        get_target_property(operon_core_deps ${operon_core_current} INTERFACE_LINK_LIBRARIES)
        if(operon_core_deps)
            foreach(dep IN LISTS operon_core_deps)
                # Skip generator expressions (none of operon's core
                # dependencies use them); a literal target name is expected.
                if(NOT dep MATCHES "\\$<" AND NOT dep IN_LIST operon_core_seen)
                    list(APPEND operon_core_queue "${dep}")
                endif()
            endforeach()
        endif()
    endif()
endwhile()

get_target_property(operon_core_includes operon::core INTERFACE_INCLUDE_DIRECTORIES)
if(operon_core_includes)
    foreach(dir IN LISTS operon_core_includes)
        if(dir MATCHES "[Ee]igen")
            message(FATAL_ERROR
                "package-consumer(core): operon::core exposes "
                "an Eigen include directory: ${dir}")
        endif()
    endforeach()
endif()

# ---- Header-closure guard: no backend header reachable from the core headers ----
# The link-interface walk above cannot see a core header that #includes a
# backend header directly, and a build environment that puts backend include
# paths on the compiler's default search path (nix, distro packages) would
# compile such an include without complaint. Ask the compiler for the actual
# transitive include set of source/main.cpp (which includes every installed
# core header) using only operon::core's transitive include directories, and
# fail if any header from a backend dependency or backend-adapter directory
# appears. -M is a GNU/Clang driver option; other compilers skip this guard.
if(CMAKE_CXX_COMPILER_ID MATCHES "^(GNU|Clang|AppleClang)$")
    set(operon_core_include_flags)
    foreach(target IN LISTS operon_core_seen)
        if(TARGET ${target})
            get_target_property(target_includes ${target} INTERFACE_INCLUDE_DIRECTORIES)
            foreach(dir IN LISTS target_includes)
                if(dir AND NOT dir MATCHES "\\$<")
                    list(APPEND operon_core_include_flags "-I${dir}")
                endif()
            endforeach()
        endif()
    endforeach()

    execute_process(
        COMMAND "${CMAKE_CXX_COMPILER}" -std=c++23 -M -x c++
                ${operon_core_include_flags}
                "${CMAKE_CURRENT_SOURCE_DIR}/source/main.cpp"
        RESULT_VARIABLE operon_closure_result
        OUTPUT_VARIABLE operon_closure_output
        ERROR_VARIABLE operon_closure_error
    )
    if(NOT operon_closure_result EQUAL 0)
        message(FATAL_ERROR
            "package-consumer(core): header-closure scan failed to preprocess "
            "source/main.cpp using only operon::core's transitive include "
            "directories -- a core header may include a header outside them\n"
            "${operon_closure_error}")
    endif()
    if(NOT operon_closure_output MATCHES "operon/core/tree\\.hpp")
        message(FATAL_ERROR
            "package-consumer(core): header-closure scan did not reach the core headers")
    endif()
    if(operon_closure_output MATCHES "/(Eigen|unsupported|eve|lbfgs|vstat|pappus|infix-parser|asmjit)/|operon/(ceres|interpreter)/")
        message(FATAL_ERROR
            "package-consumer(core): a backend header (${CMAKE_MATCH_0}) is "
            "reachable from the core headers")
    endif()
endif()

add_executable(package_consumer_core source/main.cpp)
# PRIVATE and operon::core alone: no operon::operon, no
# operon::backend_adapter. This is the actual contract under test.
target_link_libraries(package_consumer_core PRIVATE operon::core)
target_compile_features(package_consumer_core PRIVATE cxx_std_23)

enable_testing()
add_test(NAME package_consumer_core COMMAND package_consumer_core)
