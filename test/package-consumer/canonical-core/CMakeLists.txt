# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#
# Standalone consumer project for the operon::canonical_core contract. NEVER
# add_subdirectory()'d into the main operon build -- run-consumer-test.cmake
# configures/builds/runs it as an entirely separate `cmake` invocation,
# against an installed and relocated `operon` package, to prove that a
# consumer wanting only the header-only, Eigen-free canonical-core contracts
# can link operon::canonical_core alone: no Eigen, no Ceres, no L-BFGS, and
# no dependency on liboperon's compiled backend-adapter.
cmake_minimum_required(VERSION 3.25)

project(operon_package_consumer_canonical_core LANGUAGES CXX)

# This must configure without attempting backend dependency discovery.
set(CMAKE_DISABLE_FIND_PACKAGE_Eigen3 TRUE)
find_package(operon CONFIG REQUIRED COMPONENTS canonical_core)

# ---- Structural guard: operon::canonical_core must not reach Eigen ----
# Complements the executable below (which proves the headers actually
# compile and run standalone): this walks the target's transitive interface
# link libraries (breadth-first, plain queue -- no function/macro recursion,
# which would need error-prone scope juggling for a value this simple) and
# fails loudly if Eigen appears anywhere, rather than relying solely on it
# happening not to be `#include`d by the handful of headers exercised at
# runtime.
set(operon_canonical_core_queue "operon::canonical_core")
set(operon_canonical_core_seen "")
while(operon_canonical_core_queue)
    list(POP_FRONT operon_canonical_core_queue operon_canonical_core_current)
    if(operon_canonical_core_current IN_LIST operon_canonical_core_seen)
        continue()
    endif()
    list(APPEND operon_canonical_core_seen "${operon_canonical_core_current}")

    if(operon_canonical_core_current MATCHES "[Ee]igen")
        message(FATAL_ERROR
            "package-consumer(canonical-core): '${operon_canonical_core_current}' is "
            "transitively reachable from operon::canonical_core, which must stay Eigen-free")
    endif()

    if(TARGET ${operon_canonical_core_current})
        get_target_property(operon_canonical_core_deps ${operon_canonical_core_current} INTERFACE_LINK_LIBRARIES)
        if(operon_canonical_core_deps)
            foreach(dep IN LISTS operon_canonical_core_deps)
                # Skip generator expressions (none of operon's canonical-core
                # dependencies use them); a literal target name is expected.
                if(NOT dep MATCHES "\\$<" AND NOT dep IN_LIST operon_canonical_core_seen)
                    list(APPEND operon_canonical_core_queue "${dep}")
                endif()
            endforeach()
        endif()
    endif()
endwhile()

get_target_property(operon_canonical_core_includes operon::canonical_core INTERFACE_INCLUDE_DIRECTORIES)
if(operon_canonical_core_includes)
    foreach(dir IN LISTS operon_canonical_core_includes)
        if(dir MATCHES "[Ee]igen")
            message(FATAL_ERROR
                "package-consumer(canonical-core): operon::canonical_core exposes "
                "an Eigen include directory: ${dir}")
        endif()
    endforeach()
endif()

add_executable(package_consumer_canonical_core source/main.cpp)
# PRIVATE and operon::canonical_core alone: no operon::operon, no
# operon::backend_adapter. This is the actual contract under test.
target_link_libraries(package_consumer_canonical_core PRIVATE operon::canonical_core)
target_compile_features(package_consumer_canonical_core PRIVATE cxx_std_23)

enable_testing()
add_test(NAME package_consumer_canonical_core COMMAND package_consumer_canonical_core)
