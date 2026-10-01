# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#
# Standalone consumer project for an *optional* backend request. NEVER
# add_subdirectory()'d into the main operon build -- run-consumer-test.cmake
# configures/builds/runs it as an entirely separate `cmake` invocation,
# against an installed and relocated `operon` package.
#
# Contract under test: a consumer that requires operon::core and merely
# probes for the backend adapter (OPTIONAL_COMPONENTS) must configure, build
# and run when a backend dependency is unavailable, and the package must
# report the backend components as not found instead of exposing a target
# whose link interface cannot be resolved.
cmake_minimum_required(VERSION 3.25)

project(operon_package_consumer_optional_backend LANGUAGES CXX)

# Make the backend unavailable regardless of what the build environment can
# otherwise discover.
set(CMAKE_DISABLE_FIND_PACKAGE_pappus TRUE)
find_package(operon CONFIG REQUIRED COMPONENTS core OPTIONAL_COMPONENTS backend_adapter)

if(NOT operon_core_FOUND)
    message(FATAL_ERROR "package-consumer(optional-backend): operon_core_FOUND is not set")
endif()
foreach(component IN ITEMS backend_adapter operon)
    if(operon_${component}_FOUND)
        message(FATAL_ERROR
            "package-consumer(optional-backend): operon_${component}_FOUND is "
            "set although pappus is unavailable")
    endif()
endforeach()
if(TARGET pappus::pappus)
    message(FATAL_ERROR "package-consumer(optional-backend): pappus::pappus must be unavailable")
endif()

add_executable(package_consumer_optional_backend source/main.cpp)
target_link_libraries(package_consumer_optional_backend PRIVATE operon::core)
target_compile_features(package_consumer_optional_backend PRIVATE cxx_std_23)

enable_testing()
add_test(NAME package_consumer_optional_backend COMMAND package_consumer_optional_backend)
