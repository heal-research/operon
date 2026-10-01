# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#
# Standalone consumer project. NEVER add_subdirectory()'d into the main
# operon build -- run-consumer-test.cmake configures/builds/runs it as an
# entirely separate `cmake` invocation, against an installed and relocated
# `operon` package, to prove the installed package's find_package/target
# contract holds outside the operon source and build trees.
cmake_minimum_required(VERSION 3.25)

project(operon_package_consumer LANGUAGES CXX)

find_package(operon CONFIG REQUIRED)

foreach(component IN ITEMS core backend_adapter operon)
    if(NOT TARGET operon::${component})
        message(FATAL_ERROR "operon package did not provide operon::${component}")
    endif()
endforeach()

foreach(component IN ITEMS core backend_adapter operon)
    if(NOT operon_${component}_FOUND)
        message(FATAL_ERROR "operon_${component}_FOUND is not set for the default package request")
    endif()
endforeach()

get_target_property(operon_aggregate_links operon::operon INTERFACE_LINK_LIBRARIES)
if(NOT "${operon_aggregate_links}" MATCHES "operon::core" OR NOT "${operon_aggregate_links}" MATCHES "operon::backend_adapter")
    message(FATAL_ERROR "operon::operon must aggregate operon::core and operon::backend_adapter")
endif()
add_executable(package_consumer source/main.cpp)
target_link_libraries(package_consumer PRIVATE operon::operon)
target_compile_features(package_consumer PRIVATE cxx_std_23)

enable_testing()
add_test(NAME package_consumer COMMAND package_consumer)
