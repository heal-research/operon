# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
cmake_minimum_required(VERSION 3.20)

project(operonCli LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)
if (PROJECT_IS_TOP_LEVEL)
    find_package(operon REQUIRED)
endif()

find_package(cxxopts REQUIRED)
find_package(scn REQUIRED)

# Fully static-link the CLI executables (against libc and every dependency,
# not just liboperon). Distinct from BUILD_SHARED_LIBS (which only controls
# whether liboperon itself is a .a or .so): this additionally requires every
# dependency to be available as a static archive, so it's opt-in and only
# meant for a dedicated build (e.g. the flake's cli-static output) rather
# than a default derived from BUILD_SHARED_LIBS=OFF.
option(OPERON_STATIC_CLI "Fully statically link the CLI executables" OFF)

function(add_operon_cli NAME)
    add_executable(${NAME}
        source/${NAME}.cpp
        source/jit_setup.cpp
        source/operator_factory.cpp
        source/pareto_front.cpp
        source/util.cpp
    )

    target_link_libraries(${NAME} PRIVATE operon::operon cxxopts::cxxopts scn::scn)
    target_compile_features(${NAME} PRIVATE cxx_std_23)
    set_target_properties(${NAME} PROPERTIES
        CXX_VISIBILITY_PRESET hidden
        VISIBILITY_INLINES_HIDDEN YES
    )

    if(MSVC)
        target_compile_options(${NAME} PRIVATE "/std:c++latest")
    else()
        if (UNIX AND NOT APPLE)
            target_link_options(${NAME} PRIVATE "-Wl,--no-undefined")
            if (OPERON_STATIC_CLI)
                target_link_options(${NAME} PRIVATE "-static")
            endif()
        endif()
    endif()

    install(TARGETS ${NAME}
        RUNTIME
        COMPONENT operonCli_Runtime
        DESTINATION "${CMAKE_INSTALL_BINDIR}"
        )
endfunction()

add_operon_cli(operon_gp)
add_operon_cli(operon_nsgp)
add_operon_cli(operon_parse_model)
