# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
cmake_minimum_required(VERSION 3.20)

project(operonCli LANGUAGES CXX)

# Unlike BUILD_SHARED_LIBS (liboperon itself as .a vs .so), this also requires
# every dependency to be static, so it's opt-in for dedicated builds only
# (e.g. the flake's cli-static output).
option(OPERON_STATIC_CLI "Fully statically link the CLI executables" OFF)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)
if (PROJECT_IS_TOP_LEVEL)
    find_package(operon REQUIRED)
endif()

find_package(cxxopts REQUIRED)
find_package(scn REQUIRED)
find_package(glaze REQUIRED)

# Extra per-target sources (e.g. probes_config.cpp for operon_gp/operon_nsgp
# only - operon_parse_model/operon_enum don't run a GeneticAlgorithmBase
# loop, so they have nothing to wire a ProbeChain into) are passed as ARGN.
function(add_operon_cli NAME)
    add_executable(${NAME}
        source/${NAME}.cpp
        source/jit_setup.cpp
        source/operator_factory.cpp
        source/pareto_front.cpp
        source/util.cpp
        ${ARGN}
    )

    target_link_libraries(${NAME} PRIVATE operon::operon cxxopts::cxxopts scn::scn glaze::glaze)
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
            if (NOT BUILD_SHARED_LIBS)
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

add_operon_cli(operon_gp source/probes_config.cpp)
add_operon_cli(operon_nsgp source/probes_config.cpp)
add_operon_cli(operon_parse_model)
add_operon_cli(operon_enum)
