cmake_minimum_required(VERSION 3.20)

project(operonCli LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)
if (PROJECT_IS_TOP_LEVEL)
    find_package(operon REQUIRED)
endif()

find_package(cxxopts REQUIRED)
find_package(scn REQUIRED)

function(add_operon_cli NAME)
    add_executable(${NAME}
        source/${NAME}.cpp
        source/operator_factory.cpp
        source/util.cpp
        )

    target_link_libraries(${NAME} PRIVATE operon::operon cxxopts::cxxopts scn::scn)
    target_compile_features(${NAME} PRIVATE cxx_std_20)

    if(MSVC)
        target_compile_options(${NAME} PRIVATE "/std:c++latest")
    else()
        if (UNIX AND NOT APPLE)
            target_link_options(${NAME} PRIVATE "-Wl,--no-undefined")
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
