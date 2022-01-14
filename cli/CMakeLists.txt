cmake_minimum_required(VERSION 3.14)

project(operonCli LANGUAGES CXX)

include(../cmake/windows-set-path.cmake)
if (PROJECT_IS_TOP_LEVEL)
    find_package(operon REQUIRED)
endif()

find_package(cxxopts REQUIRED)
find_package(scn REQUIRED)

add_executable(operon_gp
    source/gp.cpp
    source/operator_factory.cpp
    source/util.cpp
    )

target_link_libraries(operon_gp PRIVATE operon::operon cxxopts::cxxopts scn::scn)
target_compile_features(operon_gp PRIVATE cxx_std_17)
target_include_directories(operon_gp PRIVATE ${PROJECT_SOURCE_DIR}/source/thirdparty)
target_compile_options(operon_gp PRIVATE "-march=x86-64;-mavx2;-mfma")
target_link_options(operon_gp PUBLIC "-Wl,--no-undefined")

install(TARGETS operon_gp
    RUNTIME
    COMPONENT operonCli_Runtime
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )

add_executable(operon_nsgp
    source/nsgp.cpp
    source/operator_factory.cpp
    source/util.cpp
    )

target_link_libraries(operon_nsgp PRIVATE operon::operon cxxopts::cxxopts scn::scn)
target_compile_features(operon_nsgp PRIVATE cxx_std_17)
target_include_directories(operon_nsgp PRIVATE ${PROJECT_SOURCE_DIR}/source/thirdparty)
target_compile_options(operon_nsgp PRIVATE "-march=x86-64;-mavx2;-mfma")
target_link_options(operon_nsgp PUBLIC "-Wl,--no-undefined")

install(TARGETS operon_nsgp
    RUNTIME
    COMPONENT operonCli_Runtime
    DESTINATION "${CMAKE_INSTALL_BINDIR}"
    )
