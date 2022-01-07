cmake_minimum_required(VERSION 3.14)

project(operonTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)

if(PROJECT_IS_TOP_LEVEL)
  find_package(operon REQUIRED)
  find_package(doctest REQUIRED)
  enable_testing()
endif()

add_executable(operon_test
    source/operon_test.cpp
    source/implementation/evaluation.cpp
    source/implementation/details.cpp
    source/implementation/hashing.cpp
    source/implementation/infix_parser.cpp
    source/implementation/initialization.cpp
    source/implementation/mutation.cpp
    source/implementation/random.cpp
    source/implementation/nondominatedsort.cpp
    source/performance/evaluation.cpp
    source/performance/nondominatedsort.cpp
    )
target_link_libraries(operon_test PRIVATE operon::operon doctest::doctest Ceres::ceres)
target_compile_features(operon_test PRIVATE cxx_std_17)
target_include_directories(operon_test PRIVATE ${PROJECT_SOURCE_DIR}/source/thirdparty)
target_compile_options(operon_test PRIVATE "-march=x86-64;-mavx2;-mfma")
target_link_options(operon_test PUBLIC "-Wl,--no-undefined")

add_test(NAME operon_test COMMAND operon_test)
windows_set_path(operon_test operon::operon)
