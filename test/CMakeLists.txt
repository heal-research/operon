cmake_minimum_required(VERSION 3.20)

project(operonTests LANGUAGES CXX)

include(../cmake/project-is-top-level.cmake)
include(../cmake/windows-set-path.cmake)

if(PROJECT_IS_TOP_LEVEL)
  find_package(operon REQUIRED)
  enable_testing()
endif()

find_package(doctest REQUIRED)
find_package(scn REQUIRED)

add_executable(operon_test
    source/operon_test.cpp
    source/implementation/autodiff.cpp
    source/implementation/crossover.cpp
    source/implementation/details.cpp
    source/implementation/dispatch_table.cpp
    source/implementation/diversity.cpp
    source/implementation/evaluation.cpp
    source/implementation/error_metrics.cpp
    source/implementation/hashing.cpp
    source/implementation/infix_parser.cpp
    source/implementation/initialization.cpp
    source/implementation/mutation.cpp
    source/implementation/nondominatedsort.cpp
    source/implementation/poisson_regression.cpp
    source/implementation/random.cpp
    source/performance/autodiff.cpp
    source/performance/distance.cpp
    source/performance/evaluation.cpp
    source/performance/nondominatedsort.cpp
    )
target_link_libraries(operon_test PRIVATE operon::operon doctest::doctest scn::scn)
target_compile_features(operon_test PRIVATE cxx_std_20)
target_include_directories(operon_test PRIVATE ${PROJECT_SOURCE_DIR}/source/thirdparty)
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

add_test(NAME operon_test COMMAND operon_test)
windows_set_path(operon_test operon::operon)
