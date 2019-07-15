cmake_minimum_required(VERSION 3.11)

include(FetchContent)

project(
    Operon
    VERSION 0.1
    DESCRIPTION "Genetic Programming"
    LANGUAGES CXX
)

find_package(cxxopts REQUIRED)
find_package(fmt REQUIRED)
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)

FetchContent_Declare(
    xxhash 
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/xxhash
    URL https://raw.githubusercontent.com/RedSpah/xxhash_cpp/master/xxhash/xxhash.hpp 
    DOWNLOAD_NO_EXTRACT 1
    QUIET 0
)

FetchContent_GetProperties(xxhash)
if(NOT xxhash_Populated)
    FetchContent_Populate(xxhash)
endif()

add_subdirectory(${PROJECT_SOURCE_DIR}/thirdparty/csv-parser EXCLUDE_FROM_ALL)

add_library(
    operon
    STATIC 
    src/core/tree.cpp
    src/core/problem.cpp
    src/algorithms/crossover.cpp
    src/algorithms/mutation.cpp
    src/algorithms/initialization.cpp
    src/algorithms/selection.cpp
)
target_compile_features(operon PRIVATE cxx_std_17)
target_link_libraries(operon PRIVATE csv fmt::fmt ${CERES_LIBRARIES} TBB::tbb)
target_include_directories(
    operon
    PRIVATE ${PROJECT_SOURCE_DIR}/src/core
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty
    #    PRIVATE ${CERES_INCLUDE_DIRS}
)
# necessary to prevent -isystem introduced by intel-tbb
set_target_properties(operon PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

add_executable(
    operon-cli 
    src/cli/operon_cli.cpp
)
target_compile_features(operon-cli PRIVATE cxx_std_17)
target_link_libraries(operon-cli PRIVATE operon cxxopts::cxxopts)
target_include_directories(
    operon-cli 
    PRIVATE ${PROJECT_SOURCE_DIR}/src/core
    PRIVATE ${PROJECT_SOURCE_DIR}/src/algorithms
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty
)

if(MSVC)
    target_compile_options(operon PRIVATE /W4 "$<$<CONFIG:Release>:/O2>")
    target_compile_options(operon-cli PRIVATE /W4 "$<$<CONFIG:Release>:/O2>")
    # ugly hack to set the include dir correctly
    find_path(FMT_INCLUDE_DIR fmt/core.h)
    include_directories(${FMT_INCLUDE_DIR})
else()
    target_compile_options(operon PRIVATE -Wall -Wextra -Wno-unknown-pragmas "$<$<CONFIG:Debug>:-Og;-g;--coverage>$<$<CONFIG:Release>:-O3;-march=native>")
    target_link_libraries(operon PRIVATE "$<$<CONFIG:Debug>:gcov>")
    target_compile_options(operon-cli PRIVATE -Wall -Wextra -Wno-unknown-pragmas "$<$<CONFIG:Debug>:-Og;-g;--coverage>$<$<CONFIG:Release>:-O3;-march=native>")
    target_link_libraries(operon-cli PRIVATE "$<$<CONFIG:Debug>:gcov>")
endif(MSVC)

