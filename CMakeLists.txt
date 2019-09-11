cmake_minimum_required(VERSION 3.11)

include(FetchContent)

project(
    Operon
    VERSION 0.1
    DESCRIPTION "Genetic Programming"
    LANGUAGES CXX
)

find_package(Git)
find_package(cxxopts REQUIRED)
find_package(fmt REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Ceres REQUIRED)
find_package(TBB REQUIRED)
find_package(Threads REQUIRED)

FetchContent_Declare(
    gsl
    GIT_REPOSITORY https://github.com/microsoft/GSL.git
    GIT_TAG master
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/GSL
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/GSL
)

FetchContent_Declare(
    csv
    GIT_REPOSITORY https://github.com/vincentlaucsb/csv-parser.git 
    GIT_TAG master
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/CSV
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/CSV
)

FetchContent_Declare(
    xxhash 
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/xxhash
    URL https://raw.githubusercontent.com/RedSpah/xxhash_cpp/master/xxhash/xxhash.hpp 
    DOWNLOAD_NO_EXTRACT 1
)

FetchContent_MakeAvailable(xxhash gsl csv)
# prevent compilation of unused targets
set_target_properties(csv PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(csv_stats PROPERTIES EXCLUDE_FROM_ALL TRUE)
set_target_properties(csv_info PROPERTIES EXCLUDE_FROM_ALL TRUE)

set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=bugprone-*,cppcoreguidelines-*,modernize-*,performance-*,readability-*")

add_library(
    operon
    STATIC 
    src/core/tree.cpp
    src/core/problem.cpp
    src/core/dataset.cpp
    src/operators/initialization.cpp
    src/operators/crossover.cpp
)
target_compile_features(operon PRIVATE cxx_std_17)
target_link_libraries(operon PRIVATE csv fmt::fmt ${CERES_LIBRARIES} Threads::Threads TBB::tbb)
target_include_directories(
    operon
    PRIVATE ${PROJECT_SOURCE_DIR}/include/operon
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/GSL/include
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/CSV/include
    PRIVATE ${CERES_INCLUDE_DIRS}
)
# necessary to prevent -isystem introduced by intel-tbb
set_target_properties(operon PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

#binary for GP algorithm cli version
add_executable(
    operon-gp
    src/cli/operon_gp.cpp
)
target_compile_features(operon-gp PRIVATE cxx_std_17)
target_link_libraries(operon-gp PRIVATE operon cxxopts::cxxopts TBB::tbb)
target_include_directories(
    operon-gp 
    PRIVATE ${PROJECT_SOURCE_DIR}/include/operon
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/GSL/include
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/CSV/include
    PRIVATE ${CERES_INCLUDE_DIRS}
)
set_target_properties(operon-gp PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

# binary for OSGP algorithm cli version
add_executable(
    operon-osgp
    src/cli/operon_osgp.cpp
)
target_compile_features(operon-osgp PRIVATE cxx_std_17)
target_link_libraries(operon-osgp PRIVATE operon cxxopts::cxxopts TBB::tbb)
target_include_directories(
    operon-osgp 
    PRIVATE ${PROJECT_SOURCE_DIR}/include/operon
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/GSL/include
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/CSV/include
    PRIVATE ${CERES_INCLUDE_DIRS}
)
set_target_properties(operon-osgp PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

find_package(Catch2 REQUIRED)
add_executable(
    operon-test
    test/test.cpp
    test/performance.cpp
    test/evaluation.cpp
    test/initialization.cpp
)
target_compile_features(operon-test PRIVATE cxx_std_17)
target_link_libraries(operon-test PRIVATE operon Catch2::Catch2 TBB::tbb)
target_include_directories(
    operon-test
    PRIVATE ${PROJECT_SOURCE_DIR}/include/operon
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/GSL/include
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/CSV/include
    PRIVATE ${CERES_INCLUDE_DIRS}
)
target_compile_definitions(operon-test PRIVATE CATCH_CONFIG_FAST_COMPILE CATCH_CONFIG_ENABLE_BENCHMARKING)
set_target_properties(operon-test PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

if(MSVC)
    target_compile_options(operon PRIVATE /W4 "$<$<CONFIG:Release>:/O2;/std:c++latest>")
    target_compile_options(operon-gp PRIVATE /W4 "$<$<CONFIG:Release>:/O2;/std:c++latest>")
    target_compile_options(operon-osgp PRIVATE /W4 "$<$<CONFIG:Release>:/O2;/std:c++latest>")
    target_compile_options(operon-test PRIVATE /W4 "$<$<CONFIG:Release>:/O2;/std:c++latest>")
    # ugly hack to set the include dir correctly
    find_path(FMT_INCLUDE_DIR fmt/core.h)
    include_directories(${FMT_INCLUDE_DIR})
else()
    target_compile_options(operon PRIVATE -Wall -Wextra -Wno-unknown-pragmas "$<$<CONFIG:Debug>:-Og;-g;--coverage>$<$<CONFIG:Release>:-O3;-march=native>")
    target_link_libraries(operon PRIVATE "$<$<CONFIG:Debug>:gcov>")
    target_compile_options(operon-gp PRIVATE -Wall -Wextra -Wno-unknown-pragmas "$<$<CONFIG:Debug>:-Og;-g;--coverage>$<$<CONFIG:Release>:-O3;-march=native>")
    target_link_libraries(operon-gp PRIVATE "$<$<CONFIG:Debug>:gcov>")
    target_compile_options(operon-osgp PRIVATE -Wall -Wextra -Wno-unknown-pragmas "$<$<CONFIG:Debug>:-Og;-g;--coverage>$<$<CONFIG:Release>:-O3;-march=native>")
    target_link_libraries(operon-osgp PRIVATE "$<$<CONFIG:Debug>:gcov>")
    target_compile_options(operon-test PRIVATE -Wall -Wextra -Wno-unknown-pragmas "$<$<CONFIG:Debug>:-Og;-g;--coverage>$<$<CONFIG:Release>:-O3;-march=native>")
endif(MSVC)

include(CTest)
include(Catch)
catch_discover_tests(operon-test)
