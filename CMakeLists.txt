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
find_package (Python3 COMPONENTS Development REQUIRED)

FetchContent_Declare(
    gsl
    GIT_REPOSITORY https://github.com/microsoft/GSL.git
    GIT_TAG master
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/GSL
    SOURCE_DIR ${PROJECT_SOURCE_DIR}/thirdparty/GSL
)

FetchContent_Declare(
    csv
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/CSV/include
    URL https://raw.githubusercontent.com/vincentlaucsb/csv-parser/master/single_include/csv.hpp
    DOWNLOAD_NO_EXTRACT 1
)

FetchContent_Declare(
    xxhash 
    DOWNLOAD_DIR ${PROJECT_SOURCE_DIR}/thirdparty/xxhash
    URL https://raw.githubusercontent.com/RedSpah/xxhash_cpp/master/xxhash/xxhash.hpp 
    DOWNLOAD_NO_EXTRACT 1
)

FetchContent_MakeAvailable(xxhash gsl csv)

set(THIRDPARTY_INCLUDE_DIRS
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/GSL/include
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/CSV/include
    PRIVATE ${PROJECT_SOURCE_DIR}/thirdparty/
)

#set(CMAKE_CXX_CLANG_TIDY "clang-tidy;-checks=bugprone-*,cppcoreguidelines-*,modernize-*,performance-*,readability-*")

add_library(
    operon
    SHARED
    src/core/metrics.cpp
    src/core/tree.cpp
    src/core/problem.cpp
    src/core/dataset.cpp
    src/operators/crossover.cpp
    src/operators/mutation.cpp
)
target_compile_features(operon PRIVATE cxx_std_17)
target_link_libraries(operon PRIVATE fmt::fmt ${CERES_LIBRARIES} Threads::Threads TBB::tbb)
target_include_directories(
    operon
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/include/operon
    ${THIRDPARTY_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)
# necessary to prevent -isystem introduced by intel-tbb
set_target_properties(operon PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

add_library(
    pyoperon 
    SHARED
    python/operon.cpp
)
target_compile_features(pyoperon PRIVATE cxx_std_17)
target_link_libraries(pyoperon PRIVATE operon TBB::tbb) # figure out if tbb is necessary for linkage here
target_include_directories(
    pyoperon 
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/include/operon
    ${PROJECT_SOURCE_DIR}/python
    ${THIRDPARTY_INCLUDE_DIRS}
    ${Python3_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)

#binary for GP algorithm cli version
add_executable(
    operon-gp
    src/cli/operon_gp.cpp
)
target_compile_features(operon-gp PRIVATE cxx_std_17)
target_link_libraries(operon-gp PRIVATE operon fmt::fmt cxxopts::cxxopts ${CERES_LIBRARIES} TBB::tbb)
target_include_directories(
    operon-gp 
    PRIVATE
    ${PROJECT_SOURCE_DIR}/include/operon
    ${THIRDPARTY_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)
set_target_properties(operon-gp PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

find_package(Catch2 REQUIRED)
add_executable(
    operon-test
    test/test.cpp
    test/performance/evaluation.cpp
    test/performance/initialization.cpp
    test/performance/hashing.cpp
    test/performance/distance.cpp
    test/implementation/evaluation.cpp
    test/implementation/details.cpp
    test/implementation/hashing.cpp
    test/implementation/initialization.cpp
    test/implementation/selection.cpp
)
target_compile_features(operon-test PRIVATE cxx_std_17)
target_link_libraries(operon-test PRIVATE operon fmt::fmt Catch2::Catch2 ${CERES_LIBRARIES} TBB::tbb)
target_include_directories(
    operon-test
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/include/operon
    ${THIRDPARTY_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)
target_compile_definitions(operon-test PRIVATE CATCH_CONFIG_ENABLE_BENCHMARKING)
set_target_properties(operon-test PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

add_executable(
    operon-example-gp
    examples/gp.cpp
    )
target_compile_features(operon-example-gp PRIVATE cxx_std_17)
target_link_libraries(operon-example-gp PRIVATE operon fmt::fmt cxxopts::cxxopts ${CERES_LIBRARIES} TBB::tbb)
target_include_directories(
    operon-example-gp 
    PRIVATE 
    ${PROJECT_SOURCE_DIR}/include/operon
    ${THIRDPARTY_INCLUDE_DIRS}
    ${CERES_INCLUDE_DIRS}
)
set_target_properties(operon-example-gp PROPERTIES NO_SYSTEM_FROM_IMPORTED TRUE)

if(MSVC)
    target_compile_options(operon PRIVATE /W4 "$<$<CONFIG:Release>:/O2;/std:c++latest>")
    target_compile_options(operon-gp PRIVATE /W4 "$<$<CONFIG:Release>:/O2;/std:c++latest>")
    target_compile_options(operon-test PRIVATE /W4 "$<$<CONFIG:Release>:/O2;/std:c++latest>")
    # ugly hack to set the include dir correctly
    find_path(FMT_INCLUDE_DIR fmt/core.h)
    include_directories(${FMT_INCLUDE_DIR})
else()
    set(MYFLAGS -Wall -Wextra -Wno-unknown-pragmas -Wno-deprecated -Wno-deprecated-copy)
    target_compile_options(operon PRIVATE ${MYFLAGS} "$<$<CONFIG:Debug>:-g;--coverage>$<$<CONFIG:Release>:-O3;-g;-march=native>")
    target_link_libraries(operon PRIVATE "$<$<CONFIG:Debug>:gcov>")
    target_compile_options(operon-gp PRIVATE ${MYFLAGS} "$<$<CONFIG:Debug>:-g;--coverage>$<$<CONFIG:Release>:-O3;-g;-march=native>")
    target_link_libraries(operon-gp PRIVATE "$<$<CONFIG:Debug>:gcov>")
    target_compile_options(operon-test PRIVATE ${MYFLAGS} "$<$<CONFIG:Debug>:-g;--coverage>$<$<CONFIG:Release>:-O3;-g;-march=native>")
    target_compile_options(pyoperon PRIVATE ${MYFLAGS} "$<$<CONFIG:Debug>:-g;-fPIC;--coverage>$<$<CONFIG:Release>:-O3;-g;-fPIC;-march=native>")
    target_compile_options(operon-example-gp PRIVATE ${MYFLAGS} "$<$<CONFIG:Debug>:-g;--coverage>$<$<CONFIG:Release>:-O3;-g;-march=native>")
endif(MSVC)

include(CTest)
include(Catch)
catch_discover_tests(operon-test)
