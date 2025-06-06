# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

cmake_minimum_required(VERSION 3.20)

include(cmake/prelude.cmake)

project(
    operon
    VERSION 0.3.1
    DESCRIPTION "Fast and scalable genetic programming library for symbolic regression."
    HOMEPAGE_URL "https://operongp.readthedocs.io/en/latest/"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

docs_early_return()

include(cmake/project-is-top-level.cmake)
include(cmake/variables.cmake)

# ---- Attempt to get revision information from git ----
find_package(Git)     # retrieve revision number for version info
if(GIT_FOUND AND EXISTS "${CMAKE_SOURCE_DIR}/.git")
    include(cmake/get-git-revision.cmake)
    EXECUTE_PROCESS(COMMAND ${GIT_EXECUTABLE} rev-parse --short HEAD OUTPUT_VARIABLE SHORT_SHA OUTPUT_STRIP_TRAILING_WHITESPACE)
    SET(REVISION ${SHORT_SHA} CACHE STRING "git short sha" FORCE)

    # only use the plugin to tie the configure state to the sha to force rebuilds
    # of files that depend on version.h
    get_git_head_revision(REFSPEC COMMITHASH.cmake)
else()
    message(WARNING "Git not found, cannot set version info")
    SET(REVISION "unknown")
endif()

# ---- Declare library ----

add_library(
    operon_operon
    source/algorithms/gp.cpp
    source/algorithms/nsga2.cpp
    source/algorithms/solution_archive.cpp
    source/core/dataset.cpp
    source/core/distance.cpp
    source/core/node.cpp
    source/core/pset.cpp
    source/core/tree.cpp
    source/core/version.cpp
    source/formatter/dot.cpp
    source/formatter/infix.cpp
    source/formatter/postfix.cpp
    source/formatter/tree.cpp
    source/hash/hash.cpp
    source/hash/metrohash64.cpp
    source/interpreter/interpreter.cpp
    source/operators/creator/balanced.cpp
    source/operators/creator/koza.cpp
    source/operators/creator/ptc2.cpp
    source/operators/crossover.cpp
    source/operators/evaluator_error_metrics.cpp
    source/operators/evaluator.cpp
    source/operators/generator/basic.cpp
    source/operators/generator/brood.cpp
    source/operators/generator/os.cpp
    source/operators/generator/poly.cpp
    source/operators/local_search.cpp
    source/operators/mutation.cpp
    source/operators/non_dominated_sorter/best_order_sort.cpp
    source/operators/non_dominated_sorter/deductive_sort.cpp
    source/operators/non_dominated_sorter/dominance_degree_sort.cpp
    source/operators/non_dominated_sorter/efficient_sort.cpp
    source/operators/non_dominated_sorter/hierarchical_sort.cpp
    source/operators/non_dominated_sorter/merge_sort.cpp
    source/operators/non_dominated_sorter/rank_intersect.cpp
    source/operators/non_dominated_sorter/rank_ordinal.cpp
    source/operators/selector/proportional.cpp
    source/operators/selector/tournament.cpp
    source/parser/infix.cpp
)
add_library(operon::operon ALIAS operon_operon)

# ---- Required dependencies ----
find_package(AriaCsvParser REQUIRED)
find_package(byte-lite REQUIRED)
find_package(cpp-sort REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(eve REQUIRED)
find_package(FastFloat REQUIRED)
find_package(fluky REQUIRED)
find_package(fmt REQUIRED)
find_package(lbfgs REQUIRED)
find_package(libassert REQUIRED)
find_package(mdspan REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(outcome REQUIRED)
find_package(pratt-parser REQUIRED)
find_package(span-lite REQUIRED)
find_package(Taskflow REQUIRED)
find_package(Threads REQUIRED)
find_package(unordered_dense REQUIRED)
find_package(vstat REQUIRED)
find_package(xxHash)

if(xxHash_FOUND)
    target_link_libraries(operon_operon PRIVATE xxHash::xxhash)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(xxhash IMPORTED_TARGET xxhash)
    if (NOT xxhash_FOUND)
        pkg_check_modules(xxhash IMPORTED_TARGET libxxhash)
    endif()
    if(xxhash_FOUND)
        target_link_libraries(operon_operon PRIVATE PkgConfig::xxhash)
    else()
        message(FATAL_ERROR "xxHash dependency could not be found.")
    endif()
endif()

# ---- Optional dependencies
set(HAVE_CERES FALSE)
if (USE_CERES)
    find_package(Ceres)   # use the Ceres optimizer for coefficients tuning
    if (Ceres_FOUND)
        set(HAVE_CERES TRUE)
    endif()
else()
    set(Ceres_VERSION "n/a")
endif()

if (USE_JEMALLOC)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(jemalloc IMPORTED_TARGET jemalloc)
        if (jemalloc_FOUND)
            target_link_libraries(operon_operon PUBLIC PkgConfig::jemalloc)
        endif()
    endif()
endif()

if(NOT MATH_BACKEND)
    set(MATH_BACKEND "Eigen")
endif()

message(STATUS "MATH: ${MATH_BACKEND}")

if (MATH_BACKEND STREQUAL "Arma")
    find_package(BLAS REQUIRED)
    find_package(LAPACK REQUIRED)
    find_package(Armadillo REQUIRED)
    target_link_libraries(operon_operon PUBLIC BLAS::BLAS LAPACK::LAPACK ${ADMADILLO_LIBRARIES})
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_ARMA)
elseif (MATH_BACKEND STREQUAL "Blaze")
    find_package(blaze REQUIRED)
    find_package(xsimd REQUIRED)
    target_link_libraries(operon_operon INTERFACE xsimd)
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_BLAZE BLAZE_USE_SHARED_MEMORY_PARALLELIZATION=0 BLAZE_USE_XSIMD=1 EIGEN_DONT_PARALLELIZE)
elseif (MATH_BACKEND STREQUAL "Fastor")
    find_package(Fastor REQUIRED)
    find_package(sleef REQUIRED)
    target_link_libraries(operon_operon PUBLIC Fastor::Fastor sleef::sleef)
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_FASTOR FASTOR_USE_SLEEF_U35)
elseif (MATH_BACKEND STREQUAL "Eve")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_EVE)
elseif (MATH_BACKEND STREQUAL "Vdt")
    find_package(vdt REQUIRED)
    target_link_libraries(operon_operon PUBLIC vdt::vdt)
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_VDT)
elseif (MATH_BACKEND STREQUAL "Eigen")
    message(STATUS "Using Eigen backend")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_EIGEN EIGEN_DONT_PARALLELIZE)
elseif(MATH_BACKEND STREQUAL "Stl")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_STL)
elseif (MATH_BACKEND STREQUAL "Fast_v1")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_FAST_V1)
elseif (MATH_BACKEND STREQUAL "Fast_v2")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_FAST_V2)
elseif (MATH_BACKEND STREQUAL "Fast_v3")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_FAST_V3)
endif()

# print summary of enabled/disabled features
feature_summary(WHAT ENABLED_FEATURES DESCRIPTION "Enabled features:" QUIET_ON_EMPTY)
feature_summary(WHAT DISABLED_FEATURES DESCRIPTION "Disabled features:" QUIET_ON_EMPTY)

include(GenerateExportHeader)
generate_export_header(
    operon_operon
    BASE_NAME operon
    EXPORT_FILE_NAME export/operon/operon_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE pragma_suppress_c4251
)

# ---- Timestamp the current build ----
string(TIMESTAMP OPERON_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ")

# ---- Add build information ----
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/buildinfo.hpp.in
               ${CMAKE_BINARY_DIR}/buildinfo.hpp)


if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(operon_operon PUBLIC OPERON_STATIC_DEFINE)
endif()

set_target_properties(
    operon_operon PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME operon
    OUTPUT_NAME operon
)

target_include_directories(
    operon_operon ${operon_warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(
    operon_operon SYSTEM
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_include_directories(
    operon_operon PRIVATE
    "${PROJECT_BINARY_DIR}"
)

if(Ceres_FOUND AND USE_CERES)
endif()

target_link_libraries(operon_operon INTERFACE
    eve::eve # required by vstat
)

target_link_libraries(operon_operon PUBLIC
    Eigen3::Eigen
    Threads::Threads
    fluky::fluky
    fmt::fmt
    lbfgs::lbfgs
    libassert::assert
    pratt-parser::pratt-parser # required by infix parser
    std::mdspan
    vstat::vstat
)

target_link_libraries(operon_operon PRIVATE
    AriaCsvParser::AriaCsvParser
    Taskflow::Taskflow
    cpp-sort::cpp-sort
    unordered_dense::unordered_dense
    nonstd::byte-lite
    nonstd::span-lite
    outcome::hl
)

if (USE_CERES AND Ceres_FOUND)
    target_link_libraries(operon_operon PUBLIC Ceres::ceres)
endif()

target_compile_features(operon_operon PUBLIC cxx_std_20)

if(MSVC)
    target_compile_options(operon_operon PRIVATE "/std:c++latest")
else()
    if (UNIX AND NOT APPLE)
        target_link_options(operon_operon PRIVATE "-Wl,--no-undefined")
    endif()
    target_compile_options(operon_operon PRIVATE "-fno-math-errno")
endif()

target_compile_definitions(operon_operon PUBLIC
    "$<$<BOOL:${USE_SINGLE_PRECISION}>:USE_SINGLE_PRECISION>"
    "$<$<BOOL:${HAVE_CERES}>:HAVE_CERES>"
    )

# ---- Install rules ----

if(NOT CMAKE_SKIP_INSTALL_RULES)
  include(cmake/install-rules.cmake)
endif()

# ---- Command-line programs ----

if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_CLI_PROGRAMS "Build command-line programs." TRUE)
  if (BUILD_CLI_PROGRAMS)
    add_subdirectory(cli)
  endif()
endif()

# ---- Examples ----

if(PROJECT_IS_TOP_LEVEL)
  option(BUILD_EXAMPLES "Build examples tree." "${operon_DEVELOPER_MODE}")
  if(BUILD_EXAMPLES)
    add_subdirectory(example)
  endif()
endif()

# ---- Developer mode ----

if(NOT operon_DEVELOPER_MODE)
  return()
elseif(NOT PROJECT_IS_TOP_LEVEL)
  message(
      AUTHOR_WARNING
      "Developer mode is intended for developers of operon"
  )
endif()

include(cmake/dev-mode.cmake)
