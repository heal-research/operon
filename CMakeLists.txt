# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

cmake_minimum_required(VERSION 3.14)

include(cmake/prelude.cmake)

project(
    operon
    VERSION 0.3.1
    DESCRIPTION "Fast and scalable genetic programming library for symbolic regression."
    HOMEPAGE_URL "https://operongp.readthedocs.io/en/latest/"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED True)

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
    source/core/dataset.cpp
    source/core/distance.cpp
    source/core/format.cpp
    source/core/node.cpp
    source/core/pset.cpp
    source/core/tree.cpp
    source/core/version.cpp
    source/hash/hash.cpp
    source/hash/metrohash64.cpp
    source/operators/creator/balanced.cpp
    source/operators/creator/koza.cpp
    source/operators/creator/ptc2.cpp
    source/operators/crossover.cpp
    source/operators/evaluator.cpp
    source/operators/generator/basic.cpp
    source/operators/generator/brood.cpp
    source/operators/generator/os.cpp
    source/operators/generator/poly.cpp
    source/operators/mutation.cpp
    source/operators/non_dominated_sorter/deductive_sort.cpp
    source/operators/non_dominated_sorter/dominance_degree_sort.cpp
    source/operators/non_dominated_sorter/efficient_sort.cpp
    source/operators/non_dominated_sorter/hierarchical_sort.cpp
    source/operators/non_dominated_sorter/merge_sort.cpp
    source/operators/non_dominated_sorter/rank_intersect.cpp
    source/operators/non_dominated_sorter/rank_ordinal.cpp
    source/operators/selector/proportional.cpp
    source/operators/selector/tournament.cpp
)
add_library(operon::operon ALIAS operon_operon)

# ---- Required dependencies ----

find_package(Eigen3 REQUIRED)
find_package(FastFloat REQUIRED)
find_package(Taskflow REQUIRED)
find_package(Threads REQUIRED)
find_package(fmt REQUIRED)
find_package(pratt-parser REQUIRED)
find_package(robin_hood REQUIRED)
find_package(span-lite REQUIRED)
find_package(vstat REQUIRED)

find_package(vectorclass)
if (vectorclass_FOUND)
    target_link_libraries(operon_operon PRIVATE vectorclass::vectorclass)
else ()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(vectorclass REQUIRED IMPORTED_TARGET vectorclass)
    target_link_libraries(operon_operon PRIVATE PkgConfig::vectorclass)
endif ()

find_package(xxHash)
if (xxHash_FOUND)
    target_link_libraries(operon_operon PRIVATE xxHash::xxhash)
else ()
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
endif ()

find_package(aria-csv)
if (aria-csv_FOUND)
    target_link_libraries(operon_operon PRIVATE aria-csv::aria-csv)
else()
    find_path(ARIA_CSV_INCLUDE_DIRS "aria-csv/parser.hpp")
    target_include_directories(operon_operon PRIVATE "${ARIA_CSV_INCLUDE_DIRS}")
endif()

# ---- Optional dependencies
set(HAVE_CERES FALSE)
if (USE_CERES_NNLS)
    find_package(Ceres)   # use the Ceres optimizer for coefficients tuning
    if (Ceres_FOUND)
        set(HAVE_CERES TRUE)
        target_link_libraries(operon_operon PRIVATE Ceres::ceres)
    endif()
else()
    set(Ceres_VERSION "n/a")
endif()

if (USE_OPENLIBM)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(openlibm IMPORTED_TARGET openlibm)
        if (openlibm_FOUND)
            target_link_libraries(operon_operon PRIVATE PkgConfig::openlibm)
        endif()
    endif()
endif()

if (USE_JEMALLOC)
    find_package(PkgConfig)
    if(PkgConfig_FOUND)
        pkg_check_modules(jemalloc IMPORTED_TARGET jemalloc)
        if (jemalloc_FOUND)
            target_link_libraries(operon_operon PRIVATE PkgConfig::jemalloc)
        endif()
    endif()
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

if(Ceres_FOUND AND USE_CERES_NNLS)
endif()

target_link_libraries(operon_operon PUBLIC
    Eigen3::Eigen
    Threads::Threads
    fmt::fmt
    nonstd::span-lite
    pratt-parser::pratt-parser
    robin_hood::robin_hood
    )

target_link_libraries(operon_operon PRIVATE
    FastFloat::fast_float
    Taskflow::Taskflow
    vstat::vstat
    )

target_compile_features(operon_operon PUBLIC cxx_std_17)
set_target_properties(operon_operon PROPERTIES POSITION_INDEPENDENT_CODE ${BUILD_SHARED_LIBS})

if(MSVC)
    target_compile_options(operon_operon PUBLIC "/std:c++latest")
else()
    target_compile_options(operon_operon PUBLIC "-march=x86-64;-mavx2;-mfma")
    target_link_options(operon_operon PUBLIC "-Wl,--no-undefined")
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
