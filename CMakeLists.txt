# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

cmake_minimum_required(VERSION 3.25)

include(cmake/prelude.cmake)

project(
    operon
    VERSION 0.3.1
    DESCRIPTION "Fast and scalable genetic programming library for symbolic regression."
    HOMEPAGE_URL "https://operongp.readthedocs.io/en/latest/"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
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
    source/algorithms/enumeration.cpp
    source/algorithms/gp.cpp
    source/algorithms/nsga2.cpp
    source/algorithms/solution_archive.cpp
    source/core/dataset.cpp
    source/core/distance.cpp
    source/core/grammar.cpp
    source/core/node.cpp
    source/core/pset.cpp
    source/core/tree.cpp
    source/core/serialization.cpp
    source/core/tree_diff.cpp
    source/core/version.cpp
    source/formatter/dot.cpp
    source/formatter/infix.cpp
    source/formatter/postfix.cpp
    source/formatter/tree.cpp
    source/hash/content_hash.cpp
    source/hash/hash.cpp
    source/hash/metrohash64.cpp
    source/hash/zobrist.cpp
    source/interpreter/interpreter.cpp
    source/operators/creator/creator.cpp
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
    source/operators/non_dominated_sorter/ndsort.cpp
    source/operators/selector/proportional.cpp
    source/operators/selector/tournament.cpp
    source/parser/infix.cpp
)
add_library(operon::operon ALIAS operon_operon)

# ---- Required dependencies ----
find_package(AriaCsvParser REQUIRED)
find_package(cpp-sort REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(eve REQUIRED)
find_package(FastFloat REQUIRED)
find_package(fluky REQUIRED)
find_package(glaze REQUIRED)
# Glaze declares cxx_std_23 in its INTERFACE_COMPILE_FEATURES. Strip it so
# linking glaze PRIVATE does not upgrade the whole library to C++23; instead
# we set CXX_STANDARD 23 on serialization.cpp alone via source properties.
set_target_properties(glaze::glaze PROPERTIES INTERFACE_COMPILE_FEATURES "")
find_package(fmt REQUIRED)
find_package(gtl REQUIRED)
find_package(lbfgs REQUIRED)
find_package(libassert REQUIRED)
find_package(mdspan REQUIRED)
find_package(Microsoft.GSL CONFIG REQUIRED)
find_package(infix-parser REQUIRED)
find_package(Taskflow REQUIRED)
find_package(Threads REQUIRED)
find_package(tl-expected REQUIRED)
find_package(unordered_dense REQUIRED)
find_package(ndsort REQUIRED)
find_package(vstat REQUIRED)
find_package(xxHash)

# ---- pappus (interval/affine arithmetic backends) ----
# pappus (https://github.com/heal-research/pappus) provides `pappus::interval<T>`
# and `pappus::affine_form<T>`, used by the public IntervalEvaluator /
# AffineEvaluator headers under include/operon/interpreter/. Those headers
# unconditionally `#include <pappus/pappus.hpp>` with no fallback, so pappus
# is a real public dependency of operon_operon, not an optional add-on.
find_package(pappus REQUIRED)

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
set(HAVE_ASMJIT FALSE)
if (USE_ASMJIT)
    if (NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64|i.86)$")
        message(STATUS "USE_ASMJIT requested but asmjit JIT-compiles to native x86-64; "
                       "disabling on ${CMAKE_SYSTEM_PROCESSOR}")
    elseif (WIN32)
        # jit_compiler.cpp's AVX2 intrinsics + heavy inlined Cephes-style
        # polynomials cause clang-cl's optimizer to hang indefinitely (not a
        # timeout - observed stuck for 1h45m+ on GitHub Actions windows-latest
        # runners). Disable until this can be debugged on real Windows.
        message(STATUS "USE_ASMJIT requested but disabled on Windows: clang-cl "
                       "hangs compiling jit_compiler.cpp (unresolved)")
    else()
        if (NOT USE_SINGLE_PRECISION)
            message(FATAL_ERROR "USE_ASMJIT requires USE_SINGLE_PRECISION=ON")
        endif()
        find_package(asmjit REQUIRED)
        set(HAVE_ASMJIT TRUE)
    endif()
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
    set(MATH_BACKEND "Eve")
endif()

message(STATUS "MATH: ${MATH_BACKEND}")

if (MATH_BACKEND STREQUAL "Eve")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_EVE)
elseif (MATH_BACKEND STREQUAL "MadEve")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_MAD_EVE)
elseif (MATH_BACKEND STREQUAL "Eigen")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_EIGEN)
elseif (MATH_BACKEND STREQUAL "Stl")
    target_compile_definitions(operon_operon PUBLIC OPERON_MATH_STL)
else()
    message(FATAL_ERROR "Unknown MATH_BACKEND: ${MATH_BACKEND}. Valid options: Eve, MadEve, Eigen, Stl")
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
    infix-parser::infix-parser # required by infix parser
    vstat::vstat
    std::mdspan
    pappus::pappus # required by IntervalEvaluator/AffineEvaluator
    tl::expected # Dataset::GetVariable returns tl::expected in the public dataset.hpp header
)

target_link_libraries(operon_operon PRIVATE
    AriaCsvParser::AriaCsvParser
    glaze::glaze
    ndsort::ndsort
    Taskflow::Taskflow
    cpp-sort::cpp-sort
    unordered_dense::unordered_dense
    gtl::gtl
)

target_compile_features(operon_operon PUBLIC cxx_std_23)

if(MSVC)
    target_compile_options(operon_operon PRIVATE "/std:c++latest")
else()
    if (UNIX AND NOT APPLE)
        target_link_options(operon_operon PRIVATE "-Wl,--no-undefined")
    endif()
    target_compile_options(operon_operon PRIVATE "-fno-math-errno")
endif()

if (HAVE_ASMJIT)
    target_link_libraries(operon_operon PUBLIC asmjit::asmjit)
    target_sources(operon_operon PRIVATE
        source/interpreter/backend/jit/jit_compiler.cpp
        source/interpreter/backend/jit/jit_evaluator.cpp
        source/interpreter/backend/jit/jit_factory.cpp
    )
    # jit_compiler.cpp emits AVX2 intrinsics; guard explicitly so non-preset
    # builds (plain cmake -B build) still get AVX2 codegen. HAVE_ASMJIT is
    # never true on WIN32 (see above), so this only runs on Linux/macOS,
    # where the compiler is always clang or gcc, never real MSVC.
    set_source_files_properties(source/interpreter/backend/jit/jit_compiler.cpp
        PROPERTIES COMPILE_OPTIONS "-mavx2")
endif()

target_compile_definitions(operon_operon PUBLIC
    "$<$<BOOL:${USE_SINGLE_PRECISION}>:USE_SINGLE_PRECISION>"
    "$<$<BOOL:${HAVE_ASMJIT}>:HAVE_ASMJIT>"
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
