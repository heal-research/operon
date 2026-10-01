# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

cmake_minimum_required(VERSION 3.25)

include(cmake/prelude.cmake)

project(
    operon
    VERSION 0.3.1
    DESCRIPTION "Fast and scalable genetic programming library for symbolic regression."
    HOMEPAGE_URL "https://heal-research.github.io/operon/"
    LANGUAGES CXX
)

set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED TRUE)

include(cmake/project-is-top-level.cmake)

option(BUILD_DOCS "Build the documentation site." OFF)
if(PROJECT_IS_TOP_LEVEL AND BUILD_DOCS)
  include(cmake/docs.cmake)
endif()

docs_early_return()

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
    source/algorithms/domain_pruning.cpp
    source/algorithms/enumeration.cpp
    source/algorithms/gp.cpp
    source/algorithms/nsga2.cpp
    source/algorithms/probes/builtin.cpp
    source/algorithms/probes/chain.cpp
    source/algorithms/probes/population_trace.cpp
    source/algorithms/probes/registry.cpp
    source/algorithms/solution_archive.cpp
    source/core/dataset.cpp
    source/core/distance.cpp
    source/core/serialization.cpp
    source/core/tree_diff.cpp
    source/core/version.cpp
    source/formatter/dot.cpp
    source/formatter/formatter.cpp
    source/formatter/infix.cpp
    source/formatter/postfix.cpp
    source/formatter/tree.cpp
    source/hash/content_hash.cpp
    source/hash/zobrist.cpp
    source/interpreter/affine_evaluator.cpp
    source/interpreter/interpreter.cpp
    source/interpreter/interval_evaluator.cpp
    source/interpreter/range_tightening.cpp
    source/operators/creator/creator.cpp
    source/operators/creator/balanced.cpp
    source/operators/creator/koza.cpp
    source/operators/creator/ptc2.cpp
    source/operators/crossover.cpp
    source/operators/evaluator_error_metrics.cpp
    source/operators/evaluator.cpp
    source/operators/linear_scaling.cpp
    source/operators/generator/basic.cpp
    source/operators/generator/brood.cpp
    source/operators/generator/os.cpp
    source/operators/generator/poly.cpp
    source/operators/local_search.cpp
    source/operators/mutation.cpp
    source/operators/shape_constrained_evaluator.cpp
    source/operators/non_dominated_sorter/ndsort.cpp
    source/operators/selector/proportional.cpp
    source/operators/selector/tournament.cpp
    source/parser/infix.cpp
)
# operon_operon is the compiled backend-adapter implementation: interpreter,
# JIT, LM/optimizer machinery and everything that carries Eigen/Ceres/L-BFGS.
# It is not aliased as operon::operon directly -- see the aggregate target
# below, which combines it with operon::core under that name.
add_library(operon::backend_adapter ALIAS operon_operon)
# Public headers are an explicit target-owned interface. CONFIGURE_DEPENDS
# keeps the list current when a contributor adds or removes a header without
# making the installation layout depend on an untracked directory copy.
file(GLOB_RECURSE operon_installed_headers CONFIGURE_DEPENDS
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/*.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/*.hpp"
)

# Implementation and vendored headers remain installed because public template
# headers include them, but users do not include them directly. Keep them out
# of the standalone-header contract: their owning facade covers them instead.
set(operon_detail_headers ${operon_installed_headers})
list(FILTER operon_detail_headers INCLUDE REGEX "/(ceres|detail|backend/(eigen|eve|mad_eve|plain))/")
set(operon_public_headers ${operon_installed_headers})
list(FILTER operon_public_headers EXCLUDE REGEX "/(ceres|detail|backend/(eigen|eve|mad_eve|plain))/")

# ---- Core (compiled, Eigen/Ceres/L-BFGS-free tree/grammar/hash implementation plus header-only numerical/view contracts) ----
# A curated allow-list, deliberately NOT a glob like operon_installed_headers
# above: a glob would silently let a newly-added Eigen-touching header slip
# into this target's public interface the moment it is dropped into one of
# these directories, quietly breaking the Eigen-free guarantee. Every entry
# here must be verified (by inspection of its own #includes, transitively)
# to never reach Eigen, Ceres, or L-BFGS. See
# test/package-consumer/core for the installed-package contract
# test that enforces this.
set(operon_core_headers
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/algorithms/enumeration_canonicalizer.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/collections/bitset.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/aligned_allocator.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/concepts.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/constants.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/contracts.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/grammar.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/interpreter_error.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/memory_view.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/node.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/pset.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/range.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/standard_library.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/subtree.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/tree.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/types.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/view_descriptor.h"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/view_descriptor.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/hash/hash.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/hash/metrohash64.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/mdspan/mdspan.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/optimizer/fisher_information.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/optimizer/gradient_cost.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/optimizer/least_squares.hpp"
    "${CMAKE_CURRENT_SOURCE_DIR}/include/operon/optimizer/least_squares_gradient_adapter.hpp"
)

# Core headers are installed via operon_core below; keep
# them out of the backend-adapter's own header set so the same installed path
# is not claimed by two targets.
list(REMOVE_ITEM operon_public_headers ${operon_core_headers})

# A real compiled library (not header-only/INTERFACE): source/hash/metrohash64.cpp,
# source/hash/hash.cpp, source/core/node.cpp, source/core/tree.cpp,
# source/core/grammar.cpp, source/core/pset.cpp, and
# source/algorithms/enumeration_canonicalizer.cpp form one coherent,
# Eigen/Ceres/L-BFGS-free closure (tree/grammar/primitive-set representation
# plus the hashing it relies on) that the interpreter/dispatch machinery does
# not need to build. It gets its own export macro (core_export.hpp
# below), independent of operon_operon's OPERON_EXPORT.
add_library(
    operon_core
    source/algorithms/enumeration_canonicalizer.cpp
    source/core/grammar.cpp
    source/core/node.cpp
    source/core/pset.cpp
    source/core/tree.cpp
    source/hash/hash.cpp
    source/hash/metrohash64.cpp
)
add_library(operon::core ALIAS operon_core)

target_sources(operon_core PUBLIC
    FILE_SET operon_core_headers
    TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES ${operon_core_headers}
)

target_include_directories(
    operon_core ${operon_warning_guard}
    PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_SOURCE_DIR}/include>"
)

target_include_directories(operon_core SYSTEM PUBLIC
    "$<BUILD_INTERFACE:${PROJECT_BINARY_DIR}/export>"
)

target_compile_features(operon_core PUBLIC cxx_std_23)

# Operon::Scalar (core/types.hpp, a core header) is float or double
# depending on this flag. PUBLIC and set here (not on operon_operon): every
# translation unit that defines or calls a Tree/Node/PrimitiveSet member
# taking or returning Operon::Scalar must agree on its size, including
# operon_core's own node.cpp/tree.cpp/pset.cpp -- otherwise the two
# compiled libraries silently disagree on Scalar's width (an ODR violation
# that surfaces as a mismatched-signature undefined reference at link time,
# e.g. Tree::SetCoefficients(span<double const>) vs the caller's
# span<float const>). operon_oper still sees this definition because it
# links operon::core PUBLIC.
target_compile_definitions(operon_core PUBLIC
    "$<$<BOOL:${USE_SINGLE_PRECISION}>:USE_SINGLE_PRECISION>"
)

target_sources(operon_operon PUBLIC
    FILE_SET operon_headers
    TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES ${operon_public_headers}
    FILE_SET operon_detail_headers
    TYPE HEADERS
    BASE_DIRS "${CMAKE_CURRENT_SOURCE_DIR}/include"
    FILES ${operon_detail_headers}
)

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
find_package(fmt 10 REQUIRED)
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

# operon_core's dependency set is intentionally a small subset of
# operon_operon's: header-only or near-header-only libraries only, nothing
# that could reintroduce Eigen/Ceres/L-BFGS through the back door. PUBLIC:
# used by the core public headers below, so consumers need them
# too (see test/package-consumer/core).
target_link_libraries(operon_core PUBLIC
    fluky::fluky
    fmt::fmt
    libassert::assert
    Microsoft.GSL::GSL
    std::mdspan
    Threads::Threads
    tl::expected
    unordered_dense::unordered_dense
)

# gtl (node.cpp's Descriptions() registry) and xxHash (hash.cpp, compiled
# with XXH_INLINE_ALL so it never leaves an unresolved runtime symbol) are
# implementation details of core's own .cpp files, not used by any
# core public header -- PRIVATE, so a core-only consumer never needs to
# find_package() either of them (see install-config.cmake.in).
target_link_libraries(operon_core PRIVATE
    gtl::gtl
)

if(xxHash_FOUND)
    target_link_libraries(operon_core PRIVATE xxHash::xxhash)
else()
    find_package(PkgConfig REQUIRED)
    pkg_check_modules(xxhash IMPORTED_TARGET xxhash)
    if (NOT xxhash_FOUND)
        pkg_check_modules(xxhash IMPORTED_TARGET libxxhash)
    endif()
    if(xxhash_FOUND)
        target_link_libraries(operon_core PRIVATE PkgConfig::xxhash)
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

# operon_core is its own compiled library now, not a header-only
# INTERFACE target folded into operon_operon's visibility -- it needs its own
# export macro (OPERON_CORE_EXPORT, distinct from OPERON_EXPORT
# above) so hidden-visibility symbols in node.hpp/tree.hpp/grammar.hpp/
# pset.hpp/hash.hpp/enumeration_canonicalizer.hpp are exported correctly
# whether or not operon_operon is even built alongside it.
generate_export_header(
    operon_core
    BASE_NAME operon_core
    EXPORT_FILE_NAME export/operon/core_export.hpp
    CUSTOM_CONTENT_FROM_VARIABLE pragma_suppress_c4251
)

# ---- Timestamp the current build ----
string(TIMESTAMP OPERON_BUILD_TIMESTAMP "%Y-%m-%dT%H:%M:%SZ")

# ---- Add build information ----
configure_file(${CMAKE_CURRENT_SOURCE_DIR}/include/operon/core/buildinfo.hpp.in
               ${CMAKE_BINARY_DIR}/buildinfo.hpp)


if(NOT BUILD_SHARED_LIBS)
  target_compile_definitions(operon_operon PUBLIC OPERON_STATIC_DEFINE)
  target_compile_definitions(operon_core PUBLIC OPERON_CORE_STATIC_DEFINE)
endif()

set_target_properties(
    operon_operon PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME backend_adapter
    OUTPUT_NAME operon
)

set_target_properties(
    operon_core PROPERTIES
    CXX_VISIBILITY_PRESET hidden
    VISIBILITY_INLINES_HIDDEN YES
    VERSION "${PROJECT_VERSION}"
    SOVERSION "${PROJECT_VERSION_MAJOR}"
    EXPORT_NAME core
    OUTPUT_NAME operon_core
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
    operon::core
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
    target_compile_options(operon_core PRIVATE "/std:c++latest")
else()
    if (UNIX AND NOT APPLE)
        target_link_options(operon_operon PRIVATE "-Wl,--no-undefined")
        target_link_options(operon_core PRIVATE "-Wl,--no-undefined")
    endif()
    target_compile_options(operon_operon PRIVATE "-fno-math-errno")
endif()

# FastSinCos's range reduction (interpreter/backend/eve/functions.hpp)
# requires genuine hardware FMA, and Eve needs AVX2 to select that code
# path. PUBLIC: FastSinCos is instantiated from public headers, so
# consumers need the flag too. 64-bit x86_64/non-Windows only:
# -march=x86-64-v3 is 64-bit-only (excludes i386/i486/.../i686); Windows
# gets /arch:AVX2 via the build-windows preset.
#
# Raises the minimum supported CPU to Haswell-era (2013+) x86-64 - see
# BUILDING.md.
if (CMAKE_SYSTEM_PROCESSOR MATCHES "^(x86_64|AMD64)$" AND NOT WIN32)
    target_compile_options(operon_operon PUBLIC "-march=x86-64-v3")
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
    "$<$<BOOL:${HAVE_ASMJIT}>:HAVE_ASMJIT>"
    )

# The generated export header has no source-tree counterpart. Verify every
# facade header in developer builds against the target's complete published
# usage requirements; internal implementation headers are covered by their
# owning facade, and the generated header is installed explicitly.
if(operon_DEVELOPER_MODE)
    set_property(TARGET operon_operon PROPERTY VERIFY_INTERFACE_HEADER_SETS ON)
    set_property(TARGET operon_operon PROPERTY INTERFACE_HEADER_SETS_TO_VERIFY operon_headers)
    set_property(TARGET operon_core PROPERTY VERIFY_INTERFACE_HEADER_SETS ON)
    set_property(TARGET operon_core PROPERTY INTERFACE_HEADER_SETS_TO_VERIFY operon_core_headers)
endif()

# ---- Aggregate umbrella target ----
# operon::operon is what most consumers want: the full-featured aggregate of
# the compiled, Eigen/Ceres/L-BFGS-free core (tree/grammar/
# primitive-set/hash implementation plus header-only numerical/view
# contracts) and the compiled backend-adapter (interpreter, JIT,
# LM/optimizer, Eigen/Ceres/L-BFGS). It carries no code or headers of its
# own -- it only forwards to the two targets above -- so a
# consumer linking operon::operon sees exactly the union of their usage
# requirements, unchanged from before this target existed.
add_library(operon_aggregate INTERFACE)
add_library(operon::operon ALIAS operon_aggregate)
target_link_libraries(operon_aggregate INTERFACE
    operon::core
    operon::backend_adapter
)
set_target_properties(operon_aggregate PROPERTIES EXPORT_NAME operon)

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
