# ---- Developer mode ----

# Developer mode enables targets and code paths in the CMake scripts that are
# only relevant for the developer(s) of operon
# Targets necessary to build the project must be provided unconditionally, so
# consumers can trivially build and package the project
if(PROJECT_IS_TOP_LEVEL)
  option(operon_DEVELOPER_MODE "Enable developer mode" OFF)
  option(BUILD_SHARED_LIBS "Build shared libs." OFF)

  # compile and link-time options
  set(OPENLIBM_DESCRIPTION             "Link against Julia's openlibm, a high performance mathematical library [default=OFF].")
  set(JEMALLOC_DESCRIPTION             "Link against jemalloc, a general purpose malloc(3) implementation that emphasizes fragmentation avoidance and scalable concurrency support [default=OFF].")
  set(TCMALLOC_DESCRIPTION             "Link against tcmalloc (thread-caching malloc), a malloc(3) implementation that reduces lock contention for multi-threaded programs [default=OFF].")
  set(MIMALLOC_DESCRIPTION             "Link against mimalloc, a general purpose allocator with excellent performance characteristics [default=OFF].")
  set(USE_SINGLE_PRECISION_DESCRIPTION "Perform model evaluation using floats (single precision) instead of doubles. Great for reducing runtime, might not be appropriate for all purposes [default=OFF].")
  set(USE_CERES_NNLS_DESCRIPTION       "Use the non-linear least squares optimizer from Ceres solver to tune model coefficients (if OFF, Eigen::LevenbergMarquardt will be used instead).")
  
  # option descriptions
  option(USE_OPENLIBM         ${OPENLIBM_DESCRIPTION}             OFF)
  option(USE_JEMALLOC         ${JEMALLOC_DESCRIPTION}             OFF)
  option(USE_TCMALLOC         ${TCMALLOC_DESCRIPTION}             OFF)
  option(USE_MIMALLOC         ${MIMALLOC_DESCRIPTION}             OFF)
  option(USE_SINGLE_PRECISION ${USE_SINGLE_PRECISION_DESCRIPTION} OFF)
  option(USE_CERES_NNLS       ${USE_CERES_NNLS_DESCRIPTION}       OFF)
  
  # provide a summary of configured options
  #include(FeatureSummary)
  #add_feature_info(BUILD_PYBIND         BUILD_PYBIND         ${BUILD_PYBIND_DESCRIPTION})
  #add_feature_info(USE_OPENLIBM         USE_OPENLIBM         ${OPENLIBM_DESCRIPTION})
  #add_feature_info(USE_JEMALLOC         USE_JEMALLOC         ${JEMALLOC_DESCRIPTION})
  #add_feature_info(USE_TCMALLOC         USE_TCMALLOC         ${TCMALLOC_DESCRIPTION})
  #add_feature_info(USE_MIMALLOC         USE_MIMALLOC         ${MIMALLOC_DESCRIPTION})
  #add_feature_info(USE_SINGLE_PRECISION USE_SINGLE_PRECISION ${USE_SINGLE_PRECISION_DESCRIPTION})
  #add_feature_info(CERES_TINY_SOLVER    CERES_TINY_SOLVER    ${CERES_TINY_SOLVER_DESCRIPTION})
  #add_feature_info(CERES_ALWAYS_DOUBLE  CERES_ALWAYS_DOUBLE  ${CERES_ALWAYS_DOUBLE_DESCRIPTION})
  set(CMAKE_EXPORT_COMPILE_COMMANDS ON CACHE INTERNAL "")
  if(CMAKE_EXPORT_COMPILE_COMMANDS)
    set(CMAKE_CXX_STANDARD_INCLUDE_DIRECTORIES ${CMAKE_CXX_IMPLICIT_INCLUDE_DIRECTORIES})
  endif()
endif()


# ---- Suppress C4251 on Windows ----

# Please see include/operon/operon.hpp for more details
set(pragma_suppress_c4251 "#define OPERON_SUPPRESS_C4251")
if(MSVC)
  string(APPEND pragma_suppress_c4251 [[ _Pragma("warning(suppress:4251)")]])
endif()

# ---- Warning guard ----

# target_include_directories with the SYSTEM modifier will request the compiler
# to omit warnings from the provided paths, if the compiler supports that
# This is to provide a user experience similar to find_package when
# add_subdirectory or FetchContent is used to consume this project
set(operon_warning_guard "")
if(NOT PROJECT_IS_TOP_LEVEL)
  option(
      operon_INCLUDES_WITH_SYSTEM
      "Use SYSTEM modifier for operon's includes, disabling warnings"
      ON
  )
  mark_as_advanced(operon_INCLUDES_WITH_SYSTEM)
  if(operon_INCLUDES_WITH_SYSTEM)
    set(operon_warning_guard SYSTEM)
  endif()
endif()
