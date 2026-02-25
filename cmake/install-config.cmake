include(CMakeFindDependencyMacro)

# PUBLIC dependencies
find_dependency(Eigen3 CONFIG REQUIRED)
find_dependency(Threads REQUIRED)
find_dependency(fluky CONFIG REQUIRED)
find_dependency(fmt CONFIG REQUIRED)
find_dependency(lbfgs CONFIG REQUIRED)
find_dependency(libassert CONFIG REQUIRED)
find_dependency(pratt-parser CONFIG REQUIRED)
find_dependency(mdspan CONFIG REQUIRED)
find_dependency(vstat CONFIG REQUIRED)

# INTERFACE dependencies
find_dependency(eve CONFIG REQUIRED)

# PRIVATE dependencies (needed for static library)
find_dependency(AriaCsvParser CONFIG REQUIRED)
find_dependency(Taskflow CONFIG REQUIRED)
find_dependency(cpp-sort CONFIG REQUIRED)
find_dependency(unordered_dense CONFIG REQUIRED)
find_dependency(tl-expected CONFIG REQUIRED)

include("${CMAKE_CURRENT_LIST_DIR}/operonTargets.cmake")
check_required_components(operon)
