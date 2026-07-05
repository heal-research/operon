set(CMAKE_C_COMPILER "C:/Program Files/LLVM/bin/clang-cl.exe" CACHE STRING "" FORCE)
set(CMAKE_CXX_COMPILER "C:/Program Files/LLVM/bin/clang-cl.exe" CACHE STRING "" FORCE)

# Force static CRT (/MT) to match VCPKG_CRT_LINKAGE=static in the triplet.
# This is required because vcpkg's manifest-mode install during the pip wheel
# build may not propagate VCPKG_CRT_LINKAGE correctly through vcpkg_configure_cmake.
set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded" CACHE STRING "" FORCE)

# Some dependency ports (e.g. ndsort) declare cmake_minimum_required() below
# 3.15, so CMP0091 defaults to OLD and CMAKE_MSVC_RUNTIME_LIBRARY above is
# silently ignored, falling back to CMake's classic dynamic-CRT (/MD) default.
# Force the policy so every port actually honors the static-CRT setting,
# instead of patching cmake_minimum_required in each dependency's own repo.
set(CMAKE_POLICY_DEFAULT_CMP0091 "NEW" CACHE STRING "" FORCE)

# cmake auto-detects llvm-mt.exe from the LLVM directory and uses it for manifest
# embedding via vs_link_exe. llvm-mt does not generate the manifest.rc file cmake
# expects, causing rc.exe to fail with "no such file or directory". Since all
# targets use static CRT (/MT), manifests are unnecessary — disable them.
set(CMAKE_EXE_LINKER_FLAGS_INIT "/MANIFEST:NO")
set(CMAKE_SHARED_LINKER_FLAGS_INIT "/MANIFEST:NO")
set(CMAKE_MODULE_LINKER_FLAGS_INIT "/MANIFEST:NO")
