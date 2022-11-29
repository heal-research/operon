vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO AriaFallah/csv-parser
    REF 4965c9f320d157c15bc1f5a6243de116a4caf101
    SHA512 57c50b592a992752e49cfe94f0534b4795a91a3db625b303b56b84f9ad27e0fcf71ff3e5bdb8bd4533e79db49a39075a28e6eb359484ccb145d870209b8bca46
    HEAD_REF master
)

include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake/vcpkg_cmake_build.cmake")
include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake/vcpkg_cmake_install.cmake")
include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake-config/vcpkg_cmake_config_fixup.cmake")

set(VCPKG_BUILD_TYPE release)

vcpkg_configure_cmake(
  SOURCE_PATH "${SOURCE_PATH}"
  PREFER_NINJA
)

vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME AriaCsvParser CONFIG_PATH lib/cmake/AriaCsvParser DO_NOT_DELETE_PARENT_CONFIG_PATH)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share"
                    "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_fixup_pkgconfig()

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)

