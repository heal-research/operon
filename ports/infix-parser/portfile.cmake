vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/infix-parser
    REF 18ea752693bc489000b53b7b8387322f91193c48
    SHA512 b6d8bd8bba5080c6d99be76b9b9270dc6bcb21b060cf48821737001fbbe8bef1cd5ba7175996a1daf453927e1e264c9dce5cd66b9e6a59954bd9ce65a663bb34
    HEAD_REF main
)

include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake/vcpkg_cmake_build.cmake")
include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake/vcpkg_cmake_install.cmake")
include("${VCPKG_ROOT_DIR}/ports/vcpkg-cmake-config/vcpkg_cmake_config_fixup.cmake")

set(VCPKG_BUILD_TYPE release)

vcpkg_configure_cmake(
  SOURCE_PATH "${SOURCE_PATH}"
  PREFER_NINJA
  OPTIONS
        -DBUILD_TESTING=OFF
        -DBUILD_EXAMPLES=OFF
)
vcpkg_cmake_install()
vcpkg_cmake_config_fixup(PACKAGE_NAME infix-parser CONFIG_PATH lib/cmake/infix-parser DO_NOT_DELETE_PARENT_CONFIG_PATH)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share"
                    "${CURRENT_PACKAGES_DIR}/lib")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
