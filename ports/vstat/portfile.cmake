vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO heal-research/vstat
    REF 79b9ba2d69fe14e9e16a10f35d4335ffa984f02d
    SHA512 a94973c695ece17f590b71f9c00183730030bdc6862939b893b696acc937e40216991bf33886ce2ebb5f5f540baeeb15315be2a1e5bb04de71512363a7556d12
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
vcpkg_cmake_config_fixup(PACKAGE_NAME vstat CONFIG_PATH lib/cmake/vstat DO_NOT_DELETE_PARENT_CONFIG_PATH)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share"
                    "${CURRENT_PACKAGES_DIR}/lib")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
