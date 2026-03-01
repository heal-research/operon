vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO foolnotion/lbfgs
    REF b2e339b7d169ec3905f0f724958ed060acf592c9
    SHA512 a26355d918631f94935a4ee0968f975341b05842f0222e85963f1e3ac5ea75fa8c87975a33c8779d1408600c62b287ada60424814aaede995c26f1daa6752ab6
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
vcpkg_cmake_config_fixup(PACKAGE_NAME lbfgs CONFIG_PATH lib/cmake/lbfgs DO_NOT_DELETE_PARENT_CONFIG_PATH)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share"
                    "${CURRENT_PACKAGES_DIR}/lib")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)
