vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO RedSpah/xxhash_cpp
    REF 2400ea5adc1156b586ee988ea7850be45d9011b5
    SHA512 e889e4df3c2416fdc1d7b44528349652538d5e1e8e581f476d931d5db76f8b6c67aedd81cd33f284c4f2e7f8ce305aec59755454cd0d5ce2432ff67f2f4ed2cb
    HEAD_REF master
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
)

vcpkg_cmake_install()
#vcpkg_cmake_config_fixup(PACKAGE_NAME xxhash_cpp CONFIG_PATH lib/cmake/xxhash_cpp DO_NOT_DELETE_PARENT_CONFIG_PATH)

file(REMOVE_RECURSE "${CURRENT_PACKAGES_DIR}/debug/include"
                    "${CURRENT_PACKAGES_DIR}/debug/share"
                    "${CURRENT_PACKAGES_DIR}/lib")

vcpkg_fixup_pkgconfig()

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)

