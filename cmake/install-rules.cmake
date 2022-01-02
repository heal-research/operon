if(PROJECT_IS_TOP_LEVEL)
  set(CMAKE_INSTALL_INCLUDEDIR include/operon CACHE PATH "")
endif()

include(CMakePackageConfigHelpers)
include(GNUInstallDirs)

# find_package(<package>) call for consumers to find this project
set(package operon)

install(
    DIRECTORY
    include/
    "${PROJECT_BINARY_DIR}/export/"
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
    COMPONENT operon_Development
)

install(
    TARGETS operon_operon
    EXPORT operonTargets
    RUNTIME #
    COMPONENT operon_Runtime
    LIBRARY #
    COMPONENT operon_Runtime
    NAMELINK_COMPONENT operon_Development
    ARCHIVE #
    COMPONENT operon_Development
    INCLUDES #
    DESTINATION "${CMAKE_INSTALL_INCLUDEDIR}"
)

write_basic_package_version_file(
    "${package}ConfigVersion.cmake"
    COMPATIBILITY SameMajorVersion
)

# Allow package maintainers to freely override the path for the configs
set(
    operon_INSTALL_CMAKEDIR "${CMAKE_INSTALL_DATADIR}/${package}"
    CACHE PATH "CMake package config location relative to the install prefix"
)
mark_as_advanced(operon_INSTALL_CMAKEDIR)

install(
    FILES cmake/install-config.cmake
    DESTINATION "${operon_INSTALL_CMAKEDIR}"
    RENAME "${package}Config.cmake"
    COMPONENT operon_Development
)

install(
    FILES "${PROJECT_BINARY_DIR}/${package}ConfigVersion.cmake"
    DESTINATION "${operon_INSTALL_CMAKEDIR}"
    COMPONENT operon_Development
)

install(
    EXPORT operonTargets
    NAMESPACE operon::
    DESTINATION "${operon_INSTALL_CMAKEDIR}"
    COMPONENT operon_Development
)

if(PROJECT_IS_TOP_LEVEL)
  include(CPack)
endif()
