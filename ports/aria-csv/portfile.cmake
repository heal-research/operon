vcpkg_from_github(
    OUT_SOURCE_PATH SOURCE_PATH
    REPO AriaFallah/csv-parser
    REF 544c764d0585c61d4c3bd3a023a825f3d7de1f31
    SHA512 2a16cb0d112f467337ebcd7a9911b50112632c7ff477b562bf9ecacff7a6879d70406de1cca7c097ce7379b61313f1d3062643164e057506a359f7591f2ed778
    HEAD_REF main
)

set(VCPKG_BUILD_TYPE release)

# we only need the hpp file
file(MAKE_DIRECTORY "${CURRENT_PACKAGES_DIR}/include/aria-csv")
file(COPY "${SOURCE_PATH}/parser.hpp"
    DESTINATION "${CURRENT_PACKAGES_DIR}/include/aria-csv")

file(
  INSTALL "${SOURCE_PATH}/LICENSE"
  DESTINATION "${CURRENT_PACKAGES_DIR}/share/${PORT}"
  RENAME copyright)

