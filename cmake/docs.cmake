# ---- Redefine docs_early_return ----

# This function must be a macro, so the return() takes effect in the calling
# scope. This prevents other targets from being available and potentially
# requiring dependencies. This cuts down on the time it takes to generate
# documentation in CI.
macro(docs_early_return)
  return()
endmacro()

# ---- Dependencies ----

include(FetchContent)
FetchContent_Declare(
    mcss
    GIT_REPOSITORY https://github.com/mosra/m.css.git
    GIT_TAG 0a460a7a9973a41db48f735e7b49e4da9a876325
    SOURCE_DIR "${PROJECT_BINARY_DIR}/mcss"
    UPDATE_DISCONNECTED YES
)
FetchContent_MakeAvailable(mcss)

find_package(Python3 3.9 REQUIRED)
find_program(DOXYGEN_EXECUTABLE NAMES doxygen REQUIRED)

# ---- Declare documentation target ----

set(
    DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs"
    CACHE PATH "Path for the generated Doxygen documentation"
)

set(working_dir "${PROJECT_BINARY_DIR}/docs")

foreach(file IN ITEMS Doxyfile conf.py)
  configure_file("docs/${file}.in" "${working_dir}/${file}" @ONLY)
endforeach()

set(mcss_script "${mcss_SOURCE_DIR}/documentation/doxygen.py")
set(config "${working_dir}/conf.py")

add_custom_target(
    docs
    COMMAND "${CMAKE_COMMAND}" -E remove_directory
    "${DOXYGEN_OUTPUT_DIRECTORY}/html"
    "${DOXYGEN_OUTPUT_DIRECTORY}/xml"
    # Doxygen 1.16 emits Doxyfile.xml alongside the API XML. m.css treats every
    # XML file as a documented compound, but the configuration document has no
    # compound definition.
    COMMAND "${DOXYGEN_EXECUTABLE}" "${working_dir}/Doxyfile"
    COMMAND "${CMAKE_COMMAND}" -E rm -f "${DOXYGEN_OUTPUT_DIRECTORY}/xml/Doxyfile.xml"
    COMMAND "${Python3_EXECUTABLE}" "${mcss_script}" --no-doxygen "${config}"
    COMMENT "Building documentation using Doxygen and m.css"
    WORKING_DIRECTORY "${working_dir}"
    VERBATIM
)
