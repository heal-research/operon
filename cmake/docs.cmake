# ---- Redefine docs_early_return ----

# This function must be a macro, so the return() takes effect in the calling
# scope. This prevents other targets from being available and potentially
# requiring dependencies. This cuts down on the time it takes to generate
# documentation in CI.
macro(docs_early_return)
  return()
endmacro()

# ---- Dependencies ----

find_package(Doxygen REQUIRED)
find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)

# ---- Documentation inputs and outputs ----

set(
    DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs"
    CACHE PATH "Path for generated documentation intermediates and HTML"
)
set(DOXYGEN_XML_OUTPUT_DIRECTORY "${DOXYGEN_OUTPUT_DIRECTORY}/xml")
set(SPHINX_SOURCE_DIRECTORY "${PROJECT_SOURCE_DIR}/docs/sphinx")
set(SPHINX_HTML_OUTPUT_DIRECTORY "${DOXYGEN_OUTPUT_DIRECTORY}/html")
set(DOXYGEN_CONFIG "${DOXYGEN_OUTPUT_DIRECTORY}/Doxyfile")

configure_file(
    "${PROJECT_SOURCE_DIR}/docs/Doxyfile.in"
    "${DOXYGEN_CONFIG}"
    @ONLY
)

file(
    GLOB_RECURSE doxygen_inputs
    CONFIGURE_DEPENDS
    "${PROJECT_SOURCE_DIR}/include/operon/*.h"
    "${PROJECT_SOURCE_DIR}/include/operon/*.hpp"
)
file(
    GLOB_RECURSE sphinx_sources
    CONFIGURE_DEPENDS
    "${SPHINX_SOURCE_DIRECTORY}/*"
)

set(doxygen_xml_index "${DOXYGEN_XML_OUTPUT_DIRECTORY}/index.xml")
add_custom_command(
    OUTPUT "${doxygen_xml_index}"
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "${DOXYGEN_XML_OUTPUT_DIRECTORY}"
    COMMAND "${DOXYGEN_EXECUTABLE}" "${DOXYGEN_CONFIG}"
    DEPENDS "${DOXYGEN_CONFIG}" ${doxygen_inputs}
    COMMENT "Generating Doxygen XML"
    VERBATIM
)
add_custom_target(doxygen-xml DEPENDS "${doxygen_xml_index}")

set(sphinx_html_index "${SPHINX_HTML_OUTPUT_DIRECTORY}/index.html")
add_custom_command(
    OUTPUT "${sphinx_html_index}"
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "${DOXYGEN_OUTPUT_DIRECTORY}/doctrees"
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "${SPHINX_HTML_OUTPUT_DIRECTORY}"
    COMMAND "${CMAKE_COMMAND}" -E env
        "OPERON_DOXYGEN_XML_DIR=${DOXYGEN_XML_OUTPUT_DIRECTORY}"
        "${Python3_EXECUTABLE}" -m sphinx -b html
        -d "${DOXYGEN_OUTPUT_DIRECTORY}/doctrees"
        "${SPHINX_SOURCE_DIRECTORY}"
        "${SPHINX_HTML_OUTPUT_DIRECTORY}"
    DEPENDS "${doxygen_xml_index}" ${sphinx_sources}
    COMMENT "Building Sphinx HTML documentation"
    VERBATIM
)

add_custom_target(docs DEPENDS "${sphinx_html_index}")
