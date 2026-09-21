# ---- Documentation configuration ----

macro(docs_early_return)
  return()
endmacro()

find_package(Python3 3.9 REQUIRED COMPONENTS Interpreter)
find_program(MKDOCS_EXECUTABLE NAMES mkdocs REQUIRED)
find_program(DOT_EXECUTABLE NAMES dot REQUIRED)

set(
    DOCS_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs"
    CACHE PATH "Path for generated documentation HTML and API material"
)
set(MKDOCS_SOURCE_DIRECTORY "${PROJECT_SOURCE_DIR}/docs/mkdocs")
set(MKDOCS_CONFIG "${MKDOCS_SOURCE_DIRECTORY}/mkdocs.yml")
set(MKDOCS_HTML_OUTPUT_DIRECTORY "${DOCS_OUTPUT_DIRECTORY}/html")

file(
    GLOB_RECURSE mkdocs_sources
    CONFIGURE_DEPENDS
    "${MKDOCS_SOURCE_DIRECTORY}/docs/*.md"
    "${MKDOCS_SOURCE_DIRECTORY}/docs/*.css"
)

set(MKDOCS_DIAGRAM_SOURCE_DIRECTORY "${MKDOCS_SOURCE_DIRECTORY}/diagrams")
set(MKDOCS_DIAGRAM_OUTPUT_DIRECTORY "${MKDOCS_SOURCE_DIRECTORY}/docs/assets/diagrams")

file(MAKE_DIRECTORY "${MKDOCS_DIAGRAM_OUTPUT_DIRECTORY}")
file(
    GLOB mkdocs_diagram_sources
    CONFIGURE_DEPENDS
    "${MKDOCS_DIAGRAM_SOURCE_DIRECTORY}/*.dot"
)

set(mkdocs_diagram_outputs)
foreach(diagram_source IN LISTS mkdocs_diagram_sources)
  get_filename_component(diagram_name "${diagram_source}" NAME_WE)
  set(diagram_output "${MKDOCS_DIAGRAM_OUTPUT_DIRECTORY}/${diagram_name}.svg")
  add_custom_command(
      OUTPUT "${diagram_output}"
      COMMAND "${DOT_EXECUTABLE}" -Tsvg -o "${diagram_output}" "${diagram_source}"
      DEPENDS "${diagram_source}"
      COMMENT "Rendering ${diagram_name} documentation diagram"
      VERBATIM
  )
  list(APPEND mkdocs_diagram_outputs "${diagram_output}")
endforeach()

set(mkdocs_html_index "${MKDOCS_HTML_OUTPUT_DIRECTORY}/index.html")
add_custom_command(
    OUTPUT "${mkdocs_html_index}"
    COMMAND "${CMAKE_COMMAND}" -E remove_directory "${MKDOCS_HTML_OUTPUT_DIRECTORY}"
    COMMAND "${MKDOCS_EXECUTABLE}" build
        --strict
        --config-file "${MKDOCS_CONFIG}"
        --site-dir "${MKDOCS_HTML_OUTPUT_DIRECTORY}"
    DEPENDS "${MKDOCS_CONFIG}" ${mkdocs_sources} ${mkdocs_diagram_outputs}
    COMMENT "Building MkDocs Material documentation"
    VERBATIM
)
add_custom_target(docs DEPENDS "${mkdocs_html_index}")

