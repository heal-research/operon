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
    doxygen_awesome
    GIT_REPOSITORY https://github.com/jothepro/doxygen-awesome-css.git
    GIT_TAG 46483f1e5a70ffb9ecd3b82d0a1cd1b24edf13da
    SOURCE_DIR "${PROJECT_BINARY_DIR}/doxygen-awesome"
    UPDATE_DISCONNECTED YES
)
FetchContent_MakeAvailable(doxygen_awesome)

find_program(DOXYGEN_EXECUTABLE NAMES doxygen REQUIRED)

# ---- Declare documentation target ----
set(
    DOXYGEN_OUTPUT_DIRECTORY "${PROJECT_BINARY_DIR}/docs"
    CACHE PATH "Path for the generated Doxygen documentation"
)

set(working_dir "${PROJECT_BINARY_DIR}/docs")
file(MAKE_DIRECTORY "${working_dir}")

# Generate the header with the installed Doxygen version, then add the
# Doxygen Awesome dark-mode toggle without carrying a versioned Doxygen
# template in the source tree.
set(doxygen_awesome_header "${working_dir}/header.html")
execute_process(
    COMMAND "${DOXYGEN_EXECUTABLE}" -w html
            "${doxygen_awesome_header}"
            "${working_dir}/footer.html"
            "${working_dir}/doxygen.css"
    WORKING_DIRECTORY "${working_dir}"
    COMMAND_ERROR_IS_FATAL ANY
)
file(READ "${doxygen_awesome_header}" doxygen_header)
set(doxygen_awesome_header_script [=[
<script type="text/javascript" src="$relpath^doxygen-awesome-darkmode-toggle.js"></script>
<script type="text/javascript">
  DoxygenAwesomeDarkModeToggle.init()
</script>
]=])
string(REPLACE "</head>" "${doxygen_awesome_header_script}</head>" doxygen_header "${doxygen_header}")
file(WRITE "${doxygen_awesome_header}" "${doxygen_header}")

configure_file("docs/Doxyfile.in" "${working_dir}/Doxyfile" @ONLY)

add_custom_target(
    docs
    COMMAND "${CMAKE_COMMAND}" -E remove_directory
    "${DOXYGEN_OUTPUT_DIRECTORY}/html"
    COMMAND "${DOXYGEN_EXECUTABLE}" "${working_dir}/Doxyfile"
    COMMENT "Building documentation using Doxygen Awesome"
    WORKING_DIRECTORY "${working_dir}"
    VERBATIM
)
