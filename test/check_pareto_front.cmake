# Runs operon_nsgp with --pareto-front and validates the written file.
# Required -D variables: NSGP (executable), OUT (front file), WORKDIR, ARGS (';'-separated CLI arguments).
file(REMOVE "${OUT}")

execute_process(
    COMMAND "${NSGP}" ${ARGS} --pareto-front "${OUT}"
    WORKING_DIRECTORY "${WORKDIR}"
    RESULT_VARIABLE rc
    OUTPUT_QUIET
)
if(NOT rc EQUAL 0)
    message(FATAL_ERROR "operon_nsgp exited with '${rc}'")
endif()

if(NOT EXISTS "${OUT}")
    message(FATAL_ERROR "front file was not written: ${OUT}")
endif()

file(READ "${OUT}" json)
string(JSON type ERROR_VARIABLE err TYPE "${json}")
if(err OR NOT type STREQUAL "ARRAY")
    message(FATAL_ERROR "front file is not a JSON array (type='${type}', error='${err}')")
endif()
string(JSON n LENGTH "${json}")
if(n EQUAL 0)
    message(FATAL_ERROR "front file is an empty array")
endif()

math(EXPR last "${n} - 1")
foreach(i RANGE ${last})
    string(JSON expr ERROR_VARIABLE err GET "${json}" ${i} expression)
    if(err OR expr STREQUAL "")
        message(FATAL_ERROR "entry ${i}: missing or empty 'expression' (${err})")
    endif()
    string(JSON otype ERROR_VARIABLE err TYPE "${json}" ${i} objectives)
    if(err OR NOT otype STREQUAL "ARRAY")
        message(FATAL_ERROR "entry ${i}: 'objectives' is not an array (${err})")
    endif()
endforeach()
message(STATUS "front file OK: ${n} entries")
