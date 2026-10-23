# SPDX-License-Identifier: MIT
# SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#
# Runs a CLI program and checks its exit status and output.
#   -DCLI=<executable> -DARGS=<arg1|arg2|...> -DEXPECT=fail|ok -DREGEX=<regular expression>
# EXPECT=fail: exit status must be exactly 1 (a crash or abort is not accepted) and stderr must match REGEX.
# EXPECT=ok:   exit status must be 0 and stdout must match REGEX.
string(REPLACE "|" ";" args "${ARGS}")
execute_process(
    COMMAND "${CLI}" ${args}
    RESULT_VARIABLE status
    OUTPUT_VARIABLE out
    ERROR_VARIABLE err
)

if(EXPECT STREQUAL "fail")
    if(NOT status STREQUAL "1")
        message(FATAL_ERROR "expected exit status 1, got '${status}'\nstdout: ${out}\nstderr: ${err}")
    endif()
    set(text "${err}")
elseif(EXPECT STREQUAL "ok")
    if(NOT status STREQUAL "0")
        message(FATAL_ERROR "expected exit status 0, got '${status}'\nstdout: ${out}\nstderr: ${err}")
    endif()
    set(text "${out}")
else()
    message(FATAL_ERROR "EXPECT must be 'fail' or 'ok'")
endif()

if(NOT text MATCHES "${REGEX}")
    message(FATAL_ERROR "output does not match '${REGEX}'\nstdout: ${out}\nstderr: ${err}")
endif()
