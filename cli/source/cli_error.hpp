// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_ERROR_HPP
#define OPERON_CLI_ERROR_HPP

#include <fmt/format.h>
#include <tl/expected.hpp>

#include <cstdlib>
#include <exception>
#include <string>
#include <utility>

namespace Operon::Cli {
enum class ErrorCode {
    InvalidArguments,
    Input,
    Configuration,
    Runtime,
};

struct Error {
    ErrorCode Code;
    std::string Context;
    std::string Message;
};

template<typename T>
using Result = tl::expected<T, Error>;

inline auto Report(Error const& error) -> int
{
    if (error.Context.empty()) {
        fmt::print(stderr, "error: {}\n", error.Message);
    } else {
        fmt::print(stderr, "error: {}: {}\n", error.Context, error.Message);
    }
    return EXIT_FAILURE;
}

// Transitional adapter for legacy APIs. New CLI-facing APIs return Result<T>
// directly; this boundary prevents a legacy exception from escaping main.
template<typename F>
auto Invoke(F&& fn, ErrorCode code, std::string context) -> Result<decltype(fn())>
{
    try {
        return std::forward<F>(fn)();
    } catch (std::exception const& error) {
        return tl::unexpected(Error{code, std::move(context), error.what()});
    }
}

} // namespace Operon::Cli

#endif
