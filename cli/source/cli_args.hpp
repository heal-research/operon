// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_ARGS_HPP
#define OPERON_CLI_ARGS_HPP

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <fmt/format.h>
#include <limits>
#include <optional>
#include <scn/scan.h>
#include <string_view>
#include <tl/expected.hpp>
#include <type_traits>

#include "cli_error.hpp"

namespace Operon::Cli {

// --generations / --evaluations values at or above this are "no limit" (also what the literal `unlimited` means).
inline constexpr std::uint64_t UnlimitedThreshold = std::uint64_t { 1 } << 62;

// Scans `token` as exactly one T: empty input, trailing characters ("5x"), negative values for unsigned types and
// non-finite floating-point values are all rejected.
template <typename T> [[nodiscard]] auto ScanExact(std::string_view token) -> std::optional<T>
{
    auto const result = scn::scan<T>(token, "{}");
    if (!result || !result->range().empty()) {
        return std::nullopt;
    }
    auto const value = result->value();
    if constexpr (std::is_floating_point_v<T>) {
        if (!std::isfinite(value)) {
            return std::nullopt;
        }
    }
    return value;
}

// Parses a budget option: a non-negative integer or the literal `unlimited`. Values >= 2^62 are unlimited and are
// normalised to SIZE_MAX, the value the engine treats as "never reached".
[[nodiscard]] inline auto ParseLimit(std::string_view option, std::string_view text) -> Result<std::size_t>
{
    if (text == "unlimited") {
        return std::numeric_limits<std::size_t>::max();
    }
    auto const value = ScanExact<std::uint64_t>(text);
    if (!value) {
        return tl::unexpected(Error { ErrorCode::InvalidArguments, fmt::format("--{}", option),
            fmt::format("'{}' is not a non-negative integer or 'unlimited'", text) });
    }
    if (*value >= UnlimitedThreshold) {
        return std::numeric_limits<std::size_t>::max();
    }
    return static_cast<std::size_t>(*value);
}

// An explicit worker count must be positive; omitting --threads selects the hardware concurrency.
[[nodiscard]] inline auto ParseThreads(std::size_t requested) -> Result<std::size_t>
{
    if (requested == 0) {
        return tl::unexpected(Error { ErrorCode::InvalidArguments, "--threads",
            "must be at least 1 (omit the option to use all hardware threads)" });
    }
    return requested;
}

} // namespace Operon::Cli

#endif
