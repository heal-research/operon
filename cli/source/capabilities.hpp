// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_CAPABILITIES_HPP
#define OPERON_CLI_CAPABILITIES_HPP

#include <algorithm>
#include <cstdio>
#include <fmt/format.h>
#include <string>
#include <string_view>
#include <vector>

#include "operon/core/version.hpp"

namespace Operon::Cli {

// Escapes `text` for use inside a JSON string literal.
inline auto JsonEscape(std::string_view text) -> std::string
{
    std::string out;
    out.reserve(text.size());
    for (char const c : text) {
        switch (c) {
        case '"':
            out += "\\\"";
            break;
        case '\\':
            out += "\\\\";
            break;
        case '\n':
            out += "\\n";
            break;
        case '\r':
            out += "\\r";
            break;
        case '\t':
            out += "\\t";
            break;
        default:
            if (static_cast<unsigned char>(c) < 0x20) {
                out += fmt::format("\\u{:04x}", static_cast<unsigned>(static_cast<unsigned char>(c)));
            } else {
                out += c;
            }
        }
    }
    return out;
}

inline auto JsonStringArray(std::vector<std::string> const& values) -> std::string
{
    std::string out = "[";
    for (std::size_t i = 0; i < values.size(); ++i) {
        out += fmt::format("{}\"{}\"", i == 0 ? "" : ",", JsonEscape(values[i]));
    }
    return out + "]";
}

// One-line JSON document describing the binary: identity (revision, build type, platform), numeric precision,
// compile-time ISA, available optional features and the options this program registers. `options` are the long option
// names registered with the parser (without the leading dashes); a registered option can still be rejected at run
// time (for example options that another program of the family reads), so this is a registration list, not a
// promise that every option has an effect. `features` describe the library build, not the individual program.
inline auto CapabilitiesJson(std::string_view program, std::vector<std::string> options, BuildInfo const& info)
    -> std::string
{
    std::ranges::sort(options);
    auto const hasOption = [&](std::string_view name) { return std::ranges::binary_search(options, name); };
    // The machine report (--report-json) is the only versioned output the CLI produces today.
    std::string const machineReportVersions = hasOption("report-json") ? "[1]" : "[]";
    auto const levelJson = info.IsaLevel.empty() ? std::string { "null" } : fmt::format("\"{}\"", info.IsaLevel);

    return fmt::format(
        "{{\"schema\":\"operon.capabilities\",\"schema_version\":1,\"program\":\"{program}\","
        "\"revision\":\"{revision}\",\"build_type\":\"{buildType}\",\"platform\":\"{platform}\",\"arch\":\"{arch}\","
        "\"precision\":\"{precision}\","
        "\"compiler\":{{\"id\":\"{compilerId}\",\"version\":\"{compilerVersion}\",\"flags\":\"{flags}\"}},"
        "\"dependencies\":{{\"eigen\":\"{eigen}\",\"taskflow\":\"{taskflow}\"}},"
        "\"isa\":{{\"level\":{level},\"features\":{isaFeatures}}},"
        "\"features\":{{\"jit\":{jit},\"shape_constraints\":true,\"probes\":true,\"checkpoint\":true,"
        "\"transposition_cache\":true}},"
        "\"machine_report_schema_versions\":{machineReport},"
        "\"options\":{options}}}",
        fmt::arg("program", JsonEscape(program)), fmt::arg("revision", JsonEscape(info.Revision)),
        fmt::arg("buildType", JsonEscape(info.BuildType)), fmt::arg("platform", JsonEscape(info.Platform)),
        fmt::arg("arch", JsonEscape(info.Arch)), fmt::arg("precision", JsonEscape(info.Precision)),
        fmt::arg("compilerId", JsonEscape(info.CompilerId)),
        fmt::arg("compilerVersion", JsonEscape(info.CompilerVersion)),
        fmt::arg("flags", JsonEscape(info.CompilerFlags)), fmt::arg("eigen", JsonEscape(info.EigenVersion)),
        fmt::arg("taskflow", JsonEscape(info.TaskflowVersion)), fmt::arg("level", levelJson),
        fmt::arg("isaFeatures", JsonStringArray(info.IsaFeatures)), fmt::arg("jit", info.Jit ? "true" : "false"),
        fmt::arg("machineReport", machineReportVersions), fmt::arg("options", JsonStringArray(options)));
}

} // namespace Operon::Cli

#endif
