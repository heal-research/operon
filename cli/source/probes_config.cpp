// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "probes_config.hpp"

#include <cstdint>
#include <exception>
#include <fstream>
#include <limits>
#include <sstream>
#include <utility>

#include <fmt/format.h>
#include <glaze/glaze.hpp>

#include "operon/algorithms/probes/registry.hpp"

namespace Operon {

namespace {

// Integers parse as int64_t, everything else (incl. non-integral numbers)
// as double - matches ProbeParamValue's own int64_t/double split, unlike
// glaze's default num_mode::f64 (all numbers as double) which would lose
// that distinction for every integer field in the config.
using Json = glz::generic_i64;

auto ConfigurationError(std::string message) -> Cli::Error
{
    return {Cli::ErrorCode::Configuration, "probes config", std::move(message)};
}

auto InputError(std::string message) -> Cli::Error
{
    return {Cli::ErrorCode::Input, "probes config", std::move(message)};
}

auto ToParamValue(Json const& v, std::string const& key) -> Cli::Result<ProbeParamValue>
{
    if (v.is_boolean()) { return ProbeParamValue{v.get<bool>()}; }
    if (v.holds<std::int64_t>()) { return ProbeParamValue{v.get<std::int64_t>()}; }
    if (v.is_number()) { return ProbeParamValue{v.get<double>()}; }
    if (v.is_string()) { return ProbeParamValue{v.get<std::string>()}; }
    return tl::unexpected(ConfigurationError(fmt::format("param '{}' must be a bool, number, or string", key)));
}

auto ToParams(Json const& obj) -> Cli::Result<ProbeParams>
{
    if (!obj.is_object()) {
        return tl::unexpected(ConfigurationError("'params' must be an object"));
    }

    ProbeParams params;
    for (auto const& [key, value] : obj.get_object()) {
        auto param = ToParamValue(value, key);
        if (!param) { return tl::unexpected(std::move(param.error())); }
        params.insert_or_assign(key, std::move(*param));
    }
    return params;
}

// Reads a non-negative integer field, defaulting to `fallback` if absent.
auto ToCount(Json const& entry, char const* field, std::size_t fallback) -> Cli::Result<std::size_t>
{
    if (!entry.contains(field)) { return fallback; }
    auto const& v = entry.at(field);
    if (!v.holds<std::int64_t>() || v.get<std::int64_t>() < 0) {
        return tl::unexpected(ConfigurationError(fmt::format("'{}' must be a non-negative integer", field)));
    }
    auto const value = v.get<std::int64_t>();
    if (static_cast<std::uint64_t>(value) > std::numeric_limits<std::size_t>::max()) {
        return tl::unexpected(ConfigurationError(fmt::format("'{}' is too large", field)));
    }
    return static_cast<std::size_t>(value);
}

auto RequireString(Json const& obj, char const* field, char const* context) -> Cli::Result<std::string>
{
    if (!obj.contains(field) || !obj.at(field).is_string()) {
        return tl::unexpected(ConfigurationError(fmt::format("{} requires a string '{}'", context, field)));
    }
    return obj.at(field).get<std::string>();
}

} // namespace

auto LoadProbeConfig(std::string const& path) -> Cli::Result<std::optional<ProbeChain>>
{
    if (path.empty()) { return std::optional<ProbeChain>{}; }

    std::ifstream in(path);
    if (!in) {
        return tl::unexpected(InputError(fmt::format("could not open '{}'", path)));
    }
    std::stringstream buf;
    buf << in.rdbuf();
    auto const text = buf.str();
    if (in.bad()) {
        return tl::unexpected(InputError(fmt::format("could not read '{}'", path)));
    }

    Json doc;
    if (auto ec = glz::read_json(doc, text); ec) {
        return tl::unexpected(ConfigurationError(fmt::format("'{}': {}", path, glz::format_error(ec, text))));
    }
    if (!doc.is_object()) {
        return tl::unexpected(ConfigurationError("top-level JSON value must be an object"));
    }

    ProbeRegistry registry;
    RegisterBuiltinProbes(registry);

    ProbeChain chain;

    if (doc.contains("probes")) {
        auto const& entries = doc.at("probes");
        if (!entries.is_array()) {
            return tl::unexpected(ConfigurationError("'probes' must be an array"));
        }
        for (auto const& entry : entries.get_array()) {
            if (!entry.is_object()) {
                return tl::unexpected(ConfigurationError("each probe entry must be an object"));
            }
            auto type = RequireString(entry, "type", "each probe entry");
            if (!type) { return tl::unexpected(std::move(type.error())); }
            auto every = ToCount(entry, "every", 1);
            if (!every) { return tl::unexpected(std::move(every.error())); }
            auto offset = ToCount(entry, "offset", 0);
            if (!offset) { return tl::unexpected(std::move(offset.error())); }

            ProbeParams params;
            if (entry.contains("params")) {
                auto parsed = ToParams(entry.at("params"));
                if (!parsed) { return tl::unexpected(std::move(parsed.error())); }
                params = std::move(*parsed);
            }

            try {
                auto probe = registry.Create(*type, params);
                if (!probe) {
                    return tl::unexpected(ConfigurationError(fmt::format("unknown probe type '{}'", *type)));
                }
                chain.Add(std::move(probe), *every, *offset);
            } catch (std::exception const& error) {
                return tl::unexpected(ConfigurationError(error.what()));
            }
        }
    }

    if (doc.contains("sink")) {
        auto const& sink = doc.at("sink");
        if (!sink.is_object()) {
            return tl::unexpected(ConfigurationError("'sink' must be an object"));
        }
        auto sinkType = RequireString(sink, "type", "'sink'");
        if (!sinkType) { return tl::unexpected(std::move(sinkType.error())); }
        auto sinkPath = RequireString(sink, "path", "'sink'");
        if (!sinkPath) { return tl::unexpected(std::move(sinkPath.error())); }
        if (*sinkType != "jsonl") {
            return tl::unexpected(ConfigurationError(fmt::format(
                "unknown sink type '{}' (only 'jsonl' is supported)", *sinkType)));
        }
        chain.SetSink(std::make_unique<JsonlSink>(*sinkPath));
    }

    return std::optional<ProbeChain>{std::move(chain)};
}

} // namespace Operon
