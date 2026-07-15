// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "probes_config.hpp"

#include <fstream>
#include <sstream>
#include <stdexcept>

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

auto ToParamValue(Json const& v) -> ProbeParamValue
{
    if (v.is_boolean()) { return ProbeParamValue{v.get<bool>()}; }
    if (v.holds<std::int64_t>()) { return ProbeParamValue{v.get<std::int64_t>()}; }
    if (v.is_number()) { return ProbeParamValue{v.get<double>()}; }
    if (v.is_string()) { return ProbeParamValue{v.get<std::string>()}; }
    throw std::runtime_error("probes config: probe params must be a bool, number, or string");
}

auto ToParams(Json const& obj) -> ProbeParams
{
    ProbeParams params;
    for (auto const& [key, value] : obj.get_object()) {
        params.insert_or_assign(key, ToParamValue(value));
    }
    return params;
}

// Reads a non-negative integer field, defaulting to `fallback` if absent.
auto ToCount(Json const& entry, char const* field, std::size_t fallback) -> std::size_t
{
    if (!entry.contains(field)) { return fallback; }
    auto const& v = entry.at(field);
    if (!v.holds<std::int64_t>() || v.get<std::int64_t>() < 0) {
        throw std::runtime_error(fmt::format("probes config: '{}' must be a non-negative integer", field));
    }
    return static_cast<std::size_t>(v.get<std::int64_t>());
}

auto RequireString(Json const& obj, char const* field, char const* context) -> std::string
{
    if (!obj.contains(field) || !obj.at(field).is_string()) {
        throw std::runtime_error(fmt::format("probes config: {} requires a string '{}'", context, field));
    }
    return obj.at(field).get<std::string>();
}

} // namespace

auto LoadProbeConfig(std::string const& path) -> std::optional<ProbeChain>
{
    if (path.empty()) { return std::nullopt; }

    std::ifstream in(path);
    if (!in) {
        throw std::runtime_error(fmt::format("probes config: could not open '{}'", path));
    }
    std::stringstream buf;
    buf << in.rdbuf();
    auto const text = buf.str();

    Json doc;
    if (auto ec = glz::read_json(doc, text); ec) {
        throw std::runtime_error(fmt::format("probes config: {}", glz::format_error(ec, text)));
    }

    ProbeRegistry registry;
    RegisterBuiltinProbes(registry);

    ProbeChain chain;

    if (doc.contains("probes")) {
        for (auto const& entry : doc.at("probes").get_array()) {
            auto const type = RequireString(entry, "type", "each probe entry");
            auto const every = ToCount(entry, "every", 1);
            auto const offset = ToCount(entry, "offset", 0);
            auto const params = entry.contains("params") ? ToParams(entry.at("params")) : ProbeParams{};

            auto probe = registry.Create(type, params);
            if (!probe) {
                throw std::runtime_error(fmt::format("probes config: unknown probe type '{}'", type));
            }
            chain.Add(std::move(probe), every, offset);
        }
    }

    if (doc.contains("sink")) {
        auto const& sink = doc.at("sink");
        auto const sinkType = RequireString(sink, "type", "'sink'");
        auto const sinkPath = RequireString(sink, "path", "'sink'");
        if (sinkType != "jsonl") {
            throw std::runtime_error(fmt::format("probes config: unknown sink type '{}' (only 'jsonl' is supported)", sinkType));
        }
        chain.SetSink(std::make_unique<JsonlSink>(sinkPath));
    }

    return chain;
}

} // namespace Operon
