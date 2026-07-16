// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_REGISTRY_HPP
#define OPERON_ALGORITHMS_PROBES_REGISTRY_HPP

#include <functional>
#include <memory>
#include <string>
#include <type_traits>
#include <variant>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// A single probe config parameter, already parsed down to a plain scalar -
// not glz::json_t: glaze stays a PRIVATE dependency confined to source/*.cpp
// (see core/serialization.cpp), and ProbeParams lives in a public header, so
// the JSON-vs-YAML config format question stays on the parsing side
// (cli/source/probes_config.cpp).
//
// Wraps std::variant instead of aliasing it: a bare
// variant<int64_t, double, bool, string> silently binds a `const char*` to
// `bool` rather than `string` (pointer-to-bool beats the user-defined string
// conversion). The deleted char const* constructor below is an exact match
// for that case, turning the mistake into a compile error - construct
// std::string values explicitly (`ProbeParamValue{std::string{...}}`).
class ProbeParamValue {
public:
    ProbeParamValue(std::int64_t v) : value_(v) { } // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
    ProbeParamValue(double v) : value_(v) { } // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
    ProbeParamValue(bool v) : value_(v) { } // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
    ProbeParamValue(std::string v) : value_(std::move(v)) { } // NOLINT(google-explicit-constructor,hicpp-explicit-conversions)
    ProbeParamValue(char const*) = delete; // use std::string(...) explicitly - see class comment

    template <typename T>
    [[nodiscard]] auto Holds() const -> bool { return std::holds_alternative<T>(value_); }

    template <typename T>
    [[nodiscard]] auto Get() const -> T const& { return std::get<T>(value_); }

private:
    std::variant<std::int64_t, double, bool, std::string> value_;
};

// Keeps the trap closed even if this class changes later.
static_assert(!std::is_constructible_v<ProbeParamValue, char const*>,
              "ProbeParamValue must reject char const* to avoid silently binding to bool - construct std::string explicitly");

using ProbeParams = Operon::Map<std::string, ProbeParamValue>;

using ProbeFactory = std::function<std::unique_ptr<GenerationProbe>(ProbeParams const&)>;

// Name -> factory registry, so adding a new probe type never requires
// touching config-parsing code. Plain value type, not a singleton: callers
// construct one, populate it, and use it to build a ProbeChain from a
// parsed config.
class OPERON_EXPORT ProbeRegistry {
public:
    auto Register(std::string type, ProbeFactory factory) -> void;

    [[nodiscard]] auto Contains(std::string const& type) const -> bool;

    // Returns nullptr if `type` was never registered - the caller reports
    // the unknown type itself (it has the config-file context to do so).
    [[nodiscard]] auto Create(std::string const& type, ProbeParams const& params) const -> std::unique_ptr<GenerationProbe>;

private:
    Operon::Map<std::string, ProbeFactory> factories_;
};

// Registers the three built-in probes ("population_trace", "cache_hit_rate",
// "structural_diversity") by name. A plain function, not static/global
// registration, to avoid static-init-order surprises - call it explicitly
// on a registry before parsing config that might reference these types.
OPERON_EXPORT auto RegisterBuiltinProbes(ProbeRegistry& registry) -> void;

} // namespace Operon

#endif
