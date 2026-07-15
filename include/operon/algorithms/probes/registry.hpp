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
// deliberately not glz::json_t (or any glaze type): glaze is a PRIVATE
// dependency confined to source/*.cpp translation units throughout this
// codebase (see core/serialization.cpp's header comment), and ProbeParams
// lives in a public header, so the JSON-vs-YAML-vs-whatever config format
// question stays entirely on the parsing side (cli/source/probes_config.cpp)
// - this registry only ever sees the result.
//
// A thin wrapper around std::variant, not a bare alias: a bare
// std::variant<std::int64_t, double, bool, std::string> has a well-known
// overload-resolution trap where initializing it from a `const char*`
// (e.g. a string literal at a config-parsing call site) silently selects
// the `bool` alternative instead of `std::string` - pointer-to-bool is a
// built-in standard conversion, preferred over the user-defined conversion
// to std::string. The deleted `char const*` constructor below is an exact
// match for that case, so it wins the overload resolution instead and
// turns the mistake into a compile error - construct std::string values
// explicitly (`ProbeParamValue{std::string{...}}`) at the parse boundary.
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

// Keeps the trap closed even if this class is edited later (e.g. someone
// adds a std::string_view constructor that reopens a similar ambiguity).
static_assert(!std::is_constructible_v<ProbeParamValue, char const*>,
              "ProbeParamValue must reject char const* to avoid silently binding to bool - construct std::string explicitly");

using ProbeParams = Operon::Map<std::string, ProbeParamValue>;

using ProbeFactory = std::function<std::unique_ptr<GenerationProbe>(ProbeParams const&)>;

// Name -> factory registry, so adding a new probe type never requires
// touching config-parsing code - only a Register() call for it. Plain value
// type, not a singleton: callers (e.g. the CLI) construct one, populate it
// (RegisterBuiltinProbes() plus any of their own), and use it to build a
// ProbeChain from a parsed config. No static/global registry, so tests can
// use an isolated instance without cross-test state.
class OPERON_EXPORT ProbeRegistry {
public:
    auto Register(std::string type, ProbeFactory factory) -> void;

    [[nodiscard]] auto Contains(std::string const& type) const -> bool;

    // Returns nullptr if `type` was never registered - callers are
    // expected to report the unknown type themselves (they have the
    // context, e.g. a config file path/line, that this registry doesn't).
    [[nodiscard]] auto Create(std::string const& type, ProbeParams const& params) const -> std::unique_ptr<GenerationProbe>;

private:
    Operon::Map<std::string, ProbeFactory> factories_;
};

} // namespace Operon

#endif
