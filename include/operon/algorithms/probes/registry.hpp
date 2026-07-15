// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_REGISTRY_HPP
#define OPERON_ALGORITHMS_PROBES_REGISTRY_HPP

#include <functional>
#include <memory>
#include <string>
#include <variant>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// A probe's config-file parameters, already parsed down to plain scalars -
// deliberately not glz::json_t (or any glaze type): glaze is a PRIVATE
// dependency confined to source/*.cpp translation units throughout this
// codebase (see core/serialization.cpp's header comment), and ProbeParams
// lives in a public header, so the JSON-vs-YAML-vs-whatever config format
// question stays entirely on the parsing side (cli/source/probes_config.cpp)
// - this registry only ever sees the result.
using ProbeParamValue = std::variant<std::int64_t, double, bool, std::string>;
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
