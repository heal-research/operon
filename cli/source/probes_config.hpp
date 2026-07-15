// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_PROBES_CONFIG_HPP
#define OPERON_CLI_PROBES_CONFIG_HPP

#include <optional>
#include <string>

#include "operon/algorithms/probes/chain.hpp"

namespace Operon {

// Builds a ProbeChain from a JSON config file (--probes-config). Returns
// std::nullopt if `path` is empty (flag not given). Throws
// std::runtime_error on a missing/unreadable file, malformed JSON, or an
// unrecognized probe/sink type - matches this CLI's existing convention
// (e.g. ParseEvaluator/ParseGenerator in operator_factory.cpp) of
// surfacing config errors as exceptions caught by main()'s try/catch.
//
// Schema:
//   {
//     "probes": [
//       { "type": "population_trace", "every": 5, "params": { "path": "pop.beve" } },
//       { "type": "cache_hit_rate", "every": 1 },
//       { "type": "structural_diversity", "every": 10, "params": { "hash_mode": "strict" } }
//     ],
//     "sink": { "type": "jsonl", "path": "metrics.jsonl" }
//   }
// "every"/"offset" default to 1/0; "params" defaults to empty; "sink" is
// optional (a chain with no sink still runs its probes' side effects, e.g.
// population_trace writing its own file, but nothing is recorded via Emit).
auto LoadProbeConfig(std::string const& path) -> std::optional<ProbeChain>;

} // namespace Operon

#endif
