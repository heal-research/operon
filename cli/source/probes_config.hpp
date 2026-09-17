// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_PROBES_CONFIG_HPP
#define OPERON_CLI_PROBES_CONFIG_HPP

#include <optional>
#include <string>

#include "cli_error.hpp"
#include "operon/algorithms/probes/chain.hpp"

namespace Operon {

// Builds a ProbeChain from a JSON config file (--probes-config). Returns
// std::nullopt if `path` is empty (flag not given). File I/O failures return
// Cli::ErrorCode::Input; malformed JSON, schema violations, and invalid probe
// or sink configuration return Cli::ErrorCode::Configuration.
//
// Schema:
//   {
//     "probes": [
//       { "type": "population_trace", "every": 5, "params": { "path": "pop.beve" } },
//       { "type": "cache_hit_rate", "every": 1 },
//       { "type": "shape_feasibility", "every": 1 },
//       { "type": "structural_diversity", "every": 10, "params": { "hash_mode": "strict" } }
//     ],
//     "sink": { "type": "jsonl", "path": "metrics.jsonl" }
//   }
// "every"/"offset" default to 1/0; "params" defaults to empty; "sink" is
// optional (a chain with no sink still runs its probes' side effects, e.g.
// population_trace writing its own file, but nothing is recorded via Emit).
//
// JsonlSink and PopulationTraceProbe both truncate their output file on
// construction (see chain.cpp/population_trace.cpp), so combining
// --resume with a --probes-config that reuses the same output paths as
// the resumed run discards that run's prior instrumentation history -
// the CLI warns about this at the call site, this isn't handled here.
auto LoadProbeConfig(std::string const& path) -> Cli::Result<std::optional<ProbeChain>>;

} // namespace Operon

#endif
