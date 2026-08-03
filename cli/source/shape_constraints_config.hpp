// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_SHAPE_CONSTRAINTS_CONFIG_HPP
#define OPERON_CLI_SHAPE_CONSTRAINTS_CONFIG_HPP

#include <optional>
#include <string>

#include "operon/core/constraint.hpp"

namespace Operon {

// Builds a ShapeConstraintSet from a JSON config file (--shape-constraints-config).
// Returns std::nullopt if `path` is empty (flag not given). Throws
// std::runtime_error on a missing/unreadable file, malformed JSON, or an
// entry that isn't well-formed -- same convention as LoadProbeConfig
// (probes_config.hpp).
//
// Schema (mirrors operon-publications' shape-constraints-reproduction/
// problems.yml's `variables`/`constraints` fields directly, so a problem
// entry there projects into this config mechanically, not by hand):
//   {
//     "domains": { "p": [0.1, 15], "v": [0.01, 3], "T": [-50, 250] },
//     "constraints": [
//       { "op": "id", "bound": [0.0, 1.0] },
//       { "op": "d/dp", "sign": -1 },
//       { "op": "d2/dphi2", "sign": -1 }
//     ]
//   }
// `op` is one of: "id" (the model's own output), "d/d<var>" (first
// derivative w.r.t. <var>), "d2/d<var>2" (second derivative w.r.t. the
// same <var> -- mixed second partials aren't needed by any problem in
// this codebase's benchmark set and aren't supported). Each constraint
// entry must set exactly one of "sign" (+1 non-decreasing/non-negative,
// -1 non-increasing/non-positive, threshold implicitly 0) or "bound"
// ([lo, hi] on the selected quantity).
auto LoadShapeConstraints(std::string const& path) -> std::optional<ShapeConstraintSet>;

} // namespace Operon

#endif
