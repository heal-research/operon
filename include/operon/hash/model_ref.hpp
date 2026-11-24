// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_HASH_MODEL_REF_HPP
#define OPERON_HASH_MODEL_REF_HPP

#include <string>

#include "operon/core/tree.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Identity of a model for machine-readable run records: the SHA-256 (lowercase hex) of ModelSerialization(tree).
// Two trees have the same reference exactly when they have the same nodes -- type, symbol, arity, enabled and
// optimize flags, reference target and coefficient/weight value -- in the same order. Coefficients enter with
// every bit: a model whose linear scaling was materialized into the tree carries it in the reference.
[[nodiscard]] OPERON_EXPORT auto ModelRef(Operon::Tree const& tree) -> std::string;

// The canonical text that ModelRef hashes: a version line, the node count, then one line per node in postfix
// order, `<C|V|R|F> <symbol hash, hex> <arity> <enabled> <optimize> <ref target> <value>`. Values are widened to
// double and printed with the shortest representation that reads back to the same double, so the text does not
// depend on the build precision of the producer for the same coefficient value.
[[nodiscard]] OPERON_EXPORT auto ModelSerialization(Operon::Tree const& tree) -> std::string;

} // namespace Operon

#endif
