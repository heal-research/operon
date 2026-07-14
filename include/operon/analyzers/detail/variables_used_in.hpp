// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ANALYZERS_DETAIL_VARIABLES_USED_IN_HPP
#define OPERON_ANALYZERS_DETAIL_VARIABLES_USED_IN_HPP

#include <algorithm>
#include <vector>

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon::detail {

// First-seen-order unique variable hashes referenced by `tree` - shared between
// PermutationImportance (permutation_importance.hpp) and GradientImportance
// (gradient_importance.hpp), both of which report one importance value per
// input variable the tree actually reads.
inline auto VariablesUsedIn(Operon::Tree const& tree) -> std::vector<Operon::Hash>
{
    std::vector<Operon::Hash> vars;
    for (auto const& n : tree.Nodes()) {
        if (n.IsVariable() && std::ranges::find(vars, n.HashValue) == vars.end()) {
            vars.push_back(n.HashValue);
        }
    }
    return vars;
}

} // namespace Operon::detail

#endif
