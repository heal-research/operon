// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ANALYZERS_GRADIENT_IMPORTANCE_HPP
#define OPERON_ANALYZERS_GRADIENT_IMPORTANCE_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

// Cheap complement to PermutationImportance (permutation_importance.hpp): for each
// input variable used by `tree`, the importance is mean(|d(prediction)/d(variable)|)
// over `range` - a single reverse-mode sweep per variable, no dataset perturbation or
// repeated re-evaluation needed. Agrees with permutation importance for smooth models,
// but stays meaningful on small datasets where a shuffle-based estimate is noisy, and
// is local/exact rather than a finite-difference approximation (see
// Interpreter::JacRevVariable).
inline auto GradientImportance(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Range range) -> std::vector<std::pair<Operon::Hash, double>>
{
    using Interp = Operon::Interpreter<>;

    std::vector<Operon::Hash> vars;
    for (auto const& n : tree.Nodes()) {
        if (n.IsVariable() && std::ranges::find(vars, n.HashValue) == vars.end()) {
            vars.push_back(n.HashValue);
        }
    }

    Operon::ScalarDispatch dtable;
    Interp const interpreter{ &dtable, &dataset, &tree };
    auto const coeff = tree.GetCoefficients();

    std::vector<std::pair<Operon::Hash, double>> result;
    result.reserve(vars.size());
    for (auto const variable : vars) {
        auto const derivative = interpreter.JacRevVariable(coeff, range, variable);
        auto const sum = std::transform_reduce(derivative.begin(), derivative.end(), Operon::Scalar{0}, std::plus<>{},
            [](Operon::Scalar d) -> Operon::Scalar { return std::abs(d); });
        result.emplace_back(variable, static_cast<double>(sum) / static_cast<double>(derivative.size()));
    }

    return result;
}

} // namespace Operon

#endif
