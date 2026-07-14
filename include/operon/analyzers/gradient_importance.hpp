// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ANALYZERS_GRADIENT_IMPORTANCE_HPP
#define OPERON_ANALYZERS_GRADIENT_IMPORTANCE_HPP

#include <cmath>
#include <functional>
#include <numeric>
#include <utility>
#include <vector>

#include "operon/analyzers/detail/variables_used_in.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/dataset.hpp"
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

    EXPECT(range.Size() > 0);

    auto const vars = detail::VariablesUsedIn(tree);

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

// Whole-dataset convenience overload. This is post-hoc analysis on an
// already-fixed tree - unlike training, there's no leakage risk in reading
// test rows here, and using every row available gives a less noisy result
// than restricting to a training-only range. Pass an explicit range only if
// there's a specific reason to isolate a subset (e.g. comparing train vs.
// test importance).
inline auto GradientImportance(Operon::Tree const& tree, Operon::Dataset const& dataset) -> std::vector<std::pair<Operon::Hash, double>>
{
    return GradientImportance(tree, dataset, Operon::Range(0, dataset.Rows<std::size_t>()));
}

} // namespace Operon

#endif
