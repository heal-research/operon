// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ANALYZERS_PERMUTATION_IMPORTANCE_HPP
#define OPERON_ANALYZERS_PERMUTATION_IMPORTANCE_HPP

#include <algorithm>
#include <cmath>
#include <functional>
#include <numeric>
#include <vector>

#include "operon/analyzers/detail/variables_used_in.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/error_metrics/r2_score.hpp"
#include "operon/interpreter/interpreter.hpp"

namespace Operon {

struct VariableImportance {
    Operon::Hash Variable;
    double Mean; // average, over repeated shuffles, of R2(tree) - R2(tree with this variable's column shuffled)
    double Std;  // population std of the same per-repeat differences
};

// Permutation importance (cf. HeuristicLab's RegressionSolutionVariableImpactsCalculator
// "Shuffle" method, and sklearn's permutation_importance, whose default nRepeats=5 this
// mirrors): for each input variable used by `tree`, shuffle that variable's column across
// the rows in `range` (every other column, including the target, stays fixed), re-evaluate
// the unmodified tree against the perturbed dataset, and take R2(original) - R2(perturbed)
// as the importance for that repeat. Repeated `nRepeats` times per variable and averaged
// (with std reported alongside) rather than taken from a single shuffle, since a single
// shuffle's outcome is itself a random variable.
//
// Unlike NodeImpact (node_impact.hpp), which perturbs the tree structure and leaves the
// dataset untouched, this perturbs the dataset and leaves the tree untouched - it
// naturally aggregates a variable's effect across every subtree it appears in, at the
// cost of needing repeated shuffles (it's measuring a random quantity) rather than a
// single deterministic replacement.
//
// Mutates one column of a single owning copy of `dataset` in place per repeat via
// Dataset::SetValues (restoring the original values before moving to the next variable),
// rather than cloning the whole dataset on every repeat - the clone happens once, not
// nVariables * nRepeats times.
inline auto PermutationImportance(Operon::Tree const& tree, Operon::Dataset const& dataset, Operon::Hash target, Operon::Range range, Operon::RandomGenerator& rng, size_t nRepeats = 5) -> std::vector<VariableImportance>
{
    using Interp = Operon::Interpreter<>;

    EXPECT(range.Size() > 0);
    EXPECT(nRepeats > 0);

    // Sliced to `range`, matching what Evaluate() below actually returns -
    // dataset.GetValues(target) alone is the *whole* column, so comparing it
    // directly against a `range`-sized prediction silently misaligns rows
    // whenever range.Start() != 0.
    auto const actual = dataset.GetValues(target).subspan(range.Start(), range.Size());
    auto const predicted = Interp::Evaluate(tree, dataset, range);
    Operon::Span<Operon::Scalar const> const predictedSpan{ predicted.data(), predicted.size() };
    auto const baseline = Operon::R2Score(predictedSpan, actual);

    Operon::Dataset workingCopy(dataset);

    std::vector<VariableImportance> result;
    for (auto const variable : detail::VariablesUsedIn(tree)) {
        auto const originalSpan = dataset.GetValues(variable).subspan(range.Start(), range.Size());
        std::vector<Operon::Scalar> const original(originalSpan.begin(), originalSpan.end());

        std::vector<double> diffs;
        diffs.reserve(nRepeats);

        // std::shuffle produces a uniform random permutation of whatever the
        // container currently holds, regardless of its starting order, so
        // reshuffling `shuffled` in place across repeats (rather than
        // resetting to `original` each time) is still a uniform-random
        // permutation of the same underlying values on every repeat - just
        // one fewer O(|range|) copy per repeat.
        auto shuffled = original;
        for (size_t rep = 0; rep < nRepeats; ++rep) {
            std::shuffle(shuffled.begin(), shuffled.end(), rng);
            workingCopy.SetValues(variable, range, { shuffled.data(), shuffled.size() });

            auto const perturbedPredicted = Interp::Evaluate(tree, workingCopy, range);
            Operon::Span<Operon::Scalar const> const perturbedSpan{ perturbedPredicted.data(), perturbedPredicted.size() };
            auto const perturbedR2 = Operon::R2Score(perturbedSpan, actual);

            diffs.push_back(baseline - perturbedR2);
        }

        workingCopy.SetValues(variable, range, { original.data(), original.size() });

        auto const mean = std::reduce(diffs.begin(), diffs.end(), 0.0) / static_cast<double>(diffs.size());
        auto const variance = std::transform_reduce(diffs.begin(), diffs.end(), 0.0, std::plus<>{},
            [mean](double d) -> double { return (d - mean) * (d - mean); }) / static_cast<double>(diffs.size());

        result.push_back({ variable, mean, std::sqrt(variance) });
    }

    return result;
}

} // namespace Operon

#endif
