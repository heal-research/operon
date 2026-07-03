// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INFORMATION_CRITERIA_FRACTIONAL_BAYES_FACTOR_HPP
#define OPERON_INFORMATION_CRITERIA_FRACTIONAL_BAYES_FACTOR_HPP

#include <algorithm>
#include <cmath>
#include <limits>

#include "operon/core/constants.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "weighted_complexity.hpp"

namespace Operon {

// Fractional Bayes Factor model-selection score (O'Hagan 1995 FBF +
// Bartlett et al. 2023, arXiv:2304.06333, Sec 2.3: structural complexity
// term shared with MinimumDescriptionLength, parameter-width term from a
// rectangular quantizing lattice with b = 1/sqrt(n), and the (1-b)-weighted
// negative log-likelihood).
inline auto FractionalBayesFactor(Tree const& tree, double n, double nll) -> double
{
    auto const p = static_cast<double>(std::count_if(tree.Nodes().begin(), tree.Nodes().end(),
                                                       [](auto const& node) -> bool { return node.Optimize; }));
    auto [k, fCompl] = WeightedComplexity(tree);
    (void)k;

    auto const b             = 1.0 / std::sqrt(n);
    auto const fbfParams     = (p / 2.0) * ((0.5 * std::log(n)) + std::log(Operon::Math::Tau) + 1.0 - std::log(3.0)); // NOLINT(cppcoreguidelines-avoid-magic-numbers,readability-magic-numbers)
    auto const fbfLikelihood = (1.0 - b) * nll;
    auto const fbf           = fCompl + fbfParams + fbfLikelihood;
    return std::isfinite(fbf) ? fbf : std::numeric_limits<double>::quiet_NaN();
}

} // namespace Operon

#endif
