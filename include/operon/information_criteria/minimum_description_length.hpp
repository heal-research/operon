// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INFORMATION_CRITERIA_MINIMUM_DESCRIPTION_LENGTH_HPP
#define OPERON_INFORMATION_CRITERIA_MINIMUM_DESCRIPTION_LENGTH_HPP

#include <cmath>
#include <limits>

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "weighted_complexity.hpp"

namespace Operon {

// Minimum Description Length of a fitted tree (Bartlett et al. 2023,
// arXiv:2304.06333, Sec 2.2, Eq. 5): codelength of the tree structure
// (WeightedComplexity) plus a per-parameter uniform-prior quantization cost
// derived from the diagonal Fisher information, plus the negative
// log-likelihood.
//
// `fisherDiag` is the diagonal of the Fisher information matrix for
// `coeffs`, in the same order as the tree's Optimize-flagged nodes —
// likelihood-agnostic: pass whatever Fisher diagonal your likelihood model
// produces (see GaussianLikelihood/PoissonLikelihood::ComputeFisherMatrix).
// This function does not itself know how the model was scaled to produce
// its predictions (e.g. GP's linear-scaling a,b) — if the caller applies
// such a scaling, the Jacobian/Fisher matrix used to derive `fisherDiag`
// must reflect it (d(a*tree)/d(coeffs) = a * d(tree)/d(coeffs)), or the
// parameter cost will be biased by the missing scale factor.
template<typename FisherDiag>
auto MinimumDescriptionLength(Tree const& tree, Operon::Span<Operon::Scalar const> coeffs,
                              FisherDiag const& fisherDiag, double nll) -> double
{
    constexpr auto eps               = std::numeric_limits<Operon::Scalar>::epsilon();
    constexpr auto uniformPriorScale = 12.0; // di = sqrt(12 / fi) comes from Var(Uniform[-c,c]) = (2c)²/12.
    // Generous floating-point safety margin for the Fisher diagonal's PSD
    // invariant (J^T J / sigma^2 is PSD in exact arithmetic, but near-zero
    // entries can round slightly negative) — not a statistical/physical
    // bound, just noise tolerance. Matches the eigenvalue guard used for
    // the same reason in the (experimental) joint-Fisher-matrix variant.
    constexpr auto fisherNoiseFloor  = -1e-8;

    auto [k, fCompl] = WeightedComplexity(tree);
    (void)k;
    auto const p = static_cast<double>(coeffs.size());

    auto cComplexity = fCompl;
    auto cParameters = 0.0;
    auto pi          = 0;
    for (auto const& node : tree.Nodes()) {
        if (node.Optimize) {
            auto fi = static_cast<double>(fisherDiag(pi));
            // fi == 0 is legitimate (a parameter with zero Fisher information
            // truly carries no cost — handled below via the ordinary
            // isfinite(di) quantization check, since sqrt(12/0) = inf).
            // Non-finite, or negative beyond plausible rounding noise, instead
            // violates the Fisher diagonal's PSD invariant and signals
            // invalid/corrupted input upstream, not "no information" — flag
            // rather than silently charging zero cost. A tiny negative value
            // within the noise floor is clamped to 0 and falls through to the
            // same legitimate-zero-info handling.
            if (!std::isfinite(fi) || fi < fisherNoiseFloor) { return std::numeric_limits<double>::quiet_NaN(); }
            fi = std::max(fi, 0.0);
            auto const di = std::sqrt(uniformPriorScale / fi);
            auto const ci = std::abs(static_cast<double>(coeffs[pi]));
            if (std::isfinite(ci) && std::isfinite(di) && ci / di >= 1.0) {
                cParameters += (0.5 * std::log(fi)) + std::log(ci); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
            }
            ++pi;
        } else {
            if (std::abs(node.Value) >= static_cast<double>(eps)) {
                cComplexity += std::log(std::abs(node.Value));
            }
        }
    }
    cParameters -= (p / 2.0) * std::log(3.0); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    auto const mdl = cComplexity + cParameters + nll;
    return std::isfinite(mdl) ? mdl : std::numeric_limits<double>::quiet_NaN();
}

} // namespace Operon

#endif
