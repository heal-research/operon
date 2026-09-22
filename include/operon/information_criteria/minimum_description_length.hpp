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
// (StructureDescriptionLength) plus a per-parameter uniform-prior
// quantization cost derived from the diagonal Fisher information
// (ParameterDescriptionLength, see below), plus the negative log-likelihood.
// See MinimumDescriptionLength's own doc comment further down for the
// combining function; the two components are split out separately so a
// caller (e.g. GrammarEnumerationAlgorithm's MDL ranking) can report or
// otherwise use either half on its own.

// Structural codelength component of MinimumDescriptionLength: WeightedComplexity's
// f-complexity term plus, for every *fixed* (Optimize == false) Constant leaf whose
// |value| is non-negligible, log|value| (a fixed constant still costs bits to encode,
// even though it isn't fit - e.g. Cube's exponent 3, TenExp's base 10, Log10Abs's
// 1/ln(10) scale). Depends only on the tree, not on any fitted coefficients/Fisher
// information. NOT called by GrammarEnumerationAlgorithm's MDL ranking (see
// MakeMdlScorer in algorithms/enumeration.hpp): that ranking's `structureBits`
// is only a canonical-class multiplicity penalty (log2 of the class's pre-canonical
// bucket size), not this per-tree structural codelength, trading per-tree structure
// richness for enumeration tractability. Callers wanting the fuller per-tree MDL
// structure term (e.g. a future ranking, or diagnostics) can call this directly.
inline auto StructureDescriptionLength(Tree const& tree) -> double
{
    constexpr auto eps = std::numeric_limits<Operon::Scalar>::epsilon();
    auto [k, fCompl] = WeightedComplexity(tree);
    (void)k;
    auto c = fCompl;
    for (auto const& node : tree.Nodes()) {
        if (!node.Optimize && std::abs(node.Value) >= static_cast<double>(eps)) {
            c += std::log(std::abs(node.Value));
        }
    }
    return c;
}

// Parameter codelength component of MinimumDescriptionLength (Bartlett et al. 2023,
// arXiv:2304.06333, Sec 2.2, Eq. 5): a per-parameter uniform-prior quantization cost
// derived from the diagonal Fisher information. `fisherDiag` is the diagonal of the
// Fisher information matrix for `coeffs`, in the same order as the tree's
// Optimize-flagged nodes - likelihood-agnostic: pass whatever Fisher diagonal your
// likelihood model produces (see GaussianLikelihood/PoissonLikelihood::ComputeFisherMatrix).
// Returns NaN if the Fisher diagonal violates its PSD invariant (see the noise-floor
// comment below) - propagate, don't silently charge zero cost.
template<typename FisherDiag>
auto ParameterDescriptionLength(Operon::Span<Operon::Scalar const> coeffs, FisherDiag const& fisherDiag) -> double
{
    constexpr auto uniformPriorScale = 12.0; // di = sqrt(12 / fi) comes from Var(Uniform[-c,c]) = (2c)²/12.
    // Generous floating-point safety margin for the Fisher diagonal's PSD
    // invariant: J^T J / sigma^2 is PSD in exact arithmetic, but a
    // near-zero entry can round slightly negative under floating-point
    // rounding. Not a statistical/physical bound — just noise tolerance,
    // chosen well below any Fisher magnitude that would plausibly arise
    // from real (non-degenerate) coefficients/data.
    constexpr auto fisherNoiseFloor  = -1e-8;

    auto const p = static_cast<double>(coeffs.size());
    auto cParameters = 0.0;
    for (auto pi = 0; pi < static_cast<int>(coeffs.size()); ++pi) {
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
    }
    cParameters -= (p / 2.0) * std::log(3.0); // NOLINT(cppcoreguidelines-avoid-magic-numbers)
    return cParameters;
}

// Combines StructureDescriptionLength(tree) + ParameterDescriptionLength(coeffs,
// fisherDiag) + nll into the total MDL score, in nats. This function does not itself
// know how the model was scaled to produce its predictions (e.g. GP's linear-scaling
// a,b) - if the caller applies such a scaling, the Jacobian/Fisher matrix used to
// derive `fisherDiag` must reflect it (d(a*tree)/d(coeffs) = a * d(tree)/d(coeffs)),
// or the parameter cost will be biased by the missing scale factor.
template<typename FisherDiag>
auto MinimumDescriptionLength(Tree const& tree, Operon::Span<Operon::Scalar const> coeffs,
                              FisherDiag const& fisherDiag, double nll) -> double
{
    auto const cComplexity = StructureDescriptionLength(tree);
    auto const cParameters = ParameterDescriptionLength(coeffs, fisherDiag);
    auto const mdl = cComplexity + cParameters + nll;
    return std::isfinite(mdl) ? mdl : std::numeric_limits<double>::quiet_NaN();
}

} // namespace Operon

#endif
