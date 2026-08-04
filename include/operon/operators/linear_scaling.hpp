// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_OPERATORS_LINEAR_SCALING_HPP
#define OPERON_OPERATORS_LINEAR_SCALING_HPP

#include <optional>
#include <utility>

#include "operon/core/types.hpp"
#include "operon/core/tree.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// y = Scale*f(x) + Offset -- Keijzer-style linear scaling fitted per
// individual by ordinary least squares against training targets.
struct OPERON_EXPORT LinearScaling {
    double Scale{1};
    double Offset{0};

    [[nodiscard]] auto IsIdentity() const noexcept -> bool;

    // In-place y <- Scale*y + Offset.
    void ApplyInPlace(Operon::Span<Operon::Scalar> values) const noexcept;

    // [lo,hi] -> Scale*[lo,hi] + Offset. Swaps endpoints when Scale < 0 so
    // the result stays a valid (inf <= sup) interval.
    [[nodiscard]] auto ApplyToValueInterval(Operon::Scalar lo, Operon::Scalar hi) const noexcept
        -> std::pair<Operon::Scalar, Operon::Scalar>;

    // [lo,hi] -> Scale*[lo,hi] (no offset -- d(Scale*f+Offset)/dx = Scale*f').
    // Same endpoint-swap rule as ApplyToValueInterval.
    [[nodiscard]] auto ApplyToDerivativeInterval(Operon::Scalar lo, Operon::Scalar hi) const noexcept
        -> std::pair<Operon::Scalar, Operon::Scalar>;

    // Returns a copy of `tree` with Constant(Scale)+Mul and/or
    // Constant(Offset)+Add appended at the root (skipped individually when
    // Scale == 1 / Offset == 0). Calls UpdateNodes() iff anything was
    // appended.
    [[nodiscard]] auto Materialize(Operon::Tree tree) const -> Operon::Tree;
};

// The single OLS definition used everywhere in this codebase. `weights` may
// be empty (unweighted fit). `omitNonFinite` selects a finite-subset-only
// fit (vstat nan_policy::omit) instead of the plain fit. Never returns a
// non-finite Scale (falls back to 1, matching the pre-existing
// FitLeastSquaresImpl behavior).
[[nodiscard]] OPERON_EXPORT auto FitLinearScaling(
    Operon::Span<Operon::Scalar const> estimated,
    Operon::Span<Operon::Scalar const> target,
    Operon::Span<Operon::Scalar const> weights = {},
    bool omitNonFinite = false) -> LinearScaling;

} // namespace Operon

#endif
