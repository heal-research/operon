// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_CONSTRAINT_HPP
#define OPERON_CORE_CONSTRAINT_HPP

#include <optional>
#include <string>
#include <utility>
#include <vector>

#include "operon/core/types.hpp"

namespace Operon {

// Quantity a constraint bounds: the model output itself, or a first-/
// second-order partial derivative w.r.t. one variable (same variable for
// both orders -- no mixed partials).
enum class ShapeConstraintOp {
    Identity,
    FirstDerivative,
    SecondDerivative,
};

// One shape constraint. Exactly one of Sign/Bound is set:
//   - Sign: +1 (non-decreasing/non-negative) or -1 (non-increasing/non-positive)
//   - Bound: [lo, hi] on the quantity Op selects
struct ShapeConstraint {
    ShapeConstraintOp Op{ShapeConstraintOp::Identity};
    std::string Variable; // empty when Op == Identity
    std::optional<int> Sign;
    std::optional<std::pair<Operon::Scalar, Operon::Scalar>> Bound;
};

// A variable domain box plus the constraints to check over it.
struct ShapeConstraintSet {
    Operon::Map<std::string, std::pair<Operon::Scalar, Operon::Scalar>> Domains;
    std::vector<ShapeConstraint> Constraints;
};

} // namespace Operon

#endif
