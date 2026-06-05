// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

// Mirror of Operon::NodeType (core/node.hpp) for use in SYCL translation units
// that cannot include node.hpp due to transitive header conflicts between
// Operon's unordered_dense and AdaptiveCpp's bundled version.
//
// Must stay in sync with NodeType. population_encoder.hpp carries a static_assert
// that verifies the values at compile time.

#ifndef OPERON_GPU_NODE_TYPES_HPP
#define OPERON_GPU_NODE_TYPES_HPP

#include <cstdint>

namespace Operon {

enum class NodeType : uint8_t {
    Add = 0, Mul, Sub, Div, Fmin, Fmax,        // n-ary
    Aq, Pow, Powabs,                             // binary
    Abs, Acos, Asin, Atan, Cbrt, Ceil,          // unary
    Cos, Cosh, Exp, Floor, Log, Logabs, Log1p,
    Sin, Sinh, Sqrt, Sqrtabs, Tan, Tanh, Square,
    Dynamic, Constant, Variable                  // nullary
};

} // namespace Operon

#endif // OPERON_GPU_NODE_TYPES_HPP
