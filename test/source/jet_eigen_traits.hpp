// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_TEST_JET_EIGEN_TRAITS_HPP
#define OPERON_TEST_JET_EIGEN_TRAITS_HPP

#include <limits>

#include <Eigen/Core>

#include "operon/core/jet.hpp"

// Test-only: lets Operon::Jet be the element type of Eigen arrays (the
// forward-mode oracle in operon_test.hpp evaluates whole trees with
// Eigen::Array<Jet, ...>). Operon::Jet itself deliberately has no Eigen
// dependency, so these specializations live in the test harness.
namespace Eigen {

template <typename T, int N, Operon::JetOrder Order> struct NumTraits<Operon::Jet<T, N, Order>> {
    using Real = Operon::Jet<T, N, Order>;
    using NonInteger = Operon::Jet<T, N, Order>;
    using Nested = Operon::Jet<T, N, Order>;
    using Literal = Operon::Jet<T, N, Order>;

    static auto dummy_precision() -> Real { return Real(T(1e-12)); }
    static auto epsilon() -> Real { return Real(std::numeric_limits<T>::epsilon()); }
    static auto digits10() -> int { return NumTraits<T>::digits10(); }
    static auto highest() -> Real { return Real((std::numeric_limits<T>::max)()); }
    static auto lowest() -> Real { return Real(-(std::numeric_limits<T>::max)()); }

    enum {
        IsComplex = 0,
        IsInteger = 0,
        IsSigned = 1,
        ReadCost = 1,
        AddCost = 1,
        // a Jet product costs ~N multiply-adds on top of the scalar one
        MulCost = 3,
        HasFloatingPoint = 1,
        RequireInitialization = 1
    };

    template <bool Vectorized> struct Div {
        enum { AVX = false, Cost = 3 };
    };
};

// Jet <op> scalar and scalar <op> Jet are element-wise Jet operations.
template <typename BinaryOp, typename T, int N, Operon::JetOrder Order>
struct ScalarBinaryOpTraits<Operon::Jet<T, N, Order>, T, BinaryOp> {
    using ReturnType = Operon::Jet<T, N, Order>;
};

template <typename BinaryOp, typename T, int N, Operon::JetOrder Order>
struct ScalarBinaryOpTraits<T, Operon::Jet<T, N, Order>, BinaryOp> {
    using ReturnType = Operon::Jet<T, N, Order>;
};

} // namespace Eigen

#endif
