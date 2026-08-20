// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/linear_scaling.hpp"

#include "operon/interpreter/interpreter.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <tuple>
#include <type_traits>

#include <vstat/vstat.hpp>

namespace Operon {
namespace {
    template<typename T>
    auto FitLeastSquaresImpl(Operon::Span<T const> estimated, Operon::Span<T const> target,
                             Operon::Span<T const> weights = {}) -> std::pair<double, double>
    requires std::is_arithmetic_v<T>
    {
        auto stats = weights.empty()
            ? vstat::bivariate::accumulate<T>(estimated.data(), estimated.data() + estimated.size(), target.data())
            : vstat::bivariate::accumulate<T>(estimated.data(), estimated.data() + estimated.size(), target.data(), weights.data());
        auto a = stats.covariance / stats.variance_x; // scale
        if (!std::isfinite(a)) {
            a = 1;
        }
        auto b = stats.mean_y - (a * stats.mean_x); // offset
        return {a, b};
    }

    // Finite-aware variant: computes scale/offset from the finite subset
    // only, returning the count of skipped (non-finite) pairs. A NaN/Inf row
    // no longer disables scaling for every finite row via NaN-poisoned stats.
    // NaN/Inf rows preserve their non-finiteness through `a*x + b`
    // (NaN->NaN; Inf->+/-Inf when a != 0; Inf->NaN when a == 0 since
    // 0*Inf==NaN before adding b), so the downstream FiniteSubset metric
    // still detects and skips them after the in-place transform regardless
    // of which non-finite value the scaling produces -- both NaN and Inf
    // are excluded by the shared finiteness mask. (The symmetric "could a
    // *previously finite* row become non-finite after scaling?" overflow
    // case is a pre-existing risk of any linear scaling, not introduced
    // or worsened by finite-aware scaling.) See `NormalizedMeanSquaredErrorFinite`
    // / `MeanSquaredErrorFinite` for the mask this composes through.
    template<typename T>
    auto FitLeastSquaresFiniteImpl(Operon::Span<T const> estimated, Operon::Span<T const> target,
                                   Operon::Span<T const> weights = {}) -> std::tuple<double, double, std::size_t>
    requires std::is_arithmetic_v<T>
    {
        auto [stats, skipped] = weights.empty()
            ? vstat::bivariate::accumulate<T, vstat::nan_policy::omit>(estimated.data(), estimated.data() + estimated.size(), target.data())
            : vstat::bivariate::accumulate<T, vstat::nan_policy::omit>(estimated.data(), estimated.data() + estimated.size(), target.data(), weights.data());
        auto a = stats.covariance / stats.variance_x; // scale
        if (!std::isfinite(a)) {
            a = 1;
        }
        auto b = stats.mean_y - (a * stats.mean_x); // offset
        return {a, b, skipped};
    }
} // namespace

[[nodiscard]] auto LinearScaling::IsIdentity() const noexcept -> bool
{
    return Scale == Operon::Scalar{1} && Offset == Operon::Scalar{0};
}

void LinearScaling::ApplyInPlace(Operon::Span<Operon::Scalar> values) const noexcept
{
    std::ranges::transform(values, values.begin(), [this](auto x) { return (Scale * x) + Offset; });
}

[[nodiscard]] auto LinearScaling::ApplyToValueInterval(Operon::Scalar lo, Operon::Scalar hi) const noexcept
    -> std::pair<Operon::Scalar, Operon::Scalar>
{
    return Scale >= Operon::Scalar{0}
        ? std::pair{(Scale * lo) + Offset, (Scale * hi) + Offset}
        : std::pair{(Scale * hi) + Offset, (Scale * lo) + Offset};
}

[[nodiscard]] auto LinearScaling::ApplyToDerivativeInterval(Operon::Scalar lo, Operon::Scalar hi) const noexcept
    -> std::pair<Operon::Scalar, Operon::Scalar>
{
    return Scale >= Operon::Scalar{0}
        ? std::pair{Scale * lo, Scale * hi}
        : std::pair{Scale * hi, Scale * lo};
}

[[nodiscard]] auto LinearScaling::Materialize(Operon::Tree tree) const -> Operon::Tree
{
    auto& nodes = tree.Nodes();
    auto const sz = nodes.size();
    if (std::abs(Scale - Operon::Scalar{1}) > std::numeric_limits<Operon::Scalar>::epsilon()) {
        nodes.emplace_back(Operon::Node::Constant(static_cast<Operon::Scalar>(Scale)));
        nodes.push_back(Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Mul), 2));
    }
    if (std::abs(Offset) > std::numeric_limits<Operon::Scalar>::epsilon()) {
        nodes.emplace_back(Operon::Node::Constant(static_cast<Operon::Scalar>(Offset)));
        nodes.push_back(Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Add), 2));
    }
    if (nodes.size() > sz) {
        tree.UpdateNodes();
    }
    return tree;
}

[[nodiscard]] auto FitLinearScaling(Operon::Span<Operon::Scalar const> estimated,
                                    Operon::Span<Operon::Scalar const> target,
                                    Operon::Span<Operon::Scalar const> weights,
                                    bool omitNonFinite) -> LinearScaling
{
    auto const [a, b] = [&] {
        if (omitNonFinite) {
            auto [scale, offset, skipped] = weights.empty()
                ? FitLeastSquaresFiniteImpl<Operon::Scalar>(estimated, target)
                : FitLeastSquaresFiniteImpl<Operon::Scalar>(estimated, target, weights);
            (void)skipped;
            return std::pair{scale, offset};
        }
        return weights.empty()
            ? FitLeastSquaresImpl<Operon::Scalar>(estimated, target)
            : FitLeastSquaresImpl<Operon::Scalar>(estimated, target, weights);
    }();
    return LinearScaling{a, b};
}

[[nodiscard]] auto FitLinearScaling(Operon::Tree const& tree, Operon::Problem const& problem,
                                    Operon::ScalarDispatch const& dtable, Operon::Range range)
    -> std::optional<LinearScaling>
{
    if (!problem.LinearScalingEnabled()) {
        return std::nullopt;
    }

    auto const* dataset = problem.GetDataset();
    Interpreter<Operon::Scalar, ScalarDispatch> const interpreter{&dtable, dataset, &tree};
    Operon::Vector<Operon::Scalar> estimatedValues(range.Size());
    auto coeff = tree.GetCoefficients();
    interpreter.Evaluate(coeff, range, estimatedValues);

    return FitLinearScaling(estimatedValues, problem.TargetValues(range),
        problem.Weights(range).value_or(Operon::Span<Operon::Scalar const>{}),
        problem.LinearScalingOmitsNonFinite());
}

} // namespace Operon
