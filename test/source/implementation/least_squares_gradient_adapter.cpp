// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <cmath>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <lbfgs/solver.hpp>

#include "operon/optimizer/detail/gradient_solver_adapter.hpp"
#include "operon/optimizer/least_squares_gradient_adapter.hpp"
#include "operon/optimizer/solvers/sgd.hpp"

namespace {

using Extents = std::dextents<std::size_t, 2>;
using Mapping = std::layout_stride::mapping<Extents>;

// y = c0 + c1 * x, Jacobian columns [1, x_i]. No likelihood/Fisher method.
class LinearModelCost final : public Operon::LeastSquaresCostFunction {
public:
    LinearModelCost(std::vector<Operon::Scalar> x, std::vector<Operon::Scalar> y)
        : x_(std::move(x))
        , y_(std::move(y))
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return x_.size(); }

    [[nodiscard]] auto Evaluate(
        std::span<Operon::Scalar const> parameters,
        std::span<Operon::Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> jacobian) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        for (std::size_t i = 0; i < x_.size(); ++i) {
            residuals[i] = static_cast<Operon::Scalar>(parameters[0] + (parameters[1] * x_[i]) - y_[i]);
        }
        if (jacobian) {
            for (std::size_t i = 0; i < x_.size(); ++i) {
                Operon::At(*jacobian, i, 0) = Operon::Scalar { 1 };
                Operon::At(*jacobian, i, 1) = x_[i];
            }
        }
        return {};
    }

private:
    std::vector<Operon::Scalar> x_;
    std::vector<Operon::Scalar> y_;
};

// Always fails; no statistical method.
class FailingCost final : public Operon::LeastSquaresCostFunction {
public:
    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 4; }

    [[nodiscard]] auto Evaluate(
        std::span<Operon::Scalar const> /*parameters*/,
        std::span<Operon::Scalar> /*residuals*/,
        std::optional<Operon::ScalarMatrixView> /*jacobian*/) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        return tl::unexpected(Operon::LeastSquaresError { .Code = Operon::LeastSquaresErrorCode::NonFiniteEvaluation });
    }
};

auto MakeLinearFixture(std::size_t n, Operon::Scalar c0, Operon::Scalar c1) -> LinearModelCost
{
    std::vector<Operon::Scalar> x(n);
    std::vector<Operon::Scalar> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<Operon::Scalar>(i) - (static_cast<Operon::Scalar>(n) / Operon::Scalar { 2 });
        y[i] = static_cast<Operon::Scalar>(c0 + (c1 * x[i]));
    }
    return LinearModelCost { std::move(x), std::move(y) };
}

} // namespace

TEST_CASE("LeastSquaresGradientAdapter drives lbfgs::solver to the true optimum", "[least-squares][gradient-adapter]")
{
    auto const c0 = Operon::Scalar { 1.5 };
    auto const c1 = Operon::Scalar { -0.8 };
    auto cost = MakeLinearFixture(30, c0, c1);
    Operon::LeastSquaresGradientAdapter adapter { &cost };
    Operon::detail::GradientSolverAdapter<Operon::LeastSquaresGradientAdapter> bridge { &adapter };

    lbfgs::solver solver { bridge };
    solver.max_iterations = 200;
    Eigen::Matrix<Operon::Scalar, -1, 1> x0(2);
    x0 << 0, 0;
    auto result = solver.optimize(x0);

    REQUIRE(result.has_value());
    CHECK_THAT(static_cast<double>((*result)[0]), Catch::Matchers::WithinAbs(static_cast<double>(c0), 1e-3));
    CHECK_THAT(static_cast<double>((*result)[1]), Catch::Matchers::WithinAbs(static_cast<double>(c1), 1e-3));
    CHECK_FALSE(bridge.Error().has_value());
    CHECK_FALSE(adapter.Error().has_value());
}

TEST_CASE("LeastSquaresGradientAdapter drives SGDSolver to reduce the objective", "[least-squares][gradient-adapter]")
{
    auto const c0 = Operon::Scalar { 0.6 };
    auto const c1 = Operon::Scalar { 1.1 };
    auto cost = MakeLinearFixture(30, c0, c1);
    Operon::LeastSquaresGradientAdapter adapter { &cost };
    Operon::detail::GradientSolverAdapter<Operon::LeastSquaresGradientAdapter> bridge { &adapter };

    Operon::UpdateRule::Adam<Operon::Scalar> rule { 2 };
    Operon::SGDSolver<decltype(bridge)> solver { &bridge, &rule };

    Eigen::Array<Operon::Scalar, -1, 1> x0(2);
    x0 << 0, 0;

    std::array<Operon::Scalar, 2> initialGradient {};
    auto initialCost = bridge(x0, initialGradient);

    auto xFinal = solver.Optimize(x0, 500);
    std::array<Operon::Scalar, 2> finalGradient {};
    auto finalCost = bridge(xFinal, finalGradient);

    REQUIRE_FALSE(std::isnan(static_cast<double>(finalCost)));
    CHECK(finalCost < initialCost);
}

TEST_CASE("LeastSquaresGradientAdapter: weighted fitting matches ComputeGradient directly", "[least-squares][gradient-adapter]")
{
    auto cost = MakeLinearFixture(10, Operon::Scalar { 0.4 }, Operon::Scalar { -0.6 });
    std::vector<Operon::Scalar> weights(10);
    for (std::size_t i = 0; i < weights.size(); ++i) { weights[i] = static_cast<Operon::Scalar>(1 + i); }

    Operon::LeastSquaresGradientAdapter adapter { &cost, weights };
    std::array<Operon::Scalar, 2> params { 0.1, 0.2 };
    std::array<Operon::Scalar, 2> gradient {};
    auto result = adapter.Evaluate(params, gradient);
    REQUIRE(result.has_value());

    std::vector<Operon::Scalar> residuals(cost.NumResiduals());
    std::vector<Operon::Scalar> jacBuffer(cost.NumResiduals() * 2);
    Operon::ScalarMatrixView jac { jacBuffer.data(), Mapping { Extents { cost.NumResiduals(), 2 }, std::array<std::size_t, 2> { 2, 1 } } };
    REQUIRE(cost.Evaluate(params, residuals, jac).has_value());
    std::array<Operon::Scalar, 2> expectedGradient {};
    auto expectedCost = Operon::ComputeGradient(residuals, jac, expectedGradient, weights);
    REQUIRE(expectedCost.has_value());

    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(*expectedCost, 1e-4));
    CHECK_THAT(static_cast<double>(gradient[0]), Catch::Matchers::WithinAbs(static_cast<double>(expectedGradient[0]), 1e-4));
    CHECK_THAT(static_cast<double>(gradient[1]), Catch::Matchers::WithinAbs(static_cast<double>(expectedGradient[1]), 1e-4));
}

TEST_CASE("LeastSquaresGradientAdapter: a failing cost propagates a typed error and NaN to both bridges", "[least-squares][gradient-adapter]")
{
    FailingCost cost;
    Operon::LeastSquaresGradientAdapter adapter { &cost };
    Operon::detail::GradientSolverAdapter<Operon::LeastSquaresGradientAdapter> bridge { &adapter };

    Eigen::Matrix<Operon::Scalar, -1, 1> params(2);
    params << 0, 0;
    Eigen::Matrix<Operon::Scalar, -1, 1> gradient(2);
    auto value = bridge(params, gradient);

    CHECK(std::isnan(static_cast<double>(value)));
    REQUIRE(bridge.Error().has_value());
    CHECK(bridge.Error()->Code == Operon::GradientErrorCode::NonFiniteEvaluation);
    REQUIRE(adapter.Error().has_value());
    CHECK(adapter.Error()->Code == Operon::GradientErrorCode::NonFiniteEvaluation);
    for (auto g : gradient) { CHECK(std::isnan(static_cast<double>(g))); }
}

TEST_CASE("LeastSquaresGradientAdapter: repeated full-batch evaluations are deterministic", "[least-squares][gradient-adapter]")
{
    auto cost = MakeLinearFixture(20, Operon::Scalar { 0.9 }, Operon::Scalar { -1.4 });
    Operon::LeastSquaresGradientAdapter adapter { &cost };
    std::array<Operon::Scalar, 2> params { 0.2, -0.3 };

    std::array<Operon::Scalar, 2> firstGradient {};
    auto firstCost = adapter.Evaluate(params, firstGradient);
    REQUIRE(firstCost.has_value());

    for (int replay = 0; replay < 5; ++replay) {
        std::array<Operon::Scalar, 2> gradient {};
        auto cost2 = adapter.Evaluate(params, gradient);
        REQUIRE(cost2.has_value());
        CHECK(*cost2 == *firstCost);
        CHECK(gradient == firstGradient);
    }
}

TEST_CASE("ToGradientError maps every LeastSquaresErrorCode to a distinct GradientErrorCode and preserves location", "[least-squares][gradient-adapter]")
{
    using LS = Operon::LeastSquaresErrorCode;
    using GR = Operon::GradientErrorCode;
    constexpr std::array<std::pair<LS, GR>, 6> table { {
        { LS::InvalidShape, GR::InvalidShape },
        { LS::InvalidView, GR::InvalidView },
        { LS::InvalidWeights, GR::InvalidWeights },
        { LS::NonFiniteEvaluation, GR::NonFiniteEvaluation },
        { LS::NumericalFailure, GR::NumericalFailure },
        { LS::EvaluationFailure, GR::EvaluationFailure },
    } };

    std::vector<GR> seen;
    for (auto const& [source, expected] : table) {
        Operon::LeastSquaresError error {
            .Code = source, .Expected = 11, .Actual = 7, .Row = 3, .Column = 5,
            .Cause = Operon::InterpreterError { .Kind = Operon::InterpreterError::Code::MissingVariable, .Hash = 42 }
        };
        auto const converted = Operon::ToGradientError(error);
        CHECK(converted.Code == expected);
        CHECK(converted.Expected == 11);
        CHECK(converted.Actual == 7);
        CHECK(converted.Row == 3);
        CHECK(converted.Column == 5);
        REQUIRE(converted.Cause.has_value());
        CHECK(converted.Cause->Kind == Operon::InterpreterError::Code::MissingVariable);
        CHECK(converted.Cause->Hash == 42);
        CHECK(std::ranges::count(seen, converted.Code) == 0);
        seen.push_back(converted.Code);
    }
    CHECK(seen.size() == table.size());
}

TEST_CASE("ToGradientError(WeightError) reports InvalidWeights with size and row", "[least-squares][gradient-adapter]")
{
    auto const converted = Operon::ToGradientError(Operon::WeightError {
        .Code = Operon::WeightErrorCode::NegativeValue, .Expected = 9, .Actual = 9, .Row = 4 });
    CHECK(converted.Code == Operon::GradientErrorCode::InvalidWeights);
    CHECK(converted.Expected == 9);
    CHECK(converted.Actual == 9);
    CHECK(converted.Row == 4);
    CHECK_FALSE(converted.Cause.has_value());
}

TEST_CASE("LeastSquaresGradientAdapter: invalid weights are a typed InvalidWeights error with NaN gradient", "[least-squares][gradient-adapter]")
{
    auto cost = MakeLinearFixture(6, Operon::Scalar { 0.4 }, Operon::Scalar { -0.6 });
    std::vector<Operon::Scalar> weights(6, Operon::Scalar { 1 });
    weights[4] = Operon::Scalar { -2 };

    Operon::LeastSquaresGradientAdapter adapter { &cost, weights };
    std::array<Operon::Scalar, 2> params { 0.1, 0.2 };
    std::array<Operon::Scalar, 2> gradient {};
    auto result = adapter.Evaluate(params, gradient);

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().Code == Operon::GradientErrorCode::InvalidWeights);
    CHECK(result.error().Row == 4);
    for (auto g : gradient) { CHECK(std::isnan(static_cast<double>(g))); }
}
