// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

// Covers LeastSquaresCostFunction, ValidateWeights, ComputeDiagnostics, ComputeGradient, and ComputeFisherMatrix.

#include <array>
#include <atomic>
#include <cmath>
#include <future>
#include <limits>
#include <random>
#include <thread>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/optimizer/fisher_information.hpp"
#include "operon/optimizer/least_squares.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"

namespace {

using Extents = std::dextents<std::size_t, 2>;
using Mapping = std::layout_stride::mapping<Extents>;

// y = c0 + c1 * x, Jacobian columns [1, x_i].
class LinearModelCost final : public Operon::LeastSquaresCostFunction {
public:
    LinearModelCost(std::vector<Operon::Scalar> x, std::vector<Operon::Scalar> y)
        : x_(std::move(x))
        , y_(std::move(y))
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return x_.size(); }

    [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const> parameters, std::span<Operon::Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> jacobian) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        if (parameters.size() != NumParameters()) {
            return tl::unexpected(Operon::LeastSquaresError { .Code = Operon::LeastSquaresErrorCode::InvalidShape,
                .Expected = NumParameters(),
                .Actual = parameters.size() });
        }
        if (residuals.size() != x_.size()) {
            return tl::unexpected(Operon::LeastSquaresError { .Code = Operon::LeastSquaresErrorCode::InvalidShape,
                .Expected = x_.size(),
                .Actual = residuals.size() });
        }
        for (std::size_t i = 0; i < x_.size(); ++i) {
            residuals[i] = static_cast<Operon::Scalar>(parameters[0] + (parameters[1] * x_[i]) - y_[i]);
        }
        if (jacobian) {
            if (jacobian->extent(0) != x_.size() || jacobian->extent(1) != NumParameters()) {
                return tl::unexpected(Operon::LeastSquaresError { .Code = Operon::LeastSquaresErrorCode::InvalidShape,
                    .Expected = x_.size(),
                    .Actual = jacobian->extent(0),
                    .Row = x_.size(),
                    .Column = NumParameters() });
            }
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

// Regression guard: a numerical-only cost with no likelihood/Fisher method
// must satisfy Concepts::LeastSquaresCost.
static_assert(Operon::Concepts::LeastSquaresCost<LinearModelCost>);

auto MakeLinearFixture(std::size_t n, Operon::Scalar c0, Operon::Scalar c1) -> LinearModelCost
{
    std::vector<Operon::Scalar> x(n);
    std::vector<Operon::Scalar> y(n);
    std::mt19937 rng { 42 }; // NOLINT
    std::uniform_real_distribution<double> dist(-2.0, 2.0);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<Operon::Scalar>(dist(rng));
        y[i] = static_cast<Operon::Scalar>(c0 + (c1 * x[i]));
    }
    return LinearModelCost { std::move(x), std::move(y) };
}

} // namespace

TEST_CASE("LeastSquaresCostFunction: NumParameters/NumResiduals report exact dimensions", "[least-squares]")
{
    auto cost = MakeLinearFixture(11, Operon::Scalar { 0 }, Operon::Scalar { 1 });
    CHECK(cost.NumParameters() == 2);
    CHECK(cost.NumResiduals() == 11);
}

TEST_CASE("LeastSquaresError: Cause carries a typed interpreter error", "[least-squares]")
{
    Operon::LeastSquaresError error { .Code = Operon::LeastSquaresErrorCode::NonFiniteEvaluation,
        .Cause = Operon::InterpreterError { .Kind = Operon::InterpreterError::Code::MissingVariable, .Hash = 42 } };
    REQUIRE(error.Cause.has_value());
    CHECK(error.Cause->Kind == Operon::InterpreterError::Code::MissingVariable);
    CHECK(error.Cause->Hash == 42);

    Operon::LeastSquaresError noCause { .Code = Operon::LeastSquaresErrorCode::InvalidShape };
    CHECK_FALSE(noCause.Cause.has_value());
}

TEST_CASE("LeastSquaresCostFunction: residual-only evaluation omits Jacobian", "[least-squares]")
{
    auto cost = MakeLinearFixture(8, Operon::Scalar { 1 }, Operon::Scalar { -2 });
    std::array<Operon::Scalar, 2> params { 1, -2 };
    std::vector<Operon::Scalar> residuals(cost.NumResiduals());

    auto result = cost.Evaluate(params, residuals, std::nullopt);
    REQUIRE(result.has_value());
    for (auto r : residuals) {
        CHECK_THAT(static_cast<double>(r), Catch::Matchers::WithinAbs(0.0, 1e-5));
    }
}

TEST_CASE("LeastSquaresCostFunction: analytic Jacobian matches central finite differences", "[least-squares]")
{
    auto cost = MakeLinearFixture(12, Operon::Scalar { 0.3 }, Operon::Scalar { 1.7 });
    std::array<Operon::Scalar, 2> params { 0.1, 0.2 }; // away from the true optimum on purpose
    auto const n = cost.NumResiduals();
    auto const p = cost.NumParameters();

    std::vector<Operon::Scalar> residuals(n);
    std::vector<Operon::Scalar> jacBuffer(n * p);
    Operon::ScalarMatrixView jac { jacBuffer.data(),
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };
    REQUIRE(cost.Evaluate(params, residuals, jac).has_value());

    constexpr double eps = 1e-3;
    for (std::size_t j = 0; j < p; ++j) {
        auto plus = params;
        auto minus = params;
        plus.at(j) = static_cast<Operon::Scalar>(plus.at(j) + eps);
        minus.at(j) = static_cast<Operon::Scalar>(minus.at(j) - eps);
        std::vector<Operon::Scalar> rPlus(n);
        std::vector<Operon::Scalar> rMinus(n);
        REQUIRE(cost.Evaluate(plus, rPlus, std::nullopt).has_value());
        REQUIRE(cost.Evaluate(minus, rMinus, std::nullopt).has_value());
        for (std::size_t i = 0; i < n; ++i) {
            auto const fd = (static_cast<double>(rPlus[i]) - static_cast<double>(rMinus[i])) / (2.0 * eps);
            CHECK_THAT(fd, Catch::Matchers::WithinAbs(static_cast<double>(Operon::At(jac, i, j)), 1e-3));
        }
    }
}

TEST_CASE("LeastSquaresCostFunction: invalid parameter/residual/Jacobian shapes are rejected", "[least-squares]")
{
    auto cost = MakeLinearFixture(5, Operon::Scalar { 0 }, Operon::Scalar { 1 });

    SECTION("wrong parameter count")
    {
        std::array<Operon::Scalar, 3> params { 0, 0, 0 };
        std::vector<Operon::Scalar> residuals(cost.NumResiduals());
        auto result = cost.Evaluate(params, residuals, std::nullopt);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidShape);
    }

    SECTION("wrong residual count")
    {
        std::array<Operon::Scalar, 2> params { 0, 1 };
        std::vector<Operon::Scalar> residuals(cost.NumResiduals() + 1);
        auto result = cost.Evaluate(params, residuals, std::nullopt);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidShape);
    }

    SECTION("wrong Jacobian row count")
    {
        std::array<Operon::Scalar, 2> params { 0, 1 };
        std::vector<Operon::Scalar> residuals(cost.NumResiduals());
        std::vector<Operon::Scalar> jacBuffer((cost.NumResiduals() + 1) * 2);
        Operon::ScalarMatrixView jac { jacBuffer.data(),
            Mapping { Extents { cost.NumResiduals() + 1, 2 }, std::array<std::size_t, 2> { 2, 1 } } };
        auto result = cost.Evaluate(params, residuals, jac);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidShape);
    }
}

TEST_CASE(
    "LeastSquaresCostFunction: concurrent independent evaluations agree with a serial reference", "[least-squares]")
{
    constexpr int kThreads = 8;
    constexpr std::size_t kRows = 64;
    std::array<Operon::Scalar, 2> params { 0.5, -1.25 };

    auto reference = MakeLinearFixture(kRows, Operon::Scalar { 0.5 }, Operon::Scalar { -1.25 });
    std::vector<Operon::Scalar> expected(kRows);
    REQUIRE(reference.Evaluate(params, expected, std::nullopt).has_value());

    std::vector<std::future<bool>> futures;
    futures.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        futures.push_back(std::async(std::launch::async, [&]() -> bool {
            auto local = MakeLinearFixture(kRows, Operon::Scalar { 0.5 }, Operon::Scalar { -1.25 });
            std::vector<Operon::Scalar> residuals(kRows);
            auto result = local.Evaluate(params, residuals, std::nullopt);
            if (!result.has_value()) {
                return false;
            }
            for (std::size_t i = 0; i < kRows; ++i) {
                if (residuals[i] != expected[i]) {
                    return false;
                }
            }
            return true;
        }));
    }
    for (auto& f : futures) {
        CHECK(f.get());
    }
}

TEST_CASE("ComputeDiagnostics: unweighted, uniform, and per-row weights", "[least-squares][diagnostics]")
{
    std::vector<Operon::Scalar> residuals { 1, -2, 3, -4 };

    SECTION("unweighted cost is 0.5 * sum(r^2)")
    {
        auto result = Operon::ComputeDiagnostics(residuals, std::nullopt, {});
        REQUIRE(result.has_value());
        CHECK_THAT(result->Cost, Catch::Matchers::WithinAbs(0.5 * (1 + 4 + 9 + 16), 1e-9));
        CHECK_THAT(result->ResidualNorm, Catch::Matchers::WithinAbs(std::sqrt(1.0 + 4 + 9 + 16), 1e-9));
        CHECK(std::isnan(result->GradientNorm));
    }

    SECTION("uniform weight scales cost by w")
    {
        std::vector<Operon::Scalar> weights { 2 };
        auto unweighted = Operon::ComputeDiagnostics(residuals, std::nullopt, {});
        auto weighted = Operon::ComputeDiagnostics(residuals, std::nullopt, weights);
        REQUIRE(unweighted.has_value());
        REQUIRE(weighted.has_value());
        CHECK_THAT(weighted->Cost, Catch::Matchers::WithinAbs(2.0 * unweighted->Cost, 1e-9));
    }

    SECTION("per-row weight matches manual accumulation")
    {
        std::vector<Operon::Scalar> weights { 1, 2, 0.5, 3 };
        auto result = Operon::ComputeDiagnostics(residuals, std::nullopt, weights);
        REQUIRE(result.has_value());
        double expected = 0.5 * (1.0 * 1 + 2.0 * 4 + 0.5 * 9 + 3.0 * 16);
        CHECK_THAT(result->Cost, Catch::Matchers::WithinAbs(expected, 1e-9));
    }

    SECTION("uniform per-row weight matches equivalent scalar weight")
    {
        std::vector<Operon::Scalar> uniformScalar { 3 };
        std::vector<Operon::Scalar> uniformPerRow(residuals.size(), Operon::Scalar { 3 });
        auto a = Operon::ComputeDiagnostics(residuals, std::nullopt, uniformScalar);
        auto b = Operon::ComputeDiagnostics(residuals, std::nullopt, uniformPerRow);
        REQUIRE(a.has_value());
        REQUIRE(b.has_value());
        CHECK_THAT(a->Cost, Catch::Matchers::WithinAbs(b->Cost, 1e-9));
    }
}

TEST_CASE(
    "ComputeDiagnostics: residual+Jacobian evaluation reports a finite gradient norm", "[least-squares][diagnostics]")
{
    auto cost = MakeLinearFixture(10, Operon::Scalar { 0.2 }, Operon::Scalar { -0.7 });
    std::array<Operon::Scalar, 2> params { 0.0, 0.0 }; // away from optimum -> nonzero gradient
    auto const n = cost.NumResiduals();
    auto const p = cost.NumParameters();
    std::vector<Operon::Scalar> residuals(n);
    std::vector<Operon::Scalar> jacBuffer(n * p);
    Operon::ScalarMatrixView jac { jacBuffer.data(),
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };
    REQUIRE(cost.Evaluate(params, residuals, jac).has_value());

    auto result = Operon::ComputeDiagnostics(residuals, jac, {});
    REQUIRE(result.has_value());
    CHECK_FALSE(std::isnan(result->GradientNorm));
    CHECK(result->GradientNorm > 0.0);
}

TEST_CASE("ComputeDiagnostics: shape and finiteness errors", "[least-squares][diagnostics]")
{
    std::vector<Operon::Scalar> residuals { 1, 2, 3 };

    SECTION("weights size mismatch")
    {
        std::vector<Operon::Scalar> weights { 1, 2 };
        auto result = Operon::ComputeDiagnostics(residuals, std::nullopt, weights);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidWeights);
        CHECK(result.error().Expected == residuals.size());
        CHECK(result.error().Actual == weights.size());
    }

    SECTION("non-finite residual")
    {
        std::vector<Operon::Scalar> bad { 1, std::numeric_limits<Operon::Scalar>::quiet_NaN(), 3 };
        auto result = Operon::ComputeDiagnostics(bad, std::nullopt, {});
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
    }

    SECTION("negative weight")
    {
        std::vector<Operon::Scalar> weights { 1, -1, 1 };
        auto result = Operon::ComputeDiagnostics(residuals, std::nullopt, weights);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidWeights);
        CHECK(result.error().Row == 1);
    }
}

TEST_CASE("ComputeGradient: unweighted, scalar, and per-row weighted reductions", "[least-squares][gradient]")
{
    // r = [1, -2, 3], J = [[1,0],[0,1],[1,1]] -> unweighted grad = J^T r = [1+3, -2+3] = [4, 1]
    std::vector<Operon::Scalar> residuals { 1, -2, 3 };
    std::array<Operon::Scalar, 6> jacBuffer { 1, 0, 0, 1, 1, 1 };
    Operon::ConstScalarMatrixView jac { jacBuffer.data(),
        Mapping { Extents { 3, 2 }, std::array<std::size_t, 2> { 2, 1 } } };

    SECTION("unweighted")
    {
        std::array<Operon::Scalar, 2> gradient {};
        auto cost = Operon::ComputeGradient(residuals, jac, gradient, {});
        REQUIRE(cost.has_value());
        CHECK_THAT(*cost, Catch::Matchers::WithinAbs(0.5 * (1.0 + 4.0 + 9.0), 1e-9));
        CHECK_THAT(static_cast<double>(gradient[0]), Catch::Matchers::WithinAbs(4.0, 1e-6));
        CHECK_THAT(static_cast<double>(gradient[1]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    }

    SECTION("scalar weight scales cost and gradient")
    {
        std::vector<Operon::Scalar> weight { 2 };
        std::array<Operon::Scalar, 2> gradient {};
        auto cost = Operon::ComputeGradient(residuals, jac, gradient, weight);
        REQUIRE(cost.has_value());
        CHECK_THAT(*cost, Catch::Matchers::WithinAbs(2.0 * 0.5 * (1.0 + 4.0 + 9.0), 1e-9));
        CHECK_THAT(static_cast<double>(gradient[0]), Catch::Matchers::WithinAbs(8.0, 1e-6));
        CHECK_THAT(static_cast<double>(gradient[1]), Catch::Matchers::WithinAbs(2.0, 1e-6));
    }

    SECTION("per-row weight matches manual accumulation")
    {
        std::vector<Operon::Scalar> weights { 1, 2, 0.5 };
        std::array<Operon::Scalar, 2> gradient {};
        auto cost = Operon::ComputeGradient(residuals, jac, gradient, weights);
        REQUIRE(cost.has_value());
        // grad = sum(w_i * r_i * J_i) = 1*1*[1,0] + 2*-2*[0,1] + 0.5*3*[1,1] = [1+1.5, -4+1.5] = [2.5, -2.5]
        CHECK_THAT(static_cast<double>(gradient[0]), Catch::Matchers::WithinAbs(2.5, 1e-6));
        CHECK_THAT(static_cast<double>(gradient[1]), Catch::Matchers::WithinAbs(-2.5, 1e-6));
    }
}

TEST_CASE("ComputeGradient: invalid shapes are rejected", "[least-squares][gradient]")
{
    std::vector<Operon::Scalar> residuals { 1, 2, 3 };
    std::array<Operon::Scalar, 6> jacBuffer {};
    Operon::ConstScalarMatrixView jac { jacBuffer.data(),
        Mapping { Extents { 3, 2 }, std::array<std::size_t, 2> { 2, 1 } } };

    SECTION("jacobian row count mismatch")
    {
        std::array<Operon::Scalar, 4> smallJacBuffer {};
        Operon::ConstScalarMatrixView badJac { smallJacBuffer.data(),
            Mapping { Extents { 2, 2 }, std::array<std::size_t, 2> { 2, 1 } } };
        std::array<Operon::Scalar, 2> gradient {};
        auto result = Operon::ComputeGradient(residuals, badJac, gradient, {});
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidShape);
    }

    SECTION("gradient size mismatch")
    {
        std::array<Operon::Scalar, 3> gradient {};
        auto result = Operon::ComputeGradient(residuals, jac, gradient, {});
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidShape);
    }

    SECTION("weights size mismatch")
    {
        std::vector<Operon::Scalar> weights { 1, 2 };
        std::array<Operon::Scalar, 2> gradient {};
        auto result = Operon::ComputeGradient(residuals, jac, gradient, weights);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidWeights);
    }
}

TEST_CASE("ComputeGradient: non-finite residual, Jacobian, or weight is rejected", "[least-squares][gradient]")
{
    std::array<Operon::Scalar, 2> gradient {};

    SECTION("non-finite residual")
    {
        std::vector<Operon::Scalar> residuals { 1, std::numeric_limits<Operon::Scalar>::quiet_NaN() };
        std::array<Operon::Scalar, 4> jacBuffer { 1, 0, 0, 1 };
        Operon::ConstScalarMatrixView jac { jacBuffer.data(),
            Mapping { Extents { 2, 2 }, std::array<std::size_t, 2> { 2, 1 } } };
        auto result = Operon::ComputeGradient(residuals, jac, gradient, {});
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
    }

    SECTION("negative weight")
    {
        std::vector<Operon::Scalar> residuals { 1, 2 };
        std::vector<Operon::Scalar> weights { 1, -1 };
        std::array<Operon::Scalar, 4> jacBuffer { 1, 0, 0, 1 };
        Operon::ConstScalarMatrixView jac { jacBuffer.data(),
            Mapping { Extents { 2, 2 }, std::array<std::size_t, 2> { 2, 1 } } };
        auto result = Operon::ComputeGradient(residuals, jac, gradient, weights);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::LeastSquaresErrorCode::InvalidWeights);
        CHECK(result.error().Row == 1);
    }
}

TEST_CASE("ValidateWeights: accepts empty, scalar broadcast, and per-row; rejects everything else with distinct codes",
    "[least-squares][weights]")
{
    constexpr std::size_t rows = 4;
    constexpr auto nan = std::numeric_limits<Operon::Scalar>::quiet_NaN();
    constexpr auto inf = std::numeric_limits<Operon::Scalar>::infinity();

    SECTION("accepted shapes")
    {
        CHECK(Operon::ValidateWeights({}, rows).has_value());
        std::vector<Operon::Scalar> scalar { 2 };
        CHECK(Operon::ValidateWeights(scalar, rows).has_value());
        std::vector<Operon::Scalar> perRow { 0, 1, 2.5, 3 }; // zero is a valid weight
        CHECK(Operon::ValidateWeights(perRow, rows).has_value());
    }

    SECTION("a size other than 0, 1, or rows is a SizeMismatch carrying both sizes")
    {
        std::vector<Operon::Scalar> wrong { 1, 2, 3 };
        auto result = Operon::ValidateWeights(wrong, rows);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::WeightErrorCode::SizeMismatch);
        CHECK(result.error().Expected == rows);
        CHECK(result.error().Actual == wrong.size());
    }

    SECTION("value violations are classified distinctly and locate the first offender")
    {
        auto const classify = [&](Operon::Scalar bad) -> Operon::WeightError {
            std::vector<Operon::Scalar> weights { 1, 1, 1, 1 };
            weights[2] = bad;
            weights[3] = nan; // a later offender must not mask the first one
            auto result = Operon::ValidateWeights(weights, rows);
            REQUIRE_FALSE(result.has_value());
            return result.error();
        };
        CHECK(classify(Operon::Scalar { -1 }).Code == Operon::WeightErrorCode::NegativeValue);
        CHECK(classify(nan).Code == Operon::WeightErrorCode::NotANumber);
        CHECK(classify(inf).Code == Operon::WeightErrorCode::Infinite);
        CHECK(classify(-inf).Code == Operon::WeightErrorCode::Infinite);
        CHECK(classify(Operon::Scalar { -1 }).Row == 2);
    }

    SECTION("a scalar weight is validated too")
    {
        std::vector<Operon::Scalar> scalar { Operon::Scalar { -3 } };
        auto result = Operon::ValidateWeights(scalar, rows);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::WeightErrorCode::NegativeValue);
        CHECK(result.error().Row == 0);
    }
}

TEST_CASE("ComputeGradient: matches finite-difference gradient of a fitted linear model", "[least-squares][gradient]")
{
    auto cost = MakeLinearFixture(15, Operon::Scalar { 0.4 }, Operon::Scalar { -1.1 });
    std::array<Operon::Scalar, 2> params { 0.05, 0.05 };
    auto const n = cost.NumResiduals();
    auto const p = cost.NumParameters();

    std::vector<Operon::Scalar> residuals(n);
    std::vector<Operon::Scalar> jacBuffer(n * p);
    Operon::ScalarMatrixView jac { jacBuffer.data(),
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };
    REQUIRE(cost.Evaluate(params, residuals, jac).has_value());

    std::array<Operon::Scalar, 2> gradient {};
    auto costValue = Operon::ComputeGradient(residuals, jac, gradient, {});
    REQUIRE(costValue.has_value());

    constexpr double eps = 1e-3;
    for (std::size_t j = 0; j < p; ++j) {
        auto plus = params;
        auto minus = params;
        plus.at(j) = static_cast<Operon::Scalar>(plus.at(j) + eps);
        minus.at(j) = static_cast<Operon::Scalar>(minus.at(j) - eps);
        std::vector<Operon::Scalar> rPlus(n);
        std::vector<Operon::Scalar> rMinus(n);
        REQUIRE(cost.Evaluate(plus, rPlus, std::nullopt).has_value());
        REQUIRE(cost.Evaluate(minus, rMinus, std::nullopt).has_value());
        auto costPlus = Operon::ComputeDiagnostics(rPlus, std::nullopt, {});
        auto costMinus = Operon::ComputeDiagnostics(rMinus, std::nullopt, {});
        REQUIRE(costPlus.has_value());
        REQUIRE(costMinus.has_value());
        auto const fd = (costPlus->Cost - costMinus->Cost) / (2.0 * eps);
        CHECK_THAT(fd, Catch::Matchers::WithinAbs(static_cast<double>(gradient.at(j)), 1e-2));
    }
}

TEST_CASE("ComputeGradient and ComputeDiagnostics report the same cost and gradient norm", "[least-squares][gradient]")
{
    auto cost = MakeLinearFixture(9, Operon::Scalar { 1.2 }, Operon::Scalar { 0.3 });
    std::array<Operon::Scalar, 2> params { 0.1, -0.2 };
    auto const n = cost.NumResiduals();
    auto const p = cost.NumParameters();
    std::vector<Operon::Scalar> residuals(n);
    std::vector<Operon::Scalar> jacBuffer(n * p);
    Operon::ScalarMatrixView jac { jacBuffer.data(),
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };
    REQUIRE(cost.Evaluate(params, residuals, jac).has_value());

    std::array<Operon::Scalar, 2> gradient {};
    auto gradResult = Operon::ComputeGradient(residuals, jac, gradient, {});
    REQUIRE(gradResult.has_value());
    auto diagResult = Operon::ComputeDiagnostics(residuals, jac, {});
    REQUIRE(diagResult.has_value());

    CHECK_THAT(diagResult->Cost, Catch::Matchers::WithinAbs(*gradResult, 1e-9));
    auto const gradNormSquared = (static_cast<double>(gradient[0]) * static_cast<double>(gradient[0]))
        + (static_cast<double>(gradient[1]) * static_cast<double>(gradient[1]));
    CHECK_THAT(diagResult->GradientNorm, Catch::Matchers::WithinAbs(std::sqrt(gradNormSquared), 1e-6));
}

namespace {

// Row-major Jacobian view with unused padding per row (left NaN, so a stride bug reads it).
auto MakePaddedJacobian(std::size_t rows, std::size_t cols, std::size_t padding, std::mt19937& rng)
    -> std::pair<std::vector<Operon::Scalar>, Operon::ConstScalarMatrixView>
{
    std::uniform_real_distribution<double> dist(-3.0, 3.0);
    std::vector<Operon::Scalar> buffer(rows * (cols + padding), std::numeric_limits<Operon::Scalar>::quiet_NaN());
    for (std::size_t i = 0; i < rows; ++i) {
        for (std::size_t j = 0; j < cols; ++j) {
            buffer[(i * (cols + padding)) + j] = static_cast<Operon::Scalar>(dist(rng));
        }
    }
    Operon::ConstScalarMatrixView view { buffer.data(),
        Mapping { Extents { rows, cols }, std::array<std::size_t, 2> { cols + padding, 1 } } };
    return { std::move(buffer), view };
}

} // namespace

TEST_CASE("ComputeFisherMatrix matches the closed form and ComputeFisherDiagonal for uniform, per-row, and no sigma",
    "[least-squares][fisher]")
{
    constexpr std::size_t n = 9;
    constexpr std::size_t p = 4;
    std::mt19937 rng { 7 }; // NOLINT
    auto [buffer, jac] = MakePaddedJacobian(n, p, 2, rng); // noncontiguous Jacobian view

    std::vector<Operon::Scalar> pred(
        n, Operon::Scalar { 0 }); // ComputeFisherDiagonal infers rows from pred.size() only

    // sigmaForDiagonal is never empty: ComputeFisherDiagonal requires an explicit sigma, so "no sigma"
    // is compared against the equivalent unit sigma.
    auto runAndCompare
        = [&](std::vector<Operon::Scalar> const& sigma, std::vector<Operon::Scalar> const& sigmaForDiagonal) -> void {
        std::vector<Operon::Scalar> fisherBuffer(p * p);
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, sigma, fisher);
        REQUIRE(result.has_value());

        // Independent closed form: F(a,b) = sum_i J(i,a) J(i,b) / sigma_i^2.
        auto const reference = [&](std::size_t a, std::size_t b) -> double {
            double sum = 0.0;
            for (std::size_t i = 0; i < n; ++i) {
                auto const s
                    = static_cast<double>(sigmaForDiagonal.size() == 1 ? sigmaForDiagonal[0] : sigmaForDiagonal[i]);
                sum += static_cast<double>(Operon::At(jac, i, a)) * static_cast<double>(Operon::At(jac, i, b))
                    / (s * s);
            }
            return sum;
        };
        std::vector<Operon::Scalar> diagonal(p);
        auto diagonalResult
            = Operon::GaussianLikelihood<Operon::Scalar>::ComputeFisherDiagonal(pred, jac, sigmaForDiagonal, diagonal);
        REQUIRE(diagonalResult.has_value());

        for (std::size_t a = 0; a < p; ++a) {
            CHECK_THAT(static_cast<double>(diagonal[a]), Catch::Matchers::WithinRel(reference(a, a), 1e-3));
            for (std::size_t b = 0; b < p; ++b) {
                CHECK_THAT(
                    static_cast<double>(Operon::At(fisher, a, b)), Catch::Matchers::WithinRel(reference(a, b), 1e-3));
            }
        }
        // Exact symmetry: both (a,b) and (b,a) are written from the same
        // accumulation, so any asymmetry is a write bug, not roundoff.
        for (std::size_t a = 0; a < p; ++a) {
            for (std::size_t b = 0; b < p; ++b) {
                CHECK(Operon::At(fisher, a, b) == Operon::At(fisher, b, a));
            }
        }
    };

    SECTION("uniform sigma")
    {
        std::vector<Operon::Scalar> sigma { 1.5 };
        runAndCompare(sigma, sigma);
    }

    SECTION("per-row sigma")
    {
        std::vector<Operon::Scalar> sigma(n);
        std::uniform_real_distribution<double> dist(0.5, 2.0);
        for (auto& s : sigma) {
            s = static_cast<Operon::Scalar>(dist(rng));
        }
        runAndCompare(sigma, sigma);
    }

    SECTION("no sigma is equivalent to unit sigma")
    {
        std::vector<Operon::Scalar> unit { 1 };
        runAndCompare({}, unit);
    }
}

TEST_CASE("ComputeFisherMatrix: row-major, column-major, and subview layouts agree", "[least-squares][fisher]")
{
    constexpr std::size_t n = 6;
    constexpr std::size_t p = 3;
    std::mt19937 rng { 13 }; // NOLINT
    std::uniform_real_distribution<double> dist(-2.0, 2.0);

    // Logical values shared by every layout variant below.
    std::array<std::array<Operon::Scalar, p>, n> values {};
    for (auto& row : values) {
        for (auto& v : row) {
            v = static_cast<Operon::Scalar>(dist(rng));
        }
    }

    std::vector<Operon::Scalar> fisherRowMajor(p * p);
    std::vector<Operon::Scalar> fisherColMajor(p * p);
    std::vector<Operon::Scalar> fisherSubview(p * p);

    // Row-major: stride {p, 1}.
    std::vector<Operon::Scalar> rowMajorBuffer(n * p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            rowMajorBuffer[(i * p) + j] = values.at(i).at(j);
        }
    }
    Operon::ConstScalarMatrixView rowMajorView { rowMajorBuffer.data(),
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };

    // Column-major: stride {1, n}.
    std::vector<Operon::Scalar> colMajorBuffer(n * p);
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            colMajorBuffer[(j * n) + i] = values.at(i).at(j);
        }
    }
    Operon::ConstScalarMatrixView colMajorView { colMajorBuffer.data(),
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { 1, n } } };

    // Subview: values sit inside a larger buffer at a nonzero row/column offset.
    constexpr std::size_t rowOffset = 2;
    constexpr std::size_t colOffset = 1;
    constexpr std::size_t outerCols = p + 3;
    std::vector<Operon::Scalar> outerBuffer(
        (n + rowOffset) * outerCols, std::numeric_limits<Operon::Scalar>::quiet_NaN());
    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < p; ++j) {
            outerBuffer[((i + rowOffset) * outerCols) + (j + colOffset)] = values.at(i).at(j);
        }
    }
    Operon::ConstScalarMatrixView subview { outerBuffer.data() + (rowOffset * outerCols) + colOffset,
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { outerCols, 1 } } };

    Operon::ScalarMatrixView rowMajorFisher { fisherRowMajor.data(),
        Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
    Operon::ScalarMatrixView colMajorFisher { fisherColMajor.data(),
        Mapping { Extents { p, p }, std::array<std::size_t, 2> { 1, p } } };
    Operon::ScalarMatrixView subviewFisher { fisherSubview.data(),
        Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };

    REQUIRE(Operon::ComputeFisherMatrix(rowMajorView, {}, rowMajorFisher).has_value());
    REQUIRE(Operon::ComputeFisherMatrix(colMajorView, {}, colMajorFisher).has_value());
    REQUIRE(Operon::ComputeFisherMatrix(subview, {}, subviewFisher).has_value());

    for (std::size_t a = 0; a < p; ++a) {
        for (std::size_t b = 0; b < p; ++b) {
            auto const reference = static_cast<double>(Operon::At(rowMajorFisher, a, b));
            CHECK_THAT(
                static_cast<double>(Operon::At(colMajorFisher, a, b)), Catch::Matchers::WithinAbs(reference, 1e-5));
            CHECK_THAT(
                static_cast<double>(Operon::At(subviewFisher, a, b)), Catch::Matchers::WithinAbs(reference, 1e-5));
        }
    }
}

TEST_CASE("ComputeFisherMatrix: zero, one, overdetermined, and rank-deficient systems", "[least-squares][fisher]")
{
    SECTION("zero parameters produces an empty matrix without error")
    {
        std::vector<Operon::Scalar> jacBuffer;
        Operon::ConstScalarMatrixView jac { jacBuffer.data(),
            Mapping { Extents { 5, 0 }, std::array<std::size_t, 2> { 0, 1 } } };
        std::vector<Operon::Scalar> fisherBuffer;
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { 0, 0 }, std::array<std::size_t, 2> { 0, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, {}, fisher);
        REQUIRE(result.has_value());
    }

    SECTION("one observation, one parameter")
    {
        std::array<Operon::Scalar, 1> jacBuffer { 2 };
        Operon::ConstScalarMatrixView jac { jacBuffer.data(),
            Mapping { Extents { 1, 1 }, std::array<std::size_t, 2> { 1, 1 } } };
        std::array<Operon::Scalar, 1> fisherBuffer {};
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { 1, 1 }, std::array<std::size_t, 2> { 1, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, {}, fisher);
        REQUIRE(result.has_value());
        CHECK_THAT(static_cast<double>(fisherBuffer[0]), Catch::Matchers::WithinAbs(4.0, 1e-9));
    }

    SECTION("overdetermined system: rows >> cols")
    {
        constexpr std::size_t n = 50;
        constexpr std::size_t p = 3;
        std::mt19937 rng { 11 }; // NOLINT
        auto [buffer, jac] = MakePaddedJacobian(n, p, 0, rng);
        std::vector<Operon::Scalar> fisherBuffer(p * p);
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, {}, fisher);
        REQUIRE(result.has_value());
        // Symmetric by construction.
        for (std::size_t a = 0; a < p; ++a) {
            for (std::size_t b = 0; b < p; ++b) {
                CHECK(Operon::At(fisher, a, b) == Operon::At(fisher, b, a));
            }
        }
    }

    SECTION("rank-deficient: duplicate column yields a singular Fisher matrix")
    {
        constexpr std::size_t n = 6;
        constexpr std::size_t p = 2;
        std::array<Operon::Scalar, n * p> jacBuffer {};
        for (std::size_t i = 0; i < n; ++i) {
            auto const v = static_cast<Operon::Scalar>(i + 1);
            jacBuffer.at((i * p) + 0) = v;
            jacBuffer.at((i * p) + 1) = v; // duplicate column -> rank-deficient Fisher matrix
        }
        Operon::ConstScalarMatrixView jac { jacBuffer.data(),
            Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };
        std::vector<Operon::Scalar> fisherBuffer(p * p);
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, {}, fisher);
        REQUIRE(result.has_value());
        auto const det = (static_cast<double>(Operon::At(fisher, 0, 0)) * static_cast<double>(Operon::At(fisher, 1, 1)))
            - (static_cast<double>(Operon::At(fisher, 0, 1)) * static_cast<double>(Operon::At(fisher, 1, 0)));
        CHECK_THAT(det, Catch::Matchers::WithinAbs(0.0, 1e-6));
    }
}

TEST_CASE("ComputeFisherMatrix: shape and sigma validation errors", "[least-squares][fisher]")
{
    constexpr std::size_t n = 4;
    constexpr std::size_t p = 2;
    std::array<Operon::Scalar, n * p> jacBuffer { 1, 0, 1, 1, 1, 2, 1, 3 };
    Operon::ConstScalarMatrixView jac { jacBuffer.data(),
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };

    SECTION("fisher output has the wrong extent")
    {
        std::array<Operon::Scalar, 4> fisherBuffer {};
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { 3, 3 }, std::array<std::size_t, 2> { 3, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, {}, fisher);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::FisherErrorCode::InvalidShape);
    }

    SECTION("sigma size mismatch")
    {
        std::vector<Operon::Scalar> sigma { 1, 2 }; // neither 1 nor n
        std::array<Operon::Scalar, p * p> fisherBuffer {};
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, sigma, fisher);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::FisherErrorCode::InvalidShape);
    }

    SECTION("non-positive sigma")
    {
        std::vector<Operon::Scalar> sigma { 0 };
        std::array<Operon::Scalar, p * p> fisherBuffer {};
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, sigma, fisher);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::FisherErrorCode::InvalidSigma);
    }

    SECTION("non-finite sigma")
    {
        std::vector<Operon::Scalar> sigma { std::numeric_limits<Operon::Scalar>::infinity() };
        std::array<Operon::Scalar, p * p> fisherBuffer {};
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
        auto result = Operon::ComputeFisherMatrix(jac, sigma, fisher);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::FisherErrorCode::InvalidSigma);
    }

    SECTION("non-finite Jacobian entry")
    {
        std::array<Operon::Scalar, n * p> badBuffer { 1, 0, 1, 1, 1, 2, 1, 3 };
        badBuffer[3] = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        Operon::ConstScalarMatrixView badJac { badBuffer.data(),
            Mapping { Extents { n, p }, std::array<std::size_t, 2> { p, 1 } } };
        std::array<Operon::Scalar, p * p> fisherBuffer {};
        Operon::ScalarMatrixView fisher { fisherBuffer.data(),
            Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
        auto result = Operon::ComputeFisherMatrix(badJac, {}, fisher);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::FisherErrorCode::NonFiniteResult);
    }
}

TEST_CASE("ComputeFisherMatrix: fixed-order deterministic replay", "[least-squares][fisher]")
{
    constexpr std::size_t n = 20;
    constexpr std::size_t p = 5;
    std::mt19937 rng { 99 }; // NOLINT
    auto [buffer, jac] = MakePaddedJacobian(n, p, 1, rng);
    std::vector<Operon::Scalar> sigma(n);
    std::uniform_real_distribution<double> dist(0.3, 1.7);
    for (auto& s : sigma) {
        s = static_cast<Operon::Scalar>(dist(rng));
    }

    std::vector<Operon::Scalar> first(p * p);
    Operon::ScalarMatrixView firstView { first.data(),
        Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
    REQUIRE(Operon::ComputeFisherMatrix(jac, sigma, firstView).has_value());

    for (int replay = 0; replay < 5; ++replay) {
        std::vector<Operon::Scalar> repeat(p * p);
        Operon::ScalarMatrixView repeatView { repeat.data(),
            Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
        REQUIRE(Operon::ComputeFisherMatrix(jac, sigma, repeatView).has_value());
        CHECK(repeat == first); // bit-identical: fixed accumulation order, no nondeterminism
    }
}

TEST_CASE("ComputeFisherMatrix: concurrent independent calls are race-free", "[least-squares][fisher]")
{
    constexpr int kThreads = 8;
    constexpr std::size_t n = 30;
    constexpr std::size_t p = 4;

    std::atomic<int> mismatches { 0 };
    std::vector<std::thread> threads;
    threads.reserve(kThreads);
    for (int t = 0; t < kThreads; ++t) {
        threads.emplace_back([&, t]() -> void {
            std::mt19937 rng { static_cast<std::uint32_t>(1000 + t) };
            auto [buffer, jac] = MakePaddedJacobian(n, p, static_cast<std::size_t>(t % 3), rng);
            std::vector<Operon::Scalar> fisherBuffer(p * p);
            Operon::ScalarMatrixView fisher { fisherBuffer.data(),
                Mapping { Extents { p, p }, std::array<std::size_t, 2> { p, 1 } } };
            auto result = Operon::ComputeFisherMatrix(jac, {}, fisher);
            if (!result.has_value()) {
                ++mismatches;
                return;
            }
            for (std::size_t a = 0; a < p; ++a) {
                for (std::size_t b = 0; b < p; ++b) {
                    if (Operon::At(fisher, a, b) != Operon::At(fisher, b, a)) {
                        ++mismatches;
                    }
                }
            }
        });
    }
    for (auto& th : threads) {
        th.join();
    }
    CHECK(mismatches.load() == 0);
}
