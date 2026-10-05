// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

// Deliberately includes only the public entry-point header: the file must
// compile and run using nothing from a detail:: namespace.
#include "operon/optimizer/least_squares_fit.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <optional>
#include <span>
#include <type_traits>
#include <utility>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

namespace {

using Operon::Scalar;

constexpr std::array<Operon::OptimizerType, 2> BACKENDS { Operon::OptimizerType::Tiny, Operon::OptimizerType::Eigen };

// y = c0 + c1 * x with Jacobian columns [1, x_i]. Records how it is driven and
// can be told to fail from a given call onward or to emit a NaN residual.
class AffineCost final : public Operon::LeastSquaresCostFunction {
public:
    AffineCost(std::vector<Scalar> x, std::vector<Scalar> y) : x_(std::move(x)), y_(std::move(y)) {}

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return x_.size(); }

    [[nodiscard]] auto Evaluate(std::span<Scalar const> parameters, std::span<Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> jacobian) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        ++calls;
        if (jacobian) {
            ++jacobianCalls;
        }
        if (failFromCall != 0 && calls >= failFromCall) {
            return tl::unexpected(Operon::LeastSquaresError {
                .Code = Operon::LeastSquaresErrorCode::NumericalFailure, .Row = 7, .Column = 1 });
        }
        for (std::size_t i = 0; i < x_.size(); ++i) {
            residuals[i] = static_cast<Scalar>(parameters[0] + (parameters[1] * x_[i]) - y_[i]);
        }
        if (nanRow) {
            residuals[*nanRow] = std::numeric_limits<Scalar>::quiet_NaN();
        }
        if (jacobian) {
            for (std::size_t i = 0; i < x_.size(); ++i) {
                Operon::At(*jacobian, i, 0) = Scalar { 1 };
                Operon::At(*jacobian, i, 1) = x_[i];
            }
        }
        return {};
    }

    mutable std::size_t calls { 0 };
    mutable std::size_t jacobianCalls { 0 };
    std::size_t failFromCall { 0 }; // 1-based; 0 = never fail
    std::optional<std::size_t> nanRow;

private:
    std::vector<Scalar> x_;
    std::vector<Scalar> y_;
};

// n symmetric sample points on the line c0 + c1 * x.
auto MakeAffine(std::size_t n, Scalar c0, Scalar c1) -> AffineCost
{
    std::vector<Scalar> x(n);
    std::vector<Scalar> y(n);
    for (std::size_t i = 0; i < n; ++i) {
        x[i] = static_cast<Scalar>(i) - (static_cast<Scalar>(n) / Scalar { 2 });
        y[i] = c0 + (c1 * x[i]);
    }
    return AffineCost { std::move(x), std::move(y) };
}

// r_i = c_i, no parameters.
class ConstantResidualCost final : public Operon::LeastSquaresCostFunction {
public:
    explicit ConstantResidualCost(std::vector<Scalar> values, bool fail = false)
        : values_(std::move(values))
        , fail_(fail)
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 0; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return values_.size(); }

    [[nodiscard]] auto Evaluate(std::span<Scalar const> /*parameters*/, std::span<Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> jacobian) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        ++calls;
        if (jacobian) {
            ++jacobianCalls;
        }
        if (fail_) {
            return tl::unexpected(
                Operon::LeastSquaresError { .Code = Operon::LeastSquaresErrorCode::EvaluationFailure, .Row = 2 });
        }
        std::ranges::copy(values_, residuals.begin());
        return {};
    }

    mutable std::size_t calls { 0 };
    mutable std::size_t jacobianCalls { 0 };

private:
    std::vector<Scalar> values_;
    bool fail_;
};

// r_i = a * exp(b * x_i) - y_i on x in [0, 2], with y generated from (aTrue, bTrue).
class ExponentialCost final : public Operon::LeastSquaresCostFunction {
public:
    ExponentialCost(std::size_t n, Scalar aTrue, Scalar bTrue) : x_(n), y_(n)
    {
        for (std::size_t i = 0; i < n; ++i) {
            x_[i] = Scalar { 2 } * static_cast<Scalar>(i) / static_cast<Scalar>(n - 1);
            y_[i] = aTrue * std::exp(bTrue * x_[i]);
        }
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return x_.size(); }

    [[nodiscard]] auto Evaluate(std::span<Scalar const> parameters, std::span<Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> jacobian) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        for (std::size_t i = 0; i < x_.size(); ++i) {
            auto const e = std::exp(parameters[1] * x_[i]);
            residuals[i] = (parameters[0] * e) - y_[i];
            if (jacobian) {
                Operon::At(*jacobian, i, 0) = e;
                Operon::At(*jacobian, i, 1) = parameters[0] * x_[i] * e;
            }
        }
        return {};
    }

private:
    std::vector<Scalar> x_;
    std::vector<Scalar> y_;
};

auto Near(Scalar actual, Scalar expected, double tol) -> void
{
    CHECK_THAT(static_cast<double>(actual), Catch::Matchers::WithinAbs(static_cast<double>(expected), tol));
}

} // namespace

// The public surface is expressible with public types alone.
static_assert(std::is_same_v<decltype(Operon::FitLeastSquares(std::declval<Operon::LeastSquaresCostFunction const&>(),
                                 std::declval<Operon::ConstScalarSpan>(), Operon::LeastSquaresFitOptions {})),
    Operon::FitOutcome>);
static_assert(std::is_aggregate_v<Operon::LeastSquaresFitOptions>);
static_assert(Operon::LeastSquaresFitOptions {}.Backend == Operon::OptimizerType::Tiny);
static_assert(Operon::LeastSquaresFitOptions {}.Iterations == 100);
static_assert(!Operon::LeastSquaresFitOptions {}.RecoverNonFinite);

TEST_CASE("FitLeastSquares fits an affine cost with Tiny and Eigen", "[least-squares][fit]")
{
    auto const c0 = Scalar { 2.5 };
    auto const c1 = Scalar { -1.3 };
    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));
        auto cost = MakeAffine(20, c0, c1);
        std::array<Scalar, 2> const start { 0, 0 };
        auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend });

        REQUIRE(outcome.has_value());
        REQUIRE(outcome->FinalParameters.size() == 2);
        Near(outcome->FinalParameters[0], c0, 1e-3);
        Near(outcome->FinalParameters[1], c1, 1e-3);
        CHECK(outcome->InitialParameters == std::vector<Scalar> { 0, 0 });
        CHECK(outcome->FinalCost < outcome->InitialCost);
        CHECK(outcome->Iterations > 0);
        CHECK(outcome->FunctionEvaluations > 0);
        CHECK(outcome->JacobianEvaluations > 0);
        CHECK(Operon::EvaluationError(outcome) == nullptr);
        CHECK(Operon::ConfigurationError(outcome) == nullptr);
    }
}

TEST_CASE("FitLeastSquares defaults to Tiny", "[least-squares][fit]")
{
    auto costDefault = MakeAffine(20, Scalar { 0.4 }, Scalar { 1.7 });
    auto costTiny = MakeAffine(20, Scalar { 0.4 }, Scalar { 1.7 });
    std::array<Scalar, 2> const start { 0, 0 };

    auto byDefault = Operon::FitLeastSquares(costDefault, start);
    auto explicitTiny = Operon::FitLeastSquares(costTiny, start, { .Backend = Operon::OptimizerType::Tiny });

    REQUIRE(byDefault.has_value());
    REQUIRE(explicitTiny.has_value());
    CHECK(byDefault->FinalParameters == explicitTiny->FinalParameters);
    CHECK(byDefault->Iterations == explicitTiny->Iterations);
    CHECK(byDefault->FunctionEvaluations == explicitTiny->FunctionEvaluations);
    CHECK(costDefault.calls == costTiny.calls);
}

TEST_CASE("FitLeastSquares applies per-row weights and a uniform weight scales the objective", "[least-squares][fit]")
{
    auto const c0 = Scalar { 1.0 };
    auto const c1 = Scalar { 0.5 };
    constexpr std::size_t n = 20;
    constexpr std::size_t outlier = 0;

    auto makeContaminated = [c0, c1]() -> AffineCost {
        std::vector<Scalar> x(n);
        std::vector<Scalar> y(n);
        for (std::size_t i = 0; i < n; ++i) {
            x[i] = static_cast<Scalar>(i) - (static_cast<Scalar>(n) / Scalar { 2 });
            y[i] = c0 + (c1 * x[i]) + (i == outlier ? Scalar { 5 } : Scalar { 0 });
        }
        return AffineCost { std::move(x), std::move(y) };
    };
    std::array<Scalar, 2> const start { 0, 0 };

    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));

        auto unweightedCost = makeContaminated();
        auto unweighted = Operon::FitLeastSquares(unweightedCost, start, { .Backend = backend });
        REQUIRE(unweighted.has_value());
        // The outlier drags the unweighted slope by about 5 * (x_0 - mean(x)) / Sxx = -0.07.
        CHECK(std::abs(unweighted->FinalParameters[1] - c1) > Scalar { 0.02 });

        // Zero weight on the outlier row recovers the clean line exactly.
        std::vector<Scalar> weights(n, Scalar { 1 });
        weights[outlier] = Scalar { 0 };
        auto weightedCost = makeContaminated();
        auto weighted = Operon::FitLeastSquares(weightedCost, start, { .Backend = backend, .Weights = weights });
        REQUIRE(weighted.has_value());
        Near(weighted->FinalParameters[0], c0, 1e-3);
        Near(weighted->FinalParameters[1], c1, 1e-3);

        // A single weight multiplies every row: the optimum is unchanged and the
        // cost follows the 0.5 * sum(w * r^2) convention.
        std::array<Scalar, 1> const uniform { Scalar { 4 } };
        auto uniformCost = makeContaminated();
        auto scaled = Operon::FitLeastSquares(uniformCost, start, { .Backend = backend, .Weights = uniform });
        REQUIRE(scaled.has_value());
        CHECK_THAT(static_cast<double>(scaled->InitialCost),
            Catch::Matchers::WithinRel(4.0 * static_cast<double>(unweighted->InitialCost), 1e-3));
        Near(scaled->FinalParameters[0], unweighted->FinalParameters[0], 5e-3);
        Near(scaled->FinalParameters[1], unweighted->FinalParameters[1], 5e-3);
    }
}

TEST_CASE(
    "FitLeastSquares reports invalid weights as a typed configuration error without evaluating", "[least-squares][fit]")
{
    std::array<Scalar, 2> const start { 0.25, -0.5 };
    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));

        auto const check
            = [&](std::vector<Scalar> const& weights, Operon::WeightErrorCode code, std::size_t row) -> void {
            auto cost = MakeAffine(6, Scalar { 1 }, Scalar { 1 });
            auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend, .Weights = weights });

            REQUIRE_FALSE(outcome.has_value());
            auto const* error = Operon::ConfigurationError(outcome);
            REQUIRE(error != nullptr);
            CHECK(Operon::EvaluationError(outcome) == nullptr);
            CHECK(error->Error.Code == code);
            CHECK(error->Error.Row == row);
            CHECK(error->Error.Expected == 6);
            CHECK(error->Error.Actual == weights.size());
            auto const& diag = Operon::Diagnostics(outcome);
            CHECK(diag.InitialParameters == std::vector<Scalar>(start.begin(), start.end()));
            CHECK(diag.FinalParameters == diag.InitialParameters);
            CHECK(diag.FunctionEvaluations == 0);
            CHECK(diag.JacobianEvaluations == 0);
            CHECK(cost.calls == 0);
        };

        check({ 1, 1, -1, 1, 1, 1 }, Operon::WeightErrorCode::NegativeValue, 2);
        check({ 1, 1, 1, std::numeric_limits<Scalar>::quiet_NaN(), 1, 1 }, Operon::WeightErrorCode::NotANumber, 3);
        check({ 1, 1, 1, 1, std::numeric_limits<Scalar>::infinity(), 1 }, Operon::WeightErrorCode::Infinite, 4);
        check({ 1, 1, 1 }, Operon::WeightErrorCode::SizeMismatch, 0);
    }
}

TEST_CASE("FitLeastSquares rejects a parameter-count mismatch without evaluating", "[least-squares][fit]")
{
    auto cost = MakeAffine(6, Scalar { 1 }, Scalar { 1 });
    std::array<Scalar, 3> const start { 0, 0, 0 };
    auto outcome = Operon::FitLeastSquares(cost, start);

    REQUIRE_FALSE(outcome.has_value());
    auto const* error = Operon::EvaluationError(outcome);
    REQUIRE(error != nullptr);
    CHECK(error->Error.Code == Operon::GradientErrorCode::InvalidShape);
    CHECK(error->Error.Expected == 2);
    CHECK(error->Error.Actual == 3);
    CHECK(error->FinalParameters == error->InitialParameters);
    CHECK(error->InitialParameters.size() == 3);
    CHECK(std::isnan(static_cast<double>(error->InitialCost)));
    CHECK(std::isnan(static_cast<double>(error->FinalCost)));
    CHECK(error->FunctionEvaluations == 0);
    CHECK(error->JacobianEvaluations == 0);
    CHECK(cost.calls == 0);
}

TEST_CASE("FitLeastSquares reports consistent diagnostics for a parameterless cost", "[least-squares][fit]")
{
    std::vector<Scalar> const values { 1, -2, 3, 0.5 };
    std::array<Scalar, 4> const weights { 1, 2, 0, 4 };
    // 0.5 * (1*1 + 2*4 + 0*9 + 4*0.25)
    constexpr double unweightedHalfSumSquares = 0.5 * (1.0 + 4.0 + 9.0 + 0.25);
    constexpr double weightedHalfSumSquares = 0.5 * (1.0 + 8.0 + 0.0 + 1.0);

    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));
        for (bool weighted : { false, true }) {
            CAPTURE(weighted);
            ConstantResidualCost cost { values };
            Operon::LeastSquaresFitOptions options { .Backend = backend };
            if (weighted) {
                options.Weights = weights;
            }
            auto outcome = Operon::FitLeastSquares(cost, {}, options);

            // Nothing to improve: a FitFailure (not an error) with identical costs.
            REQUIRE_FALSE(outcome.has_value());
            CHECK(Operon::EvaluationError(outcome) == nullptr);
            CHECK(Operon::ConfigurationError(outcome) == nullptr);
            auto const& diag = Operon::Diagnostics(outcome);
            auto const expected = weighted ? weightedHalfSumSquares : unweightedHalfSumSquares;
            CHECK_THAT(static_cast<double>(diag.InitialCost), Catch::Matchers::WithinAbs(expected, 1e-6));
            CHECK(diag.FinalCost == diag.InitialCost);
            CHECK(diag.Iterations == 0);
            CHECK(diag.FunctionEvaluations == 1);
            CHECK(diag.JacobianEvaluations == 0);
            CHECK(diag.InitialParameters.empty());
            CHECK(diag.FinalParameters.empty());
            CHECK(cost.calls == 1);
            CHECK(cost.jacobianCalls == 0);
        }

        ConstantResidualCost failing { values, true };
        auto outcome = Operon::FitLeastSquares(failing, {}, { .Backend = backend });
        REQUIRE_FALSE(outcome.has_value());
        auto const* error = Operon::EvaluationError(outcome);
        REQUIRE(error != nullptr);
        CHECK(error->Error.Code == Operon::GradientErrorCode::EvaluationFailure);
        CHECK(error->Error.Row == 2);
        CHECK(std::isnan(static_cast<double>(error->InitialCost)));
        CHECK(std::isnan(static_cast<double>(error->FinalCost)));
        CHECK(error->Iterations == 0);
        CHECK(error->FunctionEvaluations == 1);
        CHECK(error->JacobianEvaluations == 0);
    }
}

namespace {
// Reports a residual count no solver backend can represent. The driver must
// reject this before evaluating the cost or allocating a residual buffer.
class OversizedCost final : public Operon::LeastSquaresCostFunction {
public:
    OversizedCost(std::size_t residuals, std::size_t parameters) : residuals_(residuals), parameters_(parameters) {}

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return parameters_; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return residuals_; }

    [[nodiscard]] auto Evaluate(std::span<Scalar const> /*parameters*/, std::span<Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> /*jacobian*/) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        ++calls;
        if (residuals.size() != residuals_) {
            return tl::unexpected(Operon::LeastSquaresError { .Code = Operon::LeastSquaresErrorCode::InvalidShape,
                .Expected = residuals_,
                .Actual = residuals.size() });
        }
        return {};
    }

    mutable std::size_t calls { 0 };

private:
    std::size_t residuals_;
    std::size_t parameters_;
};
} // namespace

TEST_CASE("FitLeastSquares rejects an oversized residual count before allocating or narrowing", "[least-squares][fit]")
{
    constexpr auto limit = static_cast<std::size_t>(std::numeric_limits<int>::max());
    constexpr std::array<std::size_t, 3> sizes { limit + 1, std::size_t { 1 } << 40,
        std::numeric_limits<std::size_t>::max() };

    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));
        for (auto const rows : sizes) {
            CAPTURE(rows);
            for (std::size_t const parameters : { std::size_t { 0 }, std::size_t { 2 } }) {
                CAPTURE(parameters);
                std::vector<Scalar> const start(parameters, Scalar { 0.25 });
                auto const checkUnevaluated = [&](Operon::FitOutcome const& outcome) {
                    REQUIRE_FALSE(outcome.has_value());
                    CHECK(Operon::ConfigurationError(outcome) == nullptr);
                    auto const* error = Operon::EvaluationError(outcome);
                    REQUIRE(error != nullptr);
                    CHECK(error->FinalParameters == error->InitialParameters);
                    CHECK(error->InitialParameters == start);
                    CHECK(std::isnan(static_cast<double>(error->InitialCost)));
                    CHECK(std::isnan(static_cast<double>(error->FinalCost)));
                    CHECK(error->Iterations == 0);
                    CHECK(error->FunctionEvaluations == 0);
                    CHECK(error->JacobianEvaluations == 0);
                };

                // Oversized costs are rejected without calling user code.
                {
                    OversizedCost cost { rows, parameters };
                    auto const outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend });
                    checkUnevaluated(outcome);
                    auto const* error = Operon::EvaluationError(outcome);
                    CHECK(error->Error.Code == Operon::GradientErrorCode::InvalidShape);
                    CHECK(error->Error.Expected == limit);
                    CHECK(error->Error.Actual == rows);
                    CHECK(cost.calls == 0);
                }
            }
        }
    }
}

TEST_CASE("FitLeastSquares propagates a cost error with its location and driver counters", "[least-squares][fit]")
{
    std::array<Scalar, 2> const start { 0, 0 };
    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));
        auto cost = MakeAffine(12, Scalar { 3 }, Scalar { -2 });
        cost.failFromCall = 3; // the initial point and the first step evaluate fine
        auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend });

        REQUIRE_FALSE(outcome.has_value());
        CHECK(Operon::ConfigurationError(outcome) == nullptr);
        auto const* error = Operon::EvaluationError(outcome);
        REQUIRE(error != nullptr);
        CHECK(error->Error.Code == Operon::GradientErrorCode::NumericalFailure);
        CHECK(error->Error.Row == 7);
        CHECK(error->Error.Column == 1);
        CHECK(error->InitialParameters == std::vector<Scalar> { 0, 0 });
        CHECK(error->FinalParameters.size() == 2);
        REQUIRE(cost.calls >= 3);
        CHECK(static_cast<std::size_t>(error->JacobianEvaluations) == cost.jacobianCalls);
        if (backend == Operon::OptimizerType::Tiny) {
            // Tiny asks for residuals on every call and the Jacobian on some.
            CHECK(static_cast<std::size_t>(error->FunctionEvaluations) == cost.calls);
        } else {
            // Eigen requests residuals and Jacobians in separate calls.
            CHECK(static_cast<std::size_t>(error->FunctionEvaluations + error->JacobianEvaluations) == cost.calls);
        }
    }
}

TEST_CASE("FitLeastSquares treats a nonfinite output as a typed error by default", "[least-squares][fit]")
{
    std::array<Scalar, 2> const start { 0, 0 };
    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));
        auto cost = MakeAffine(8, Scalar { 1 }, Scalar { 1 });
        cost.nanRow = 4;
        auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend });

        REQUIRE_FALSE(outcome.has_value());
        auto const* error = Operon::EvaluationError(outcome);
        REQUIRE(error != nullptr);
        CHECK(error->Error.Code == Operon::GradientErrorCode::NonFiniteEvaluation);
        CHECK(error->Error.Row == 4);
        CHECK(error->FinalParameters == error->InitialParameters);
    }
}

TEST_CASE("FitLeastSquares performs no step with a zero iteration budget on either backend", "[least-squares][fit]")
{
    std::array<Scalar, 2> const start { 0, 0 };
    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));
        auto cost = MakeAffine(10, Scalar { 1 }, Scalar { 2 });
        auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend, .Iterations = 0 });

        // No accepted step is taken: the parameters are untouched and the cost does not improve.
        REQUIRE_FALSE(outcome.has_value());
        auto const& diag = Operon::Diagnostics(outcome);
        CHECK(diag.Iterations == 0);
        CHECK(diag.FinalParameters == diag.InitialParameters);
        CHECK(std::isfinite(static_cast<double>(diag.InitialCost)));
        CHECK(diag.FinalCost == diag.InitialCost);
        CHECK(Operon::EvaluationError(outcome) == nullptr);
        CHECK(Operon::ConfigurationError(outcome) == nullptr);
        // The initial point is evaluated, but no trial step is ever requested.
        CHECK(cost.calls >= 1);
        CHECK(cost.calls <= 2);
    }
}

TEST_CASE("FitLeastSquares treats the iteration budget as accepted steps", "[least-squares][fit]")
{
    std::array<Scalar, 2> const start { 0, 0 };
    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));
        auto cost = MakeAffine(10, Scalar { 1 }, Scalar { 2 });
        auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend, .Iterations = 1 });

        // One accepted step on a linear problem; the diagnostics count it as one iteration.
        REQUIRE(outcome.has_value());
        CHECK(outcome->Iterations == 1);
        CHECK(outcome->FinalCost < outcome->InitialCost);
    }

    // A budget of N must permit N accepted Eigen steps (Eigen's own counter starts at 1).
    // The exponential fit from a poor start needs far more than two steps to converge.
    constexpr std::array<Scalar, 2> expStart { 1, 0 };
    for (std::size_t budget : { std::size_t { 2 }, std::size_t { 3 } }) {
        CAPTURE(budget);
        ExponentialCost cost { 20, Scalar { 2 }, Scalar { 1 } };
        auto outcome = Operon::FitLeastSquares(
            cost, expStart, { .Backend = Operon::OptimizerType::Eigen, .Iterations = budget });
        auto const& diag = Operon::Diagnostics(outcome);
        CHECK(static_cast<std::size_t>(diag.Iterations) == budget);
        CHECK(static_cast<std::size_t>(diag.JacobianEvaluations) >= budget);
    }
    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));
        ExponentialCost cost { 20, Scalar { 2 }, Scalar { 1 } };
        auto outcome = Operon::FitLeastSquares(cost, expStart, { .Backend = backend, .Iterations = 2 });
        CHECK(Operon::Diagnostics(outcome).Iterations <= 2);
    }
}

TEST_CASE("FitLeastSquares saturates oversized iteration budgets instead of wrapping", "[least-squares][fit]")
{
    constexpr auto sizeMax = std::numeric_limits<std::size_t>::max();
    // Values that wrap to zero or a negative int when narrowed, or whose
    // product with (n + 1) = 3 overflows size_t.
    std::vector<std::size_t> budgets { sizeMax, sizeMax / 2, (sizeMax / 3) + 1,
        static_cast<std::size_t>(std::numeric_limits<int>::max()) + 1,
        static_cast<std::size_t>(std::numeric_limits<std::uint32_t>::max()) };
    if constexpr (sizeof(std::size_t) > sizeof(std::uint32_t)) {
        budgets.push_back(std::size_t { 1 } << 32U); // narrows to int 0
    }

    std::array<Scalar, 2> const start { 0, 0 };
    for (auto backend : BACKENDS) {
        for (auto budget : budgets) {
            CAPTURE(static_cast<int>(backend), budget);
            auto cost = MakeAffine(20, Scalar { 2.5 }, Scalar { -1.3 });
            auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = backend, .Iterations = budget });

            REQUIRE(outcome.has_value());
            CHECK(outcome->Iterations > 0);
            Near(outcome->FinalParameters[0], Scalar { 2.5 }, 1e-3);
            Near(outcome->FinalParameters[1], Scalar { -1.3 }, 1e-3);
        }
    }
}

TEST_CASE("FitLeastSquares rejects an underdetermined Eigen fit as InvalidShape before solving", "[least-squares][fit]")
{
    // One residual, two parameters.
    auto cost = MakeAffine(1, Scalar { 1 }, Scalar { 1 });
    std::array<Scalar, 2> const start { 0.25, -0.5 };
    auto outcome = Operon::FitLeastSquares(cost, start, { .Backend = Operon::OptimizerType::Eigen });

    REQUIRE_FALSE(outcome.has_value());
    auto const* error = Operon::EvaluationError(outcome);
    REQUIRE(error != nullptr);
    CHECK(Operon::ConfigurationError(outcome) == nullptr);
    CHECK(error->Error.Code == Operon::GradientErrorCode::InvalidShape);
    CHECK(error->Error.Expected == 2);
    CHECK(error->Error.Actual == 1);
    CHECK(error->InitialParameters == std::vector<Scalar>(start.begin(), start.end()));
    CHECK(error->FinalParameters == error->InitialParameters);
    CHECK(std::isnan(static_cast<double>(error->InitialCost)));
    CHECK(std::isnan(static_cast<double>(error->FinalCost)));
    CHECK(error->Iterations == 0);
    CHECK(error->FunctionEvaluations == 0);
    CHECK(error->JacobianEvaluations == 0);
    CHECK(cost.calls == 0);

    // The restriction is Eigen's: Tiny still runs the same problem.
    auto tinyCost = MakeAffine(1, Scalar { 1 }, Scalar { 1 });
    auto tiny = Operon::FitLeastSquares(tinyCost, start, { .Backend = Operon::OptimizerType::Tiny });
    CHECK(tinyCost.calls > 0);
    if (auto const* tinyError = Operon::EvaluationError(tiny); tinyError != nullptr) {
        CHECK(tinyError->Error.Code != Operon::GradientErrorCode::InvalidShape);
    }
}

TEST_CASE("FitLeastSquares reports NaN costs when the initial evaluation fails", "[least-squares][fit]")
{
    std::array<Scalar, 2> const start { 0, 0 };
    for (auto backend : BACKENDS) {
        CAPTURE(static_cast<int>(backend));

        auto failing = MakeAffine(8, Scalar { 1 }, Scalar { 1 });
        failing.failFromCall = 1;
        auto failed = Operon::FitLeastSquares(failing, start, { .Backend = backend });
        REQUIRE_FALSE(failed.has_value());
        auto const* failure = Operon::EvaluationError(failed);
        REQUIRE(failure != nullptr);
        CHECK(failure->Error.Code == Operon::GradientErrorCode::NumericalFailure);
        CHECK(failure->Error.Row == 7);
        CHECK(failure->FinalParameters == failure->InitialParameters);
        CHECK(std::isnan(static_cast<double>(failure->InitialCost)));
        CHECK(std::isnan(static_cast<double>(failure->FinalCost)));
        CHECK(failure->Iterations == 0);
        CHECK(failure->FunctionEvaluations == 1);

        auto nonFinite = MakeAffine(8, Scalar { 1 }, Scalar { 1 });
        nonFinite.nanRow = 2;
        auto rejected = Operon::FitLeastSquares(nonFinite, start, { .Backend = backend });
        REQUIRE_FALSE(rejected.has_value());
        auto const* nonFiniteError = Operon::EvaluationError(rejected);
        REQUIRE(nonFiniteError != nullptr);
        CHECK(nonFiniteError->Error.Code == Operon::GradientErrorCode::NonFiniteEvaluation);
        CHECK(std::isnan(static_cast<double>(nonFiniteError->InitialCost)));
        CHECK(std::isnan(static_cast<double>(nonFiniteError->FinalCost)));
        CHECK(nonFiniteError->Iterations == 0);
    }
}
