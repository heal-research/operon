// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <optional>
#include <span>
#include <tuple>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <unsupported/Eigen/LevenbergMarquardt>

#include "operon/ceres/tiny_solver.h"
#include "operon/optimizer/least_squares_lm_adapter.hpp"

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

// Always fails, to exercise the adapter's error path.
class FailingCost final : public Operon::LeastSquaresCostFunction {
public:
    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 5; }

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

TEST_CASE("LeastSquaresLMAdapter drives Eigen::LevenbergMarquardt to the true optimum", "[least-squares][lm-adapter]")
{
    auto const c0 = Operon::Scalar { 2.5 };
    auto const c1 = Operon::Scalar { -1.3 };
    auto cost = MakeLinearFixture(20, c0, c1);
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };

    Eigen::Matrix<Operon::Scalar, -1, 1> params(2);
    params << 0, 0;
    Eigen::LevenbergMarquardt<decltype(adapter)> lm(adapter);
    auto status = lm.minimize(params);

    CHECK(status >= Eigen::LevenbergMarquardtSpace::Status::RelativeReductionTooSmall); // converged, not still running/improper
    CHECK_THAT(static_cast<double>(params[0]), Catch::Matchers::WithinAbs(static_cast<double>(c0), 1e-3));
    CHECK_THAT(static_cast<double>(params[1]), Catch::Matchers::WithinAbs(static_cast<double>(c1), 1e-3));
    CHECK(adapter.ResidualCalls() > 0);
    CHECK(adapter.JacobianCalls() > 0);
    CHECK_FALSE(adapter.Error().has_value());
}

TEST_CASE("LeastSquaresLMAdapter drives ceres::TinySolver to the true optimum", "[least-squares][lm-adapter]")
{
    auto const c0 = Operon::Scalar { -0.7 };
    auto const c1 = Operon::Scalar { 2.2 };
    auto cost = MakeLinearFixture(20, c0, c1);
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };

    ceres::TinySolver<decltype(adapter)> solver;
    typename decltype(solver)::ParameterVector params;
    params.resize(2);
    params << 0, 0;
    solver.Solve(adapter, &params);

    CHECK_THAT(static_cast<double>(params[0]), Catch::Matchers::WithinAbs(static_cast<double>(c0), 1e-3));
    CHECK_THAT(static_cast<double>(params[1]), Catch::Matchers::WithinAbs(static_cast<double>(c1), 1e-3));
    CHECK(solver.summary.final_cost < solver.summary.initial_cost);
}

TEST_CASE("LeastSquaresLMAdapter surfaces evaluation errors and fills NaN", "[least-squares][lm-adapter]")
{
    FailingCost cost;
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };

    std::array<Operon::Scalar, 2> params { 0, 0 };
    std::vector<Operon::Scalar> residuals(5, Operon::Scalar { 1 });
    std::vector<Operon::Scalar> jacobian(10, Operon::Scalar { 1 });

    auto ok = adapter.Evaluate(params.data(), residuals.data(), jacobian.data());
    CHECK_FALSE(ok);
    REQUIRE(adapter.Error().has_value());
    CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
    for (auto r : residuals) { CHECK(std::isnan(static_cast<double>(r))); }
    for (auto j : jacobian) { CHECK(std::isnan(static_cast<double>(j))); }
}

TEST_CASE("LeastSquaresLMAdapter: Jacobian-only evaluation matches a residual+Jacobian call", "[least-squares][lm-adapter]")
{
    auto cost = MakeLinearFixture(8, Operon::Scalar { 1 }, Operon::Scalar { 0.5 });
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
    std::array<Operon::Scalar, 2> params { 0.2, -0.1 };

    std::vector<Operon::Scalar> jacobianOnly(cost.NumResiduals() * 2);
    auto okJacOnly = adapter.Evaluate(params.data(), nullptr, jacobianOnly.data());
    REQUIRE(okJacOnly);

    std::vector<Operon::Scalar> residuals(cost.NumResiduals());
    std::vector<Operon::Scalar> jacobianBoth(cost.NumResiduals() * 2);
    auto okBoth = adapter.Evaluate(params.data(), residuals.data(), jacobianBoth.data());
    REQUIRE(okBoth);

    CHECK(jacobianOnly == jacobianBoth);
}

TEST_CASE("LeastSquaresLMAdapter: row-major and column-major Jacobian storage agree", "[least-squares][lm-adapter]")
{
    auto cost = MakeLinearFixture(6, Operon::Scalar { 0.3 }, Operon::Scalar { -0.2 });
    std::array<Operon::Scalar, 2> params { 0.1, 0.1 };
    auto const n = cost.NumResiduals();

    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> colMajorAdapter { &cost };
    Operon::LeastSquaresLMAdapter<Eigen::RowMajor> rowMajorAdapter { &cost };

    std::vector<Operon::Scalar> colJacobian(n * 2);
    std::vector<Operon::Scalar> rowJacobian(n * 2);
    REQUIRE(colMajorAdapter.Evaluate(params.data(), nullptr, colJacobian.data()));
    REQUIRE(rowMajorAdapter.Evaluate(params.data(), nullptr, rowJacobian.data()));

    for (std::size_t i = 0; i < n; ++i) {
        for (std::size_t j = 0; j < 2; ++j) {
            auto const colValue = colJacobian.at((j * n) + i); // column-major: stride {1, n}
            auto const rowValue = rowJacobian.at((i * 2) + j); // row-major: stride {2, 1}
            CHECK(colValue == rowValue);
        }
    }
}

TEST_CASE("LeastSquaresLMAdapter: weighted fit agrees between Tiny and Eigen backends", "[least-squares][lm-adapter]")
{
    auto const c0 = Operon::Scalar { 1.1 };
    auto const c1 = Operon::Scalar { -0.4 };
    auto cost = MakeLinearFixture(16, c0, c1);
    std::vector<Operon::Scalar> weights(cost.NumResiduals());
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = Operon::Scalar { 1 } + (Operon::Scalar { 0.1 } * static_cast<Operon::Scalar>(i));
    }

    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> eigenAdapter { &cost, weights };
    Eigen::Matrix<Operon::Scalar, -1, 1> eigenParams(2);
    eigenParams << 0, 0;
    Eigen::LevenbergMarquardt<decltype(eigenAdapter)> lm(eigenAdapter);
    lm.minimize(eigenParams);

    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> tinyAdapter { &cost, weights };
    ceres::TinySolver<decltype(tinyAdapter)> solver;
    typename decltype(solver)::ParameterVector tinyParams;
    tinyParams.resize(2);
    tinyParams << 0, 0;
    solver.Solve(tinyAdapter, &tinyParams);

    CHECK_THAT(static_cast<double>(eigenParams[0]), Catch::Matchers::WithinAbs(static_cast<double>(tinyParams[0]), 1e-3));
    CHECK_THAT(static_cast<double>(eigenParams[1]), Catch::Matchers::WithinAbs(static_cast<double>(tinyParams[1]), 1e-3));
    CHECK_THAT(static_cast<double>(eigenParams[0]), Catch::Matchers::WithinAbs(static_cast<double>(c0), 1e-2));
    CHECK_THAT(static_cast<double>(eigenParams[1]), Catch::Matchers::WithinAbs(static_cast<double>(c1), 1e-2));
}

TEST_CASE("LeastSquaresLMAdapter: uniform scalar weight scales the objective without changing the optimum", "[least-squares][lm-adapter]")
{
    auto const c0 = Operon::Scalar { 0.6 };
    auto const c1 = Operon::Scalar { 1.4 };
    auto cost = MakeLinearFixture(10, c0, c1);
    std::array<Operon::Scalar, 1> const scalarWeight { Operon::Scalar { 2.5 } };

    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, scalarWeight };
    Eigen::Matrix<Operon::Scalar, -1, 1> params(2);
    params << 0, 0;
    Eigen::LevenbergMarquardt<decltype(adapter)> lm(adapter);
    lm.minimize(params);

    CHECK_THAT(static_cast<double>(params[0]), Catch::Matchers::WithinAbs(static_cast<double>(c0), 1e-3));
    CHECK_THAT(static_cast<double>(params[1]), Catch::Matchers::WithinAbs(static_cast<double>(c1), 1e-3));
}

TEST_CASE("LeastSquaresLMAdapter rejects nonfinite canonical outputs", "[least-squares][lm-adapter]")
{
    class NonFiniteCost final : public Operon::LeastSquaresCostFunction {
    public:
        [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 1; }
        [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 1; }
        [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const>, std::span<Operon::Scalar> residuals,
            std::optional<Operon::ScalarMatrixView> jacobian) const
            -> tl::expected<void, Operon::LeastSquaresError> override
        {
            residuals[0] = std::numeric_limits<Operon::Scalar>::quiet_NaN();
            if (jacobian) { Operon::At(*jacobian, 0, 0) = Operon::Scalar { 1 }; }
            return {};
        }
    } cost;
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
    Operon::Scalar parameter = 0;
    Operon::Scalar residual = 0;
    Operon::Scalar jacobian = 0;
    CHECK_FALSE(adapter.Evaluate(&parameter, &residual, &jacobian));
    REQUIRE(adapter.Error().has_value());
    CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
}

TEST_CASE("LeastSquaresLMAdapter reports the backend limit for an oversized problem without calling the cost", "[least-squares][lm-adapter]")
{
    class OversizedInvalidCost final : public Operon::LeastSquaresCostFunction {
    public:
        [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
        [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return std::numeric_limits<std::size_t>::max(); }
        [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const>, std::span<Operon::Scalar>,
            std::optional<Operon::ScalarMatrixView>) const
            -> tl::expected<void, Operon::LeastSquaresError> override
        {
            ++calls;
            return tl::unexpected(Operon::LeastSquaresError { .Code = Operon::LeastSquaresErrorCode::InvalidShape, .Expected = 9, .Actual = 4, .Row = 1, .Column = 2 });
        }

        mutable std::size_t calls {};
    } cost;

    // Construction must neither allocate nor throw.
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
    CHECK(adapter.ResidualCount() == std::numeric_limits<std::size_t>::max());
    CHECK(adapter.ParameterCount() == 2);
    CHECK(adapter.ExceedsBackendLimit());
    CHECK_FALSE(adapter.Error().has_value());

    // A Jacobian-only call needs internal scratch; it must reject the
    // oversized shape before calling the cost.
    std::array<Operon::Scalar, 2> const parameters { 0, 0 };
    CHECK_FALSE(adapter.Evaluate(parameters.data(), nullptr, nullptr));
    REQUIRE(adapter.Error().has_value());
    CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::InvalidShape);
    CHECK(adapter.Error()->Expected == decltype(adapter)::MaxBackendResiduals);
    CHECK(adapter.Error()->Actual == std::numeric_limits<std::size_t>::max());
    CHECK(adapter.Error()->Row == 0);
    CHECK(adapter.Error()->Column == 0);
    CHECK(adapter.ResidualCalls() == 0);
    CHECK(adapter.JacobianCalls() == 0);
    CHECK(cost.calls == 0);
}

TEST_CASE("LeastSquaresLMAdapter recovers nonfinite solver trials when enabled", "[least-squares][lm-adapter]")
{
    class NonFiniteCost final : public Operon::LeastSquaresCostFunction {
    public:
        [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 1; }
        [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 1; }

        [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const>, std::span<Operon::Scalar> residuals,
            std::optional<Operon::ScalarMatrixView> jacobian) const
            -> tl::expected<void, Operon::LeastSquaresError> override
        {
            residuals[0] = std::numeric_limits<Operon::Scalar>::quiet_NaN();
            if (jacobian) { Operon::At(*jacobian, 0, 0) = Operon::Scalar { 1 }; }
            return {};
        }
    } cost;
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, {}, true };
    Operon::Scalar parameter = 0;
    Operon::Scalar residual = 0;
    Operon::Scalar jacobian = 0;
    CHECK(adapter.Evaluate(&parameter, &residual, &jacobian));
    CHECK(jacobian == Operon::Scalar { 1 });
    CHECK_FALSE(adapter.Error().has_value());
}

namespace {
// Records the canonical view the adapter hands to the cost, then delegates to a linear model.
class RecordingCost final : public Operon::LeastSquaresCostFunction {
public:
    struct Call {
        std::size_t Parameters {};
        std::size_t Residuals {};
        bool HasJacobian {};
        std::size_t Rows {};
        std::size_t Columns {};
        std::size_t RowStride {};
        std::size_t ColumnStride {};
    };

    RecordingCost(std::vector<Operon::Scalar> x, std::vector<Operon::Scalar> y)
        : inner_(std::move(x), std::move(y))
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return inner_.NumParameters(); }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return inner_.NumResiduals(); }

    [[nodiscard]] auto Evaluate(
        std::span<Operon::Scalar const> parameters,
        std::span<Operon::Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> jacobian) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        Call call { .Parameters = parameters.size(), .Residuals = residuals.size(), .HasJacobian = jacobian.has_value() };
        if (jacobian) {
            call.Rows = jacobian->extent(0);
            call.Columns = jacobian->extent(1);
            call.RowStride = jacobian->stride(0);
            call.ColumnStride = jacobian->stride(1);
        }
        calls.push_back(call);
        return inner_.Evaluate(parameters, residuals, jacobian);
    }

    mutable std::vector<Call> calls;

private:
    LinearModelCost inner_;
};
} // namespace

TEST_CASE("LeastSquaresLMAdapter hands the cost exact-size spans and a logical [row,column] view", "[least-squares][lm-adapter]")
{
    std::vector<Operon::Scalar> x(7);
    std::vector<Operon::Scalar> y(7);
    for (std::size_t i = 0; i < x.size(); ++i) {
        x[i] = static_cast<Operon::Scalar>(i);
        y[i] = Operon::Scalar { 1 } + (Operon::Scalar { 2 } * x[i]);
    }
    RecordingCost cost { std::move(x), std::move(y) };
    std::array<Operon::Scalar, 2> params { 0.5, 0.5 };
    std::vector<Operon::Scalar> residuals(7);
    std::vector<Operon::Scalar> jacobian(14);

    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> colAdapter { &cost };
    CHECK(colAdapter.NumResiduals() == 7);
    CHECK(colAdapter.NumParameters() == 2);
    REQUIRE(colAdapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
    REQUIRE(colAdapter.Evaluate(params.data(), residuals.data(), nullptr));
    REQUIRE(colAdapter.Evaluate(params.data(), nullptr, jacobian.data()));

    REQUIRE(cost.calls.size() == 3);
    for (auto const& call : cost.calls) {
        CHECK(call.Parameters == 2);
        CHECK(call.Residuals == 7);
    }
    CHECK(cost.calls[0].HasJacobian);
    CHECK_FALSE(cost.calls[1].HasJacobian); // residual-only request carries no Jacobian
    CHECK(cost.calls[2].HasJacobian); // Jacobian-only request still supplies exact-size residual storage
    CHECK(cost.calls[0].Rows == 7);
    CHECK(cost.calls[0].Columns == 2);
    CHECK(cost.calls[0].RowStride == 1);
    CHECK(cost.calls[0].ColumnStride == 7);
    CHECK(colAdapter.ResidualCalls() == 2);
    CHECK(colAdapter.JacobianCalls() == 2);

    cost.calls.clear();
    Operon::LeastSquaresLMAdapter<Eigen::RowMajor> rowAdapter { &cost };
    REQUIRE(rowAdapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
    REQUIRE(cost.calls.size() == 1);
    CHECK(cost.calls[0].Rows == 7);
    CHECK(cost.calls[0].Columns == 2);
    CHECK(cost.calls[0].RowStride == 2);
    CHECK(cost.calls[0].ColumnStride == 1);
}

TEST_CASE("LeastSquaresLMAdapter: weighted Jacobian-only evaluation scales rows by sqrt(weight)", "[least-squares][lm-adapter]")
{
    auto cost = MakeLinearFixture(5, Operon::Scalar { 1 }, Operon::Scalar { 0.5 });
    std::vector<Operon::Scalar> weights { 1, 4, 9, 16, 25 };
    std::array<Operon::Scalar, 2> params { 0.2, -0.1 };
    auto const n = cost.NumResiduals();

    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> plain { &cost };
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> weighted { &cost, weights };

    std::vector<Operon::Scalar> plainJacobian(n * 2);
    std::vector<Operon::Scalar> weightedJacobian(n * 2);
    REQUIRE(plain.Evaluate(params.data(), nullptr, plainJacobian.data()));
    REQUIRE(weighted.Evaluate(params.data(), nullptr, weightedJacobian.data()));

    for (std::size_t i = 0; i < n; ++i) {
        auto const sw = static_cast<Operon::Scalar>(i + 1);
        for (std::size_t j = 0; j < 2; ++j) {
            CHECK(weightedJacobian[(j * n) + i] == sw * plainJacobian[(j * n) + i]);
        }
    }

    std::vector<Operon::Scalar> plainResiduals(n);
    std::vector<Operon::Scalar> weightedResiduals(n);
    REQUIRE(plain.Evaluate(params.data(), plainResiduals.data(), nullptr));
    REQUIRE(weighted.Evaluate(params.data(), weightedResiduals.data(), nullptr));
    for (std::size_t i = 0; i < n; ++i) {
        CHECK(weightedResiduals[i] == static_cast<Operon::Scalar>(i + 1) * plainResiduals[i]);
    }
}

TEST_CASE("LeastSquaresLMAdapter: invalid weights are a typed InvalidWeights error, never a sqrt of a negative", "[least-squares][lm-adapter]")
{
    auto cost = MakeLinearFixture(5, Operon::Scalar { 1 }, Operon::Scalar { 0.5 });
    std::array<Operon::Scalar, 2> params { 0.2, -0.1 };
    auto const n = cost.NumResiduals();

    auto const checkRejected = [&](std::vector<Operon::Scalar> const& weights, std::size_t expectedRow) -> void {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, weights };
        REQUIRE(adapter.Error().has_value());
        CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::InvalidWeights);
        CHECK(adapter.Error()->Row == expectedRow);

        std::vector<Operon::Scalar> residuals(n);
        std::vector<Operon::Scalar> jacobian(n * 2);
        CHECK_FALSE(adapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
        for (auto r : residuals) { CHECK(std::isnan(static_cast<double>(r))); }
        for (auto j : jacobian) { CHECK(std::isnan(static_cast<double>(j))); }
        CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::InvalidWeights);
        CHECK(adapter.ResidualCalls() == 0);
    };

    SECTION("negative weight") {
        checkRejected({ 1, 1, -4, 1, 1 }, 2);
    }

    SECTION("NaN weight") {
        checkRejected({ 1, std::numeric_limits<Operon::Scalar>::quiet_NaN(), 1, 1, 1 }, 1);
    }

    SECTION("wrong per-row size") {
        std::vector<Operon::Scalar> wrong { 1, 2, 3 };
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, wrong };
        REQUIRE(adapter.Error().has_value());
        CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::InvalidWeights);
        CHECK(adapter.Error()->Expected == n);
        CHECK(adapter.Error()->Actual == wrong.size());
        std::vector<Operon::Scalar> residuals(n);
        CHECK_FALSE(adapter.Evaluate(params.data(), residuals.data(), nullptr));
    }

    SECTION("valid weights record no error") {
        std::vector<Operon::Scalar> weights { 1, 2, 3, 4, 5 };
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, weights };
        CHECK_FALSE(adapter.Error().has_value());
    }
}

TEST_CASE("LeastSquaresLMAdapter preserves typed cost errors through both solver backends", "[least-squares][lm-adapter]")
{
    class TypedFailureCost final : public Operon::LeastSquaresCostFunction {
    public:
        [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
        [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 4; }
        [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const>, std::span<Operon::Scalar>,
            std::optional<Operon::ScalarMatrixView>) const
            -> tl::expected<void, Operon::LeastSquaresError> override
        {
            return tl::unexpected(Operon::LeastSquaresError {
                .Code = Operon::LeastSquaresErrorCode::EvaluationFailure,
                .Expected = 4,
                .Actual = 3,
                .Cause = Operon::InterpreterError { .Kind = Operon::InterpreterError::Code::MissingVariable, .Hash = 77 } });
        }
    };

    auto const checkError = [](auto const& adapter) {
        REQUIRE(adapter.Error().has_value());
        CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::EvaluationFailure);
        CHECK(adapter.Error()->Expected == 4);
        CHECK(adapter.Error()->Actual == 3);
        REQUIRE(adapter.Error()->Cause.has_value());
        CHECK(adapter.Error()->Cause->Kind == Operon::InterpreterError::Code::MissingVariable);
        CHECK(adapter.Error()->Cause->Hash == 77);
    };

    TypedFailureCost cost;
    {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
        ceres::TinySolver<decltype(adapter)> solver;
        typename decltype(solver)::ParameterVector params;
        params.resize(2);
        params << 0, 0;
        solver.Solve(adapter, &params);
        checkError(adapter);
        CHECK(params[0] == Operon::Scalar { 0 }); // a failed evaluation never moves the parameters
        CHECK(params[1] == Operon::Scalar { 0 });
    }
    {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
        Eigen::Matrix<Operon::Scalar, -1, 1> params(2);
        params << 0, 0;
        Eigen::LevenbergMarquardt<decltype(adapter)> lm(adapter);
        lm.minimize(params);
        checkError(adapter);
        CHECK(params[0] == Operon::Scalar { 0 });
        CHECK(params[1] == Operon::Scalar { 0 });
    }
}

TEST_CASE("LeastSquaresLMAdapter reports where the first non-finite output occurred", "[least-squares][lm-adapter]")
{
    class PoisonedCost final : public Operon::LeastSquaresCostFunction {
    public:
        PoisonedCost(std::size_t badResidual, std::size_t badRow, std::size_t badColumn)
            : badResidual_(badResidual)
            , badRow_(badRow)
            , badColumn_(badColumn)
        {
        }
        [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 3; }
        [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 4; }
        [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const>, std::span<Operon::Scalar> residuals,
            std::optional<Operon::ScalarMatrixView> jacobian) const
            -> tl::expected<void, Operon::LeastSquaresError> override
        {
            std::ranges::fill(residuals, Operon::Scalar { 1 });
            if (badResidual_ < residuals.size()) { residuals[badResidual_] = std::numeric_limits<Operon::Scalar>::infinity(); }
            if (jacobian) {
                for (std::size_t i = 0; i < 4; ++i) {
                    for (std::size_t j = 0; j < 3; ++j) { Operon::At(*jacobian, i, j) = Operon::Scalar { 1 }; }
                }
                if (badRow_ < 4) { Operon::At(*jacobian, badRow_, badColumn_) = std::numeric_limits<Operon::Scalar>::quiet_NaN(); }
            }
            return {};
        }

    private:
        std::size_t badResidual_;
        std::size_t badRow_;
        std::size_t badColumn_;
    };

    std::array<Operon::Scalar, 3> params { 0, 0, 0 };
    std::vector<Operon::Scalar> residuals(4);
    std::vector<Operon::Scalar> jacobian(12);

    PoisonedCost residualCost { 2, 99, 0 };
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> residualAdapter { &residualCost };
    CHECK_FALSE(residualAdapter.Evaluate(params.data(), residuals.data(), nullptr));
    REQUIRE(residualAdapter.Error().has_value());
    CHECK(residualAdapter.Error()->Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
    CHECK(residualAdapter.Error()->Row == 2);

    PoisonedCost jacobianCost { 99, 3, 1 };
    Operon::LeastSquaresLMAdapter<Eigen::RowMajor> jacobianAdapter { &jacobianCost };
    CHECK_FALSE(jacobianAdapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
    REQUIRE(jacobianAdapter.Error().has_value());
    CHECK(jacobianAdapter.Error()->Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
    CHECK(jacobianAdapter.Error()->Row == 3); // logical row/column, independent of storage order
    CHECK(jacobianAdapter.Error()->Column == 1);
}

TEST_CASE("LeastSquaresLMAdapter TinySolver runs are deterministic", "[least-squares][lm-adapter]")
{
    auto cost = MakeLinearFixture(12, Operon::Scalar { 0.9 }, Operon::Scalar { -1.7 });
    std::vector<Operon::Scalar> weights(cost.NumResiduals());
    for (std::size_t i = 0; i < weights.size(); ++i) { weights[i] = Operon::Scalar { 1 } + static_cast<Operon::Scalar>(i); }

    auto const solve = [&] {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, weights };
        ceres::TinySolver<decltype(adapter)> solver;
        typename decltype(solver)::ParameterVector params;
        params.resize(2);
        params << 0.3, 0.3;
        solver.Solve(adapter, &params);
        return std::tuple { params[0], params[1], solver.summary.final_cost, solver.summary.iterations, adapter.ResidualCalls(), adapter.JacobianCalls() };
    };

    CHECK(solve() == solve());
}

namespace {
// y_i = 1 + 2 x_i + 0.1 x_i^2 + deterministic perturbation: the model misfit makes the weighted optimum differ from the unweighted one.
struct NoisyLine {
    std::vector<Operon::Scalar> X;
    std::vector<Operon::Scalar> Y;
    std::vector<Operon::Scalar> W;
};

auto MakeNoisyLine(std::size_t n) -> NoisyLine
{
    NoisyLine line;
    for (std::size_t i = 0; i < n; ++i) {
        auto const x = (Operon::Scalar { 0.5 } * static_cast<Operon::Scalar>(i)) - (static_cast<Operon::Scalar>(n) / Operon::Scalar { 4 });
        line.X.push_back(x);
        line.Y.push_back(Operon::Scalar { 1 } + (Operon::Scalar { 2 } * x) + (Operon::Scalar { 0.1 } * x * x) + (Operon::Scalar { 0.8 } * static_cast<Operon::Scalar>(std::sin(3.0 * static_cast<double>(i)))));
        line.W.push_back(Operon::Scalar { 0.1 } + (static_cast<Operon::Scalar>(i) * static_cast<Operon::Scalar>(i)));
    }
    return line;
}

// Closed-form minimiser of sum_i w_i (a + b x_i - y_i)^2 via the 2x2 normal equations.
auto SolveWeightedLine(NoisyLine const& line, bool weighted) -> std::array<double, 2>
{
    double sw = 0;
    double sx = 0;
    double sy = 0;
    double sxx = 0;
    double sxy = 0;
    for (std::size_t i = 0; i < line.X.size(); ++i) {
        auto const w = weighted ? static_cast<double>(line.W[i]) : 1.0;
        auto const x = static_cast<double>(line.X[i]);
        auto const y = static_cast<double>(line.Y[i]);
        sw += w;
        sx += w * x;
        sy += w * y;
        sxx += w * x * x;
        sxy += w * x * y;
    }
    auto const det = (sw * sxx) - (sx * sx);
    return { ((sxx * sy) - (sx * sxy)) / det, ((sw * sxy) - (sx * sy)) / det };
}

auto WeightedObjective(NoisyLine const& line, double a, double b) -> double
{
    double total = 0;
    for (std::size_t i = 0; i < line.X.size(); ++i) {
        auto const r = a + (b * static_cast<double>(line.X[i])) - static_cast<double>(line.Y[i]);
        total += static_cast<double>(line.W[i]) * r * r;
    }
    return total;
}
} // namespace

TEST_CASE("LeastSquaresLMAdapter: both backends minimise the same weighted objective and expose the same residual/Jacobian layout", "[least-squares][lm-adapter]")
{
    auto const line = MakeNoisyLine(24);
    auto const wls = SolveWeightedLine(line, true);
    auto const ols = SolveWeightedLine(line, false);
    REQUIRE(std::abs(wls[1] - ols[1]) > 0.1); // weights must actually move the optimum for this to be a weighted test
    auto const referenceObjective = WeightedObjective(line, wls[0], wls[1]);
    auto const n = line.X.size();

    LinearModelCost cost { line.X, line.Y };
    auto const checkLayout = [&](auto const& residuals, auto const& jacobian) {
        for (std::size_t i = 0; i < n; ++i) {
            auto const sw = std::sqrt(static_cast<double>(line.W[i]));
            auto const x = static_cast<double>(line.X[i]);
            auto const r = wls[0] + (wls[1] * x) - static_cast<double>(line.Y[i]);
            CHECK_THAT(static_cast<double>(residuals[i]), Catch::Matchers::WithinAbs(sw * r, 5e-2));
            CHECK_THAT(static_cast<double>(jacobian(i, 0)), Catch::Matchers::WithinAbs(sw, 1e-4));
            CHECK_THAT(static_cast<double>(jacobian(i, 1)), Catch::Matchers::WithinAbs(sw * x, 1e-4));
        }
    };

    {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, line.W };
        Eigen::Matrix<Operon::Scalar, -1, 1> params(2);
        params << 0, 0;
        Eigen::LevenbergMarquardt<decltype(adapter)> lm(adapter);
        lm.minimize(params);
        CHECK_THAT(static_cast<double>(params[0]), Catch::Matchers::WithinAbs(wls[0], 2e-3));
        CHECK_THAT(static_cast<double>(params[1]), Catch::Matchers::WithinAbs(wls[1], 2e-3));
        CHECK_THAT(static_cast<double>(lm.fnorm() * lm.fnorm()), Catch::Matchers::WithinRel(referenceObjective, 1e-3)); // fnorm = sqrt(sum w r^2)
        Eigen::Matrix<Operon::Scalar, -1, -1> jacobian(n, 2);
        REQUIRE(adapter.df(params, jacobian) == 0);
        checkLayout(lm.fvec(), jacobian);
    }
    {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, line.W };
        ceres::TinySolver<decltype(adapter)> solver;
        typename decltype(solver)::ParameterVector params;
        params.resize(2);
        params << 0, 0;
        solver.Solve(adapter, &params);
        CHECK_THAT(static_cast<double>(params[0]), Catch::Matchers::WithinAbs(wls[0], 2e-3));
        CHECK_THAT(static_cast<double>(params[1]), Catch::Matchers::WithinAbs(wls[1], 2e-3));
        CHECK_THAT(static_cast<double>(solver.summary.final_cost * 2), Catch::Matchers::WithinRel(referenceObjective, 1e-3)); // final_cost = 1/2 sum w r^2
        checkLayout(solver.Residuals(), solver.Jacobian());
    }
}

TEST_CASE("LeastSquaresLMAdapter: scalar weight is equivalent to the broadcast per-row weight on both backends", "[least-squares][lm-adapter]")
{
    auto const line = MakeNoisyLine(12);
    LinearModelCost cost { line.X, line.Y };
    std::array<Operon::Scalar, 1> const scalar { Operon::Scalar { 2.5 } };
    std::vector<Operon::Scalar> const broadcast(line.X.size(), Operon::Scalar { 2.5 });

    auto const solveEigen = [&](Operon::ConstScalarSpan weights) {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, weights };
        Eigen::Matrix<Operon::Scalar, -1, 1> params(2);
        params << 0.3, 0.3;
        Eigen::LevenbergMarquardt<decltype(adapter)> lm(adapter);
        lm.minimize(params);
        return std::tuple { params[0], params[1], lm.fnorm(), lm.iterations(), adapter.ResidualCalls(), adapter.JacobianCalls() };
    };
    auto const solveTiny = [&](Operon::ConstScalarSpan weights) {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, weights };
        ceres::TinySolver<decltype(adapter)> solver;
        typename decltype(solver)::ParameterVector params;
        params.resize(2);
        params << 0.3, 0.3;
        solver.Solve(adapter, &params);
        return std::tuple { params[0], params[1], solver.summary.final_cost, solver.summary.iterations, adapter.ResidualCalls(), adapter.JacobianCalls() };
    };

    CHECK(solveEigen(scalar) == solveEigen(broadcast));
    CHECK(solveTiny(scalar) == solveTiny(broadcast));

    // the weighted objective is sum(w r^2): a uniform weight scales the unweighted objective at fixed parameters.
    std::array<Operon::Scalar, 2> const params { 0.3, 0.3 };
    std::vector<Operon::Scalar> plain(line.X.size());
    std::vector<Operon::Scalar> weighted(line.X.size());
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> plainAdapter { &cost };
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> weightedAdapter { &cost, scalar };
    REQUIRE(plainAdapter.Evaluate(params.data(), plain.data(), nullptr));
    REQUIRE(weightedAdapter.Evaluate(params.data(), weighted.data(), nullptr));
    double plainObjective = 0;
    double weightedObjective = 0;
    for (std::size_t i = 0; i < plain.size(); ++i) {
        plainObjective += static_cast<double>(plain[i]) * static_cast<double>(plain[i]);
        weightedObjective += static_cast<double>(weighted[i]) * static_cast<double>(weighted[i]);
    }
    CHECK_THAT(weightedObjective, Catch::Matchers::WithinRel(2.5 * plainObjective, 1e-4));
}

TEST_CASE("LeastSquaresLMAdapter: call counters agree with each backend's own diagnostics and replay deterministically", "[least-squares][lm-adapter]")
{
    auto const line = MakeNoisyLine(16);
    LinearModelCost cost { line.X, line.Y };

    auto const solveEigen = [&] {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, line.W };
        Eigen::Matrix<Operon::Scalar, -1, 1> params(2);
        params << 0.3, 0.3;
        Eigen::LevenbergMarquardt<decltype(adapter)> lm(adapter);
        lm.minimize(params);
        // Eigen invokes the functor once per residual-only and once per Jacobian-only request.
        CHECK(adapter.ResidualCalls() == static_cast<std::size_t>(lm.nfev()));
        CHECK(adapter.JacobianCalls() == static_cast<std::size_t>(lm.njev()));
        CHECK(adapter.JacobianCalls() > 0);
        return std::tuple { params[0], params[1], lm.fnorm(), lm.iterations(), adapter.ResidualCalls(), adapter.JacobianCalls() };
    };
    auto const solveTiny = [&] {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, line.W };
        ceres::TinySolver<decltype(adapter)> solver;
        typename decltype(solver)::ParameterVector params;
        params.resize(2);
        params << 0.3, 0.3;
        solver.Solve(adapter, &params);
        CHECK(adapter.ResidualCalls() > 0);
        CHECK(adapter.JacobianCalls() > 0);
        return std::tuple { params[0], params[1], solver.summary.final_cost, solver.summary.iterations, adapter.ResidualCalls(), adapter.JacobianCalls() };
    };

    CHECK(solveEigen() == solveEigen());
    CHECK(solveTiny() == solveTiny());
}

TEST_CASE("LeastSquaresLMAdapter: a cost failing mid-solve surfaces the same typed error on both backends", "[least-squares][lm-adapter]")
{
    // Succeeds at the start point (0, 0) and fails for every other parameter vector, i.e. on the first trial step.
    class FailsAwayFromStart final : public Operon::LeastSquaresCostFunction {
    public:
        FailsAwayFromStart(std::vector<Operon::Scalar> x, std::vector<Operon::Scalar> y)
            : inner_(std::move(x), std::move(y))
        {
        }
        [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return inner_.NumParameters(); }
        [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return inner_.NumResiduals(); }
        [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const> parameters, std::span<Operon::Scalar> residuals,
            std::optional<Operon::ScalarMatrixView> jacobian) const
            -> tl::expected<void, Operon::LeastSquaresError> override
        {
            if (parameters[0] != Operon::Scalar { 0 } || parameters[1] != Operon::Scalar { 0 }) {
                return tl::unexpected(Operon::LeastSquaresError {
                    .Code = Operon::LeastSquaresErrorCode::EvaluationFailure,
                    .Expected = 6,
                    .Actual = 5,
                    .Cause = Operon::InterpreterError { .Kind = Operon::InterpreterError::Code::MissingVariable, .Hash = 99 } });
            }
            return inner_.Evaluate(parameters, residuals, jacobian);
        }

    private:
        LinearModelCost inner_;
    };

    auto const line = MakeNoisyLine(10);
    FailsAwayFromStart cost { line.X, line.Y };

    auto const checkError = [](auto const& adapter) {
        REQUIRE(adapter.Error().has_value());
        CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::EvaluationFailure);
        CHECK(adapter.Error()->Expected == 6);
        CHECK(adapter.Error()->Actual == 5);
        REQUIRE(adapter.Error()->Cause.has_value());
        CHECK(adapter.Error()->Cause->Kind == Operon::InterpreterError::Code::MissingVariable);
        CHECK(adapter.Error()->Cause->Hash == 99);
    };

    {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
        ceres::TinySolver<decltype(adapter)> solver;
        typename decltype(solver)::ParameterVector params;
        params.resize(2);
        params << 0, 0;
        solver.Solve(adapter, &params);
        checkError(adapter);
        CHECK(std::isnan(static_cast<double>(solver.summary.final_cost)));
        CHECK(params[0] == Operon::Scalar { 0 }); // the rejected trial never becomes the reported solution
        CHECK(params[1] == Operon::Scalar { 0 });
    }
    {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
        Eigen::Matrix<Operon::Scalar, -1, 1> params(2);
        params << 0, 0;
        Eigen::LevenbergMarquardt<decltype(adapter)> lm(adapter);
        auto const status = lm.minimize(params);
        CHECK(status == Eigen::LevenbergMarquardtSpace::Status::UserAsked);
        checkError(adapter);
        CHECK(params[0] == Operon::Scalar { 0 });
        CHECK(params[1] == Operon::Scalar { 0 });
    }
}

TEST_CASE("LeastSquaresLMAdapter: non-finite trial steps are recovered or reported identically on both backends", "[least-squares][lm-adapter]")
{
    // r(p) = log(p) - log(4): minimum at p = 4, NaN residual for p < 0. From p = 20 the first Gauss-Newton
    // step lands near p = -12, so a solver only converges if it can reject a non-finite trial.
    class LogCost final : public Operon::LeastSquaresCostFunction {
    public:
        [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 1; }
        [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 1; }
        [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const> parameters, std::span<Operon::Scalar> residuals,
            std::optional<Operon::ScalarMatrixView> jacobian) const
            -> tl::expected<void, Operon::LeastSquaresError> override
        {
            auto const p = static_cast<double>(parameters[0]);
            residuals[0] = static_cast<Operon::Scalar>(std::log(p) - std::log(4.0));
            if (jacobian) { Operon::At(*jacobian, 0, 0) = static_cast<Operon::Scalar>(1.0 / p); }
            return {};
        }
    } cost;

    auto const runEigen = [&](bool recover) {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, {}, recover };
        Eigen::Matrix<Operon::Scalar, -1, 1> params(1);
        params << 20;
        Eigen::LevenbergMarquardt<decltype(adapter)> lm(adapter);
        lm.minimize(params);
        return std::tuple { static_cast<double>(params[0]), adapter.Error() };
    };
    auto const runTiny = [&](bool recover) {
        Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, {}, recover };
        ceres::TinySolver<decltype(adapter)> solver;
        typename decltype(solver)::ParameterVector params;
        params.resize(1);
        params << 20;
        solver.Solve(adapter, &params);
        return std::tuple { static_cast<double>(params[0]), adapter.Error() };
    };

    for (auto const& [value, error] : { runEigen(true), runTiny(true) }) {
        CHECK_THAT(value, Catch::Matchers::WithinAbs(4.0, 1e-3));
        CHECK_FALSE(error.has_value());
    }
    for (auto const& [value, error] : { runEigen(false), runTiny(false) }) {
        CHECK(value == 20.0); // solve aborted at the first non-finite trial; no accepted step
        REQUIRE(error.has_value());
        CHECK(error->Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
        CHECK(error->Row == 0);
    }
}

namespace {
// Fails every call with an error whose fields encode the call index, so a test can tell which call's error survived.
class SequencedFailureCost final : public Operon::LeastSquaresCostFunction {
public:
    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 3; }
    [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const>, std::span<Operon::Scalar>,
        std::optional<Operon::ScalarMatrixView>) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        auto const call = ++calls;
        return tl::unexpected(Operon::LeastSquaresError {
            .Code = call == 1 ? Operon::LeastSquaresErrorCode::EvaluationFailure : Operon::LeastSquaresErrorCode::NumericalFailure,
            .Expected = 10 * call,
            .Actual = 20 * call,
            .Row = call,
            .Column = call + 1,
            .Cause = Operon::InterpreterError { .Kind = Operon::InterpreterError::Code::MissingVariable, .Hash = 1000 + call } });
    }
    mutable std::size_t calls {};
};

// Counts invocations; reports what the test configures (NaN residual/Jacobian, or a typed error) without a model.
class ScriptedCost final : public Operon::LeastSquaresCostFunction {
public:
    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t override { return 3; }
    [[nodiscard]] auto Evaluate(std::span<Operon::Scalar const>, std::span<Operon::Scalar> residuals,
        std::optional<Operon::ScalarMatrixView> jacobian) const
        -> tl::expected<void, Operon::LeastSquaresError> override
    {
        ++calls;
        if (failWithError) {
            return tl::unexpected(Operon::LeastSquaresError { .Code = Operon::LeastSquaresErrorCode::EvaluationFailure, .Row = 99 });
        }
        std::ranges::fill(residuals, Operon::Scalar { 2 });
        if (nanResidual < residuals.size()) { residuals[nanResidual] = std::numeric_limits<Operon::Scalar>::quiet_NaN(); }
        if (jacobian) {
            for (std::size_t i = 0; i < jacobian->extent(0); ++i) {
                for (std::size_t j = 0; j < jacobian->extent(1); ++j) { Operon::At(*jacobian, i, j) = Operon::Scalar { 1 }; }
            }
        }
        return {};
    }
    mutable std::size_t calls {};
    bool failWithError { false };
    std::size_t nanResidual { std::numeric_limits<std::size_t>::max() };
};
} // namespace

TEST_CASE("LeastSquaresLMAdapter keeps the first typed error across repeated failing calls", "[least-squares][lm-adapter]")
{
    SequencedFailureCost cost;
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
    std::array<Operon::Scalar, 2> params { 0, 0 };

    for (std::size_t call = 1; call <= 3; ++call) {
        std::vector<Operon::Scalar> residuals(3, Operon::Scalar { 1 });
        std::vector<Operon::Scalar> jacobian(6, Operon::Scalar { 1 });
        CHECK_FALSE(adapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
        // Outputs are poisoned on every failing call, not only the first.
        for (auto r : residuals) { CHECK(std::isnan(static_cast<double>(r))); }
        for (auto j : jacobian) { CHECK(std::isnan(static_cast<double>(j))); }

        REQUIRE(adapter.Error().has_value());
        auto const& error = *adapter.Error();
        CHECK(error.Code == Operon::LeastSquaresErrorCode::EvaluationFailure);
        CHECK(error.Expected == 10);
        CHECK(error.Actual == 20);
        CHECK(error.Row == 1);
        CHECK(error.Column == 2);
        REQUIRE(error.Cause.has_value());
        CHECK(error.Cause->Kind == Operon::InterpreterError::Code::MissingVariable);
        CHECK(error.Cause->Hash == 1001);
    }
    CHECK(cost.calls == 3); // later calls still reach the cost; only the stored error is frozen
}

TEST_CASE("LeastSquaresLMAdapter: a detected non-finite output is not replaced by a later cost error or a later success", "[least-squares][lm-adapter]")
{
    ScriptedCost cost;
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
    std::array<Operon::Scalar, 2> params { 0, 0 };
    std::vector<Operon::Scalar> residuals(3);
    std::vector<Operon::Scalar> jacobian(6);

    cost.nanResidual = 1;
    CHECK_FALSE(adapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
    REQUIRE(adapter.Error().has_value());
    CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
    CHECK(adapter.Error()->Row == 1);

    cost.nanResidual = std::numeric_limits<std::size_t>::max();
    cost.failWithError = true;
    CHECK_FALSE(adapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
    REQUIRE(adapter.Error().has_value());
    CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
    CHECK(adapter.Error()->Row == 1);

    cost.failWithError = false;
    CHECK(adapter.Evaluate(params.data(), residuals.data(), jacobian.data())); // a later success does not clear it
    REQUIRE(adapter.Error().has_value());
    CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::NonFiniteEvaluation);
    CHECK(adapter.Error()->Row == 1);
}

TEST_CASE("LeastSquaresLMAdapter: invalid stored configuration fails every call without evaluating, counting, or replacing the error", "[least-squares][lm-adapter]")
{
    ScriptedCost cost; // would succeed if it were ever called
    std::vector<Operon::Scalar> const weights { 1, -2, 1 };
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, weights };
    std::array<Operon::Scalar, 2> params { 0, 0 };

    auto const checkStoredError = [&]() -> void {
        REQUIRE(adapter.Error().has_value());
        CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::InvalidWeights);
        CHECK(adapter.Error()->Expected == 3);
        CHECK(adapter.Error()->Actual == 3);
        CHECK(adapter.Error()->Row == 1);
        CHECK_FALSE(adapter.Error()->Cause.has_value());
    };
    checkStoredError();

    for (int repeat = 0; repeat < 3; ++repeat) {
        std::vector<Operon::Scalar> residuals(3, Operon::Scalar { 1 });
        std::vector<Operon::Scalar> jacobian(6, Operon::Scalar { 1 });
        CHECK_FALSE(adapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
        for (auto r : residuals) { CHECK(std::isnan(static_cast<double>(r))); }
        for (auto j : jacobian) { CHECK(std::isnan(static_cast<double>(j))); }

        std::ranges::fill(residuals, Operon::Scalar { 1 });
        CHECK_FALSE(adapter.Evaluate(params.data(), residuals.data(), nullptr));
        for (auto r : residuals) { CHECK(std::isnan(static_cast<double>(r))); }

        std::ranges::fill(jacobian, Operon::Scalar { 1 });
        CHECK_FALSE(adapter.Evaluate(params.data(), nullptr, jacobian.data()));
        for (auto j : jacobian) { CHECK(std::isnan(static_cast<double>(j))); }

        checkStoredError();
    }
    CHECK(cost.calls == 0);
    CHECK(adapter.ResidualCalls() == 0);
    CHECK(adapter.JacobianCalls() == 0);
}

TEST_CASE("LeastSquaresLMAdapter counters count calls that reached the cost, per requested output", "[least-squares][lm-adapter]")
{
    auto cost = MakeLinearFixture(4, Operon::Scalar { 1 }, Operon::Scalar { 2 });
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost };
    std::array<Operon::Scalar, 2> params { 0, 0 };
    std::vector<Operon::Scalar> residuals(4);
    std::vector<Operon::Scalar> jacobian(8);

    CHECK(adapter.ResidualCalls() == 0);
    CHECK(adapter.JacobianCalls() == 0);

    REQUIRE(adapter.Evaluate(params.data(), residuals.data(), nullptr));
    CHECK(adapter.ResidualCalls() == 1);
    CHECK(adapter.JacobianCalls() == 0);

    REQUIRE(adapter.Evaluate(params.data(), nullptr, jacobian.data()));
    CHECK(adapter.ResidualCalls() == 1);
    CHECK(adapter.JacobianCalls() == 1);

    REQUIRE(adapter.Evaluate(params.data(), residuals.data(), jacobian.data())); // counts in both
    CHECK(adapter.ResidualCalls() == 2);
    CHECK(adapter.JacobianCalls() == 2);

    // A call that reaches the cost and fails is still counted.
    SequencedFailureCost failing;
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> failingAdapter { &failing };
    std::vector<Operon::Scalar> failingResiduals(3);
    std::vector<Operon::Scalar> failingJacobian(6);
    CHECK_FALSE(failingAdapter.Evaluate(params.data(), failingResiduals.data(), failingJacobian.data()));
    CHECK_FALSE(failingAdapter.Evaluate(params.data(), failingResiduals.data(), nullptr));
    CHECK(failingAdapter.ResidualCalls() == 2);
    CHECK(failingAdapter.JacobianCalls() == 1);

    // Counters are per adapter instance.
    CHECK(adapter.ResidualCalls() == 2);
    CHECK(adapter.JacobianCalls() == 2);
}

TEST_CASE("LeastSquaresLMAdapter with recoverNonFinite returns weighted non-finite outputs and still records cost errors", "[least-squares][lm-adapter]")
{
    ScriptedCost cost;
    cost.nanResidual = 0;
    std::vector<Operon::Scalar> const weights { 4, 9, 16 };
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, weights, true };
    std::array<Operon::Scalar, 2> params { 0, 0 };
    std::vector<Operon::Scalar> residuals(3);
    std::vector<Operon::Scalar> jacobian(6);

    REQUIRE(adapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
    CHECK_FALSE(adapter.Error().has_value());
    CHECK(std::isnan(static_cast<double>(residuals[0]))); // returned to the solver, not poisoned or rejected
    CHECK(residuals[1] == Operon::Scalar { 6 }); // 2 * sqrt(9)
    CHECK(residuals[2] == Operon::Scalar { 8 }); // 2 * sqrt(16)
    // column-major 3x2: rows scaled by sqrt(w_i)
    for (std::size_t j = 0; j < 2; ++j) {
        CHECK(jacobian[(j * 3) + 0] == Operon::Scalar { 2 });
        CHECK(jacobian[(j * 3) + 1] == Operon::Scalar { 3 });
        CHECK(jacobian[(j * 3) + 2] == Operon::Scalar { 4 });
    }
    CHECK(adapter.ResidualCalls() == 1);
    CHECK(adapter.JacobianCalls() == 1);

    // An error returned by the cost itself is still a recorded failure, and the first one is kept.
    cost.failWithError = true;
    CHECK_FALSE(adapter.Evaluate(params.data(), residuals.data(), jacobian.data()));
    REQUIRE(adapter.Error().has_value());
    CHECK(adapter.Error()->Code == Operon::LeastSquaresErrorCode::EvaluationFailure);
    CHECK(adapter.Error()->Row == 99);
    for (auto r : residuals) { CHECK(std::isnan(static_cast<double>(r))); }
    for (auto j : jacobian) { CHECK(std::isnan(static_cast<double>(j))); }
}
