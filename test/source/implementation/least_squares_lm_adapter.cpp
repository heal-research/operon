// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <array>
#include <cmath>
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
    [[nodiscard]] auto NumResiduals() const noexcept -> std::size_t { return x_.size(); }

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
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, cost.NumResiduals() };

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
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, cost.NumResiduals() };

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
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, 5 };

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
    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> adapter { &cost, cost.NumResiduals() };
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

    Operon::LeastSquaresLMAdapter<Eigen::ColMajor> colMajorAdapter { &cost, n };
    Operon::LeastSquaresLMAdapter<Eigen::RowMajor> rowMajorAdapter { &cost, n };

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
