// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <vector>

#include "operon/core/concepts.hpp"
#include "operon/optimizer/gaussian_gradient_cost.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/statistical_concepts.hpp"


#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/optimizer/gradient_cost.hpp"

namespace {

// f(x) = 0.5 * sum((x_i - target_i)^2), grad = x - target. No interpreter,
// likelihood, or Fisher method -- a purely numerical cost with no
// statistical interpretation.
class QuadraticBowlCost final : public Operon::GradientCostFunction {
public:
    using Scalar = Operon::Scalar;

    explicit QuadraticBowlCost(std::vector<Operon::Scalar> target)
        : target_(std::move(target))
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return target_.size(); }

    [[nodiscard]] auto Evaluate(
        Operon::ConstScalarSpan parameters,
        Operon::ScalarSpan gradient) const
        -> tl::expected<Operon::Scalar, Operon::GradientError> override
    {
        if (parameters.size() != target_.size()) {
            return tl::unexpected(Operon::GradientError {
                .Code = Operon::GradientErrorCode::InvalidShape, .Expected = target_.size(), .Actual = parameters.size() });
        }
        if (gradient.size() != target_.size()) {
            return tl::unexpected(Operon::GradientError {
                .Code = Operon::GradientErrorCode::InvalidShape, .Expected = target_.size(), .Actual = gradient.size() });
        }
        double cost = 0.0;
        for (std::size_t i = 0; i < target_.size(); ++i) {
            auto const diff = static_cast<double>(parameters[i]) - static_cast<double>(target_[i]);
            cost += 0.5 * diff * diff;
            gradient[i] = static_cast<Operon::Scalar>(diff);
        }
        return static_cast<Operon::Scalar>(cost);
    }

private:
    std::vector<Operon::Scalar> target_;
};

// Always fails; exercises the error path without any statistical method.
class FailingGradientCost final : public Operon::GradientCostFunction {
public:
    using Scalar = Operon::Scalar;

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t override { return 2; }

    [[nodiscard]] auto Evaluate(
        Operon::ConstScalarSpan /*parameters*/,
        Operon::ScalarSpan /*gradient*/) const
        -> tl::expected<Operon::Scalar, Operon::GradientError> override
    {
        return tl::unexpected(Operon::GradientError { .Code = Operon::GradientErrorCode::EvaluationFailure });
    }
};

// Regression guard: a numerical-only cost with no likelihood/Fisher method
// must satisfy Concepts::GradientCost. Adding a statistical requirement to
// this concept would break this assertion.
static_assert(Operon::Concepts::GradientCost<QuadraticBowlCost>);
// The numerical/statistical boundary is part of the contract: no gradient
// cost — synthetic or shipped — is a likelihood or Fisher producer, and a
static_assert(!Operon::Concepts::Likelihood<Operon::GaussianGradientCostFunction<Operon::Scalar>>);
static_assert(!Operon::Concepts::HasFisherDiagonal<Operon::GaussianGradientCostFunction<Operon::Scalar>>);
static_assert(!Operon::Concepts::GradientCost<Operon::GaussianLikelihood<Operon::Scalar>>);
static_assert(Operon::Concepts::Likelihood<Operon::GaussianLikelihood<Operon::Scalar>>);
static_assert(Operon::Concepts::HasFisherDiagonal<Operon::GaussianLikelihood<Operon::Scalar>>);

} // namespace

TEST_CASE("GradientCostFunction: objective and gradient match a known quadratic minimum", "[gradient-cost]")
{
    QuadraticBowlCost cost({1.0, -2.0, 0.5});
    std::vector<Operon::Scalar> parameters { 2.0, -2.0, 3.0 };
    std::vector<Operon::Scalar> gradient(3);

    auto result = cost.Evaluate(parameters, gradient);
    REQUIRE(result.has_value());
    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinAbs(0.5 * (1.0 + 0.0 + 6.25), 1e-9));
    CHECK_THAT(static_cast<double>(gradient[0]), Catch::Matchers::WithinAbs(1.0, 1e-6));
    CHECK_THAT(static_cast<double>(gradient[1]), Catch::Matchers::WithinAbs(0.0, 1e-6));
    CHECK_THAT(static_cast<double>(gradient[2]), Catch::Matchers::WithinAbs(2.5, 1e-6));
}

TEST_CASE("GradientCostFunction: at the target the objective and gradient are zero", "[gradient-cost]")
{
    QuadraticBowlCost cost({1.0, -2.0, 0.5});
    std::vector<Operon::Scalar> parameters { 1.0, -2.0, 0.5 };
    std::vector<Operon::Scalar> gradient(3);

    auto result = cost.Evaluate(parameters, gradient);
    REQUIRE(result.has_value());
    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinAbs(0.0, 1e-9));
    for (auto g : gradient) { CHECK_THAT(static_cast<double>(g), Catch::Matchers::WithinAbs(0.0, 1e-9)); }
}

TEST_CASE("GradientCostFunction: invalid parameter/gradient shapes are rejected", "[gradient-cost]")
{
    QuadraticBowlCost cost({1.0, -2.0});

    SECTION("wrong parameter count") {
        std::vector<Operon::Scalar> parameters { 1.0 };
        std::vector<Operon::Scalar> gradient(2);
        auto result = cost.Evaluate(parameters, gradient);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::GradientErrorCode::InvalidShape);
    }

    SECTION("wrong gradient count") {
        std::vector<Operon::Scalar> parameters { 1.0, 2.0 };
        std::vector<Operon::Scalar> gradient(1);
        auto result = cost.Evaluate(parameters, gradient);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::GradientErrorCode::InvalidShape);
    }
}

TEST_CASE("GradientCostFunction: a failing cost reports its typed error", "[gradient-cost]")
{
    FailingGradientCost cost;
    std::vector<Operon::Scalar> parameters { 0, 0 };
    std::vector<Operon::Scalar> gradient(2);

    auto result = cost.Evaluate(parameters, gradient);
    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().Code == Operon::GradientErrorCode::EvaluationFailure);
    CHECK_FALSE(result.error().Cause.has_value());
}
