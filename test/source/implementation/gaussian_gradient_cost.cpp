// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

// Covers GaussianGradientCostFunction: unweighted/scalar/per-row WLS
// objective and exact gradient, range offsets, the minibatch contract,
// interpreter errors, counters, and objective/gradient equality with
// ComputeGradient over the same raw residual/Jacobian.

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/optimizer/gaussian_gradient_cost.hpp"
#include "operon/optimizer/interpreter_gradient_cost.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"

namespace {

using Extents = std::dextents<std::size_t, 2>;
using Mapping = std::layout_stride::mapping<Extents>;

// y = X1 + X2 + X3 (linear, unique solution w1=w2=w3=1). Variable weights
// start at 0.2: a well-conditioned linear problem with 3 optimizable
// coefficients.
struct Fixture {
    static constexpr auto Nrow { 40 };
    static constexpr auto Ncol { 4 };

    Operon::RandomGenerator rng { 0 }; // NOLINT(readability-identifier-naming)
    Operon::Dataset ds; // NOLINT(readability-identifier-naming)
    Operon::Tree tree; // NOLINT(readability-identifier-naming)
    using DTable = Operon::DispatchTable<Operon::Scalar>;
    DTable dtable; // NOLINT(readability-identifier-naming)

    Fixture()
        : ds([&]() -> Operon::Dataset {
            std::vector<std::vector<Operon::Scalar>> cols(Ncol, std::vector<Operon::Scalar>(Nrow));
            for (auto j = 0; j < Ncol - 1; ++j) {
                for (auto i = 0; i < Nrow; ++i) {
                    cols[j][i] = Operon::Random::Uniform(rng, -1.0F, +1.0F);
                }
            }
            for (auto i = 0; i < Nrow; ++i) {
                cols[Ncol - 1][i] = cols[0][i] + cols[1][i] + cols[2][i];
            }
            return Operon::Dataset(cols);
        }())
        , tree([&]() -> Operon::Tree {
            auto t = Operon::InfixParser::ParseOrThrow("X1 + X2 + X3", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) {
                    node.Value = Operon::Scalar { 0.2 };
                }
            }
            return t;
        }())
    {
    }
};

// Reference reduction: raw residual (prediction - target) and Jacobian
// through the interpreter directly, then Operon::ComputeGradient -- the
// same numerical primitive GaussianGradientCostFunction itself uses.
auto ReferenceCost(Operon::Interpreter<Operon::Scalar, Fixture::DTable> const& interpreter,
    Operon::ConstScalarSpan params, Operon::ConstScalarSpan target, Operon::Range range,
    Operon::ConstScalarSpan weights, std::vector<Operon::Scalar>& gradient) -> Operon::AccumulationScalar
{
    auto const n = range.Size();
    auto const p = params.size();
    std::vector<Operon::Scalar> residuals(n);
    REQUIRE(interpreter.Evaluate(params, range, residuals).has_value());
    for (std::size_t i = 0; i < n; ++i) {
        residuals[i] -= target[range.Start() + i];
    }
    std::vector<Operon::Scalar> jacBuffer(n * p);
    Operon::ScalarMatrixView jac { jacBuffer.data(),
        Mapping { Extents { n, p }, std::array<std::size_t, 2> { 1, n } } };
    REQUIRE(interpreter.JacRev(params, range, jacBuffer).has_value());
    auto result = Operon::ComputeGradient(residuals, jac, gradient, weights);
    REQUIRE(result.has_value());
    return *result;
}

static_assert(Operon::Concepts::GradientCost<Operon::GaussianGradientCostFunction>);
static_assert(Operon::Concepts::InterpreterGradientCost<Operon::GaussianGradientCostFunction>);
// Gaussian sample weights are numerical WLS weights: the optimizers forward
// the dataset's weights to this cost.
static_assert(Operon::GaussianGradientCostFunction::UsesDatasetWeights);

} // namespace

TEST_CASE(
    "GaussianGradientCostFunction: unweighted objective and gradient match ComputeGradient", "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };

    Operon::GaussianGradientCostFunction cost { &interpreter, target, range };
    auto params = fix.tree.GetCoefficients();
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);
    REQUIRE(result.has_value());

    std::vector<Operon::Scalar> expectedGradient(params.size());
    auto expectedCost = ReferenceCost(interpreter, params, target, range, {}, expectedGradient);

    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(static_cast<double>(expectedCost), 1e-3));
    for (std::size_t i = 0; i < gradient.size(); ++i) {
        CHECK_THAT(static_cast<double>(gradient[i]),
            Catch::Matchers::WithinRel(static_cast<double>(expectedGradient[i]), 1e-3));
    }
    CHECK(cost.FunctionEvaluations() == 1);
    CHECK(cost.JacobianEvaluations() == 1);
}

TEST_CASE("GaussianGradientCostFunction: scalar weight matches ComputeGradient with a broadcast weight",
    "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };
    std::array<Operon::Scalar, 1> const weight { Operon::Scalar { 2.5 } };

    Operon::GaussianGradientCostFunction cost { &interpreter, target, range, nullptr, 0, weight };
    auto params = fix.tree.GetCoefficients();
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);
    REQUIRE(result.has_value());

    std::vector<Operon::Scalar> expectedGradient(params.size());
    auto expectedCost = ReferenceCost(interpreter, params, target, range, weight, expectedGradient);

    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(static_cast<double>(expectedCost), 1e-3));
    for (std::size_t i = 0; i < gradient.size(); ++i) {
        CHECK_THAT(static_cast<double>(gradient[i]),
            Catch::Matchers::WithinRel(static_cast<double>(expectedGradient[i]), 1e-3));
    }
}

TEST_CASE("GaussianGradientCostFunction: per-row weight matches ComputeGradient with the same weights",
    "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };
    std::vector<Operon::Scalar> weights(Fixture::Nrow);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = Operon::Scalar { 1 } + (Operon::Scalar { 0.05 } * static_cast<Operon::Scalar>(i));
    }

    Operon::GaussianGradientCostFunction cost { &interpreter, target, range, nullptr, 0, weights };
    auto params = fix.tree.GetCoefficients();
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);
    REQUIRE(result.has_value());

    std::vector<Operon::Scalar> expectedGradient(params.size());
    auto expectedCost = ReferenceCost(interpreter, params, target, range, weights, expectedGradient);

    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(static_cast<double>(expectedCost), 1e-3));
    for (std::size_t i = 0; i < gradient.size(); ++i) {
        CHECK_THAT(static_cast<double>(gradient[i]),
            Catch::Matchers::WithinRel(static_cast<double>(expectedGradient[i]), 1e-3));
    }
}

TEST_CASE("GaussianGradientCostFunction: a non-zero range offset is honored", "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 10, Fixture::Nrow };

    Operon::GaussianGradientCostFunction cost { &interpreter, target, range };
    auto params = fix.tree.GetCoefficients();
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);
    REQUIRE(result.has_value());

    std::vector<Operon::Scalar> expectedGradient(params.size());
    auto expectedCost = ReferenceCost(interpreter, params, target, range, {}, expectedGradient);

    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(static_cast<double>(expectedCost), 1e-3));
    for (std::size_t i = 0; i < gradient.size(); ++i) {
        CHECK_THAT(static_cast<double>(gradient[i]),
            Catch::Matchers::WithinRel(static_cast<double>(expectedGradient[i]), 1e-3));
    }
}

TEST_CASE("GaussianGradientCostFunction: batchSize==0 always evaluates the full range", "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };

    // No rng supplied and batchSize left at its 0 default: never touches
    // the RNG, so a null pointer is safe here.
    Operon::GaussianGradientCostFunction cost { &interpreter, target, range };
    auto params = fix.tree.GetCoefficients();
    std::vector<Operon::Scalar> firstGradient(params.size());
    std::vector<Operon::Scalar> secondGradient(params.size());
    auto first = cost.Evaluate(params, firstGradient);
    auto second = cost.Evaluate(params, secondGradient);
    REQUIRE(first.has_value());
    REQUIRE(second.has_value());
    CHECK(*first == *second);
    CHECK(firstGradient == secondGradient);
}

TEST_CASE("GaussianGradientCostFunction: a nonzero batch size selects a reproducible random subrange",
    "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };
    auto params = fix.tree.GetCoefficients();

    Operon::RandomGenerator rngA { 7 };
    Operon::GaussianGradientCostFunction costA { &interpreter, target, range, &rngA, 10 };
    std::vector<Operon::Scalar> gradientA(params.size());
    auto resultA = costA.Evaluate(params, gradientA);
    REQUIRE(resultA.has_value());

    Operon::RandomGenerator rngB { 7 };
    Operon::GaussianGradientCostFunction costB { &interpreter, target, range, &rngB, 10 };
    std::vector<Operon::Scalar> gradientB(params.size());
    auto resultB = costB.Evaluate(params, gradientB);
    REQUIRE(resultB.has_value());

    CHECK(*resultA == *resultB);
    CHECK(gradientA == gradientB);
}

TEST_CASE("GaussianGradientCostFunction: a minibatch of an offset range uses matching absolute target and weight rows",
    "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 5, Fixture::Nrow };
    constexpr std::size_t batchSize = 10;
    std::vector<Operon::Scalar> weights(Fixture::Nrow);
    for (std::size_t i = 0; i < weights.size(); ++i) {
        weights[i] = Operon::Scalar { 1 } + (Operon::Scalar { 0.3 } * static_cast<Operon::Scalar>(i));
    }
    auto params = fix.tree.GetCoefficients();

    Operon::RandomGenerator rng { 11 };
    Operon::GaussianGradientCostFunction cost { &interpreter, target, range, &rng, batchSize, weights };
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);
    REQUIRE(result.has_value());

    // The batch is some contiguous batchSize-row subrange inside the offset
    // range; the result must equal the reference cost and gradient (with the
    // matching absolute target and weight rows) for one of the possible starts.
    auto const close = [](double a, double b) { return std::abs(a - b) <= 1e-3 * std::max(std::abs(a), std::abs(b)); };
    bool matched = false;
    for (std::size_t offset = 0; offset <= range.Size() - batchSize && !matched; ++offset) {
        Operon::Range batch { range.Start() + offset, range.Start() + offset + batchSize };
        std::vector<Operon::Scalar> expectedGradient(params.size());
        auto expectedCost = ReferenceCost(interpreter, params, target, batch,
            Operon::ConstScalarSpan { weights }.subspan(batch.Start(), batch.Size()), expectedGradient);
        matched = close(static_cast<double>(*result), static_cast<double>(expectedCost));
        for (std::size_t i = 0; matched && i < gradient.size(); ++i) {
            matched = close(static_cast<double>(gradient[i]), static_cast<double>(expectedGradient[i]));
        }
    }
    CHECK(matched);
}

TEST_CASE(
    "GaussianGradientCostFunction: interpreter failures are typed with the original cause", "[gaussian-gradient-cost]")
{
    Fixture fix;
    constexpr auto missingVariable = Operon::Hash { 0xBADF00D };
    auto const variableTree = Operon::Tree({ Operon::Node { Operon::NodeType::Variable, missingVariable } });
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &variableTree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };

    Operon::GaussianGradientCostFunction cost { &interpreter, target, range };
    std::vector<Operon::Scalar> params(cost.NumParameters());
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().Code == Operon::GradientErrorCode::EvaluationFailure);
    REQUIRE(result.error().Cause.has_value());
    CHECK(result.error().Cause->Kind == Operon::InterpreterError::Code::MissingVariable);
    CHECK(result.error().Cause->Hash == missingVariable);
    REQUIRE(cost.Error().has_value());
    CHECK(cost.Error()->Code == Operon::GradientErrorCode::EvaluationFailure);
    for (auto g : gradient) {
        CHECK(std::isnan(static_cast<double>(g)));
    }
    CHECK(cost.FunctionEvaluations() == 1);
    CHECK(cost.JacobianEvaluations() == 0);
}

TEST_CASE("GaussianGradientCostFunction: FunctionEvaluations and JacobianEvaluations count Evaluate calls",
    "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };

    Operon::GaussianGradientCostFunction cost { &interpreter, target, range };
    auto params = fix.tree.GetCoefficients();
    std::vector<Operon::Scalar> gradient(params.size());

    CHECK(cost.FunctionEvaluations() == 0);
    CHECK(cost.JacobianEvaluations() == 0);
    for (int i = 0; i < 3; ++i) {
        auto result = cost.Evaluate(params, gradient);
        REQUIRE(result.has_value());
    }
    CHECK(cost.FunctionEvaluations() == 3);
    CHECK(cost.JacobianEvaluations() == 3);
}

TEST_CASE("GaussianGradientCostFunction: only successful evaluations count as Jacobian evaluations",
    "[gaussian-gradient-cost]")
{
    Fixture fix;
    constexpr auto missingVariable = Operon::Hash { 0xBADF00D };
    auto const variableTree = Operon::Tree({ Operon::Node { Operon::NodeType::Variable, missingVariable } });
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> failingInterpreter { &fix.dtable, &fix.ds, &variableTree };
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };

    Operon::GaussianGradientCostFunction failing { &failingInterpreter, target, range };
    std::vector<Operon::Scalar> failingParams(failing.NumParameters());
    std::vector<Operon::Scalar> failingGradient(failingParams.size());
    REQUIRE_FALSE(failing.Evaluate(failingParams, failingGradient).has_value());
    CHECK(failing.JacobianEvaluations() == 0);

    Operon::GaussianGradientCostFunction cost { &interpreter, target, range };
    auto params = fix.tree.GetCoefficients();
    std::vector<Operon::Scalar> gradient(params.size());
    REQUIRE(cost.Evaluate(params, gradient).has_value());
    CHECK(cost.JacobianEvaluations() == 1);
}

TEST_CASE("GaussianGradientCostFunction: invalid weights are typed InvalidWeights errors, not assertions",
    "[gaussian-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, Fixture::DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto target = fix.ds.GetValues("X4");
    Operon::Range range { 0, Fixture::Nrow };
    auto const params = fix.tree.GetCoefficients();

    SECTION("scalar weight that is NaN, infinite, or negative")
    {
        for (auto const weight : { std::numeric_limits<Operon::Scalar>::quiet_NaN(),
                 std::numeric_limits<Operon::Scalar>::infinity(), Operon::Scalar { -1 } }) {
            std::array<Operon::Scalar, 1> weights { weight };
            Operon::GaussianGradientCostFunction cost { &interpreter, target, range, nullptr, 0, weights };
            auto gradient = std::vector<Operon::Scalar>(cost.NumParameters());
            auto result = cost.Evaluate(params, gradient);
            REQUIRE_FALSE(result.has_value());
            CHECK(result.error().Code == Operon::GradientErrorCode::InvalidWeights);
            CHECK(result.error().Row == 0);
            for (auto g : gradient) {
                CHECK(std::isnan(static_cast<double>(g)));
            }
        }
    }

    SECTION("per-row weights of the wrong size report expected and actual sizes")
    {
        std::vector<Operon::Scalar> weights(7, Operon::Scalar { 1 });
        Operon::GaussianGradientCostFunction cost { &interpreter, target, range, nullptr, 0, weights };
        auto gradient = std::vector<Operon::Scalar>(cost.NumParameters());
        auto result = cost.Evaluate(params, gradient);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::GradientErrorCode::InvalidWeights);
        CHECK(result.error().Expected == static_cast<std::size_t>(Fixture::Nrow));
        CHECK(result.error().Actual == weights.size());
    }

    SECTION("a per-row violation reports the absolute row; rows outside the range are not read")
    {
        Operon::Range offsetRange { 5, Fixture::Nrow };
        std::vector<Operon::Scalar> weights(Fixture::Nrow, Operon::Scalar { 1 });
        weights[2] = Operon::Scalar { -1 }; // before the range: ignored
        {
            Operon::GaussianGradientCostFunction cost { &interpreter, target, offsetRange, nullptr, 0, weights };
            auto gradient = std::vector<Operon::Scalar>(cost.NumParameters());
            REQUIRE(cost.Evaluate(params, gradient).has_value());
        }
        weights[12] = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        Operon::GaussianGradientCostFunction cost { &interpreter, target, offsetRange, nullptr, 0, weights };
        auto gradient = std::vector<Operon::Scalar>(cost.NumParameters());
        auto result = cost.Evaluate(params, gradient);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::GradientErrorCode::InvalidWeights);
        CHECK(result.error().Row == 12);
        REQUIRE(cost.Error().has_value());
        CHECK(cost.Error()->Code == Operon::GradientErrorCode::InvalidWeights);
    }
}
