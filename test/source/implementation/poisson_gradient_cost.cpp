// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

// Covers PoissonGradientCostFunction: exact NLL (including the
// coefficient-independent lgamma term) and exact chain-rule gradient for
// both LogInput modes, exposure broadcast/per-row scaling, the LogInput=false
// domain check, and finite-difference gradient agreement.

#include <array>
#include <cmath>
#include <limits>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/optimizer/interpreter_gradient_cost.hpp"
#include "operon/optimizer/poisson_gradient_cost.hpp"
#include "operon/parser/infix.hpp"

namespace {

using DTable = Operon::DispatchTable<Operon::Scalar>;

// X1 in [0.1, 0.9], Y = small nonnegative counts. tree = a*X1 + b: usable as
// a LogInput=true log-rate model directly, and as a LogInput=false rate
// model when a, b are chosen to keep a*X1+b strictly positive over X1.
struct Fixture {
    Operon::Dataset ds; // NOLINT(readability-identifier-naming)
    Operon::Tree tree; // NOLINT(readability-identifier-naming)
    DTable dtable; // NOLINT(readability-identifier-naming)

    Fixture()
        : ds([&]() -> Operon::Dataset {
            std::vector<Operon::Scalar> x { 0.1F, 0.3F, 0.5F, 0.7F, 0.9F };
            std::vector<Operon::Scalar> y { 1.0F, 2.0F, 0.0F, 3.0F, 1.0F };
            return Operon::Dataset({ "X1", "Y" }, { x, y });
        }())
        , tree([&]() -> Operon::Tree {
            auto t = Operon::InfixParser::ParseOrThrow("1.0 * X1 + 1.0", ds);
            for (auto& node : t.Nodes()) {
                node.Optimize = node.IsConstant();
            }
            return t;
        }())
    {
    }

    [[nodiscard]] auto Target() const -> Operon::ConstScalarSpan { return ds.GetValues("Y"); }
    [[nodiscard]] auto TrainingRange() const -> Operon::Range { return { 0, static_cast<std::size_t>(ds.Rows()) }; }
};

// Reference NLL/gradient computed directly from the closed-form formulas,
// independent of PoissonGradientCostFunction's own implementation.
template <bool LogInput>
auto ReferenceNll(Operon::ConstScalarSpan z, Operon::ConstScalarSpan y, Operon::ConstScalarSpan exposure) -> double
{
    double nll = 0.0;
    for (std::size_t i = 0; i < z.size(); ++i) {
        auto const e = exposure.empty() ? 1.0 : static_cast<double>(exposure.size() == 1 ? exposure[0] : exposure[i]);
        auto const yi = static_cast<double>(y[i]);
        if constexpr (LogInput) {
            auto const eta = e * static_cast<double>(z[i]);
            nll += std::exp(eta) - (yi * eta) + std::lgamma(yi + 1.0);
        } else {
            auto const mu = e * static_cast<double>(z[i]);
            nll += mu - (yi * std::log(mu)) + std::lgamma(yi + 1.0);
        }
    }
    return nll;
}

static_assert(Operon::Concepts::GradientCost<Operon::PoissonGradientCostFunction<true>>);
static_assert(Operon::Concepts::GradientCost<Operon::PoissonGradientCostFunction<false>>);
static_assert(Operon::Concepts::InterpreterGradientCost<Operon::PoissonGradientCostFunction<true>>);
static_assert(Operon::Concepts::InterpreterGradientCost<Operon::PoissonGradientCostFunction<false>>);
// Exposure is an explicit Poisson parameter, never ordinary dataset sample
// weights.
static_assert(!Operon::PoissonGradientCostFunction<true>::UsesDatasetWeights);
static_assert(!Operon::PoissonGradientCostFunction<false>::UsesDatasetWeights);

} // namespace

TEST_CASE("PoissonGradientCostFunction LogInput=true: NLL matches the closed-form log-rate formula",
    "[poisson-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    Operon::PoissonGradientCostFunction<true> cost { &interpreter, fix.Target(), fix.TrainingRange() };

    std::vector<Operon::Scalar> params { 0.6F, -0.3F };
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);
    REQUIRE(result.has_value());

    auto z = interpreter.Evaluate(params, fix.TrainingRange()).value();
    auto expected = ReferenceNll<true>(Operon::ConstScalarSpan { z.data(), z.size() }, fix.Target(), {});
    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(expected, 1e-4));
}

TEST_CASE("PoissonGradientCostFunction LogInput=false: NLL matches the closed-form positive-mean formula",
    "[poisson-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    Operon::PoissonGradientCostFunction<false> cost { &interpreter, fix.Target(), fix.TrainingRange() };

    // a=1.5, b=2 keeps a*X1+b strictly positive over X1 in [0.1, 0.9].
    std::vector<Operon::Scalar> params { 1.5F, 2.0F };
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);
    REQUIRE(result.has_value());

    auto z = interpreter.Evaluate(params, fix.TrainingRange()).value();
    auto expected = ReferenceNll<false>(Operon::ConstScalarSpan { z.data(), z.size() }, fix.Target(), {});
    CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(expected, 1e-4));
}

TEST_CASE(
    "PoissonGradientCostFunction LogInput=true: exposure broadcast and per-row scaling match the closed-form formula",
    "[poisson-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    std::vector<Operon::Scalar> params { 0.4F, 0.1F };
    auto z = interpreter.Evaluate(params, fix.TrainingRange()).value();
    Operon::ConstScalarSpan zSpan { z.data(), z.size() };

    SECTION("scalar exposure broadcasts")
    {
        std::array<Operon::Scalar, 1> const exposure { Operon::Scalar { 2 } };
        Operon::PoissonGradientCostFunction<true> cost { &interpreter, fix.Target(), fix.TrainingRange(), nullptr, 0,
            exposure };
        std::vector<Operon::Scalar> gradient(params.size());
        auto result = cost.Evaluate(params, gradient);
        REQUIRE(result.has_value());
        auto expected = ReferenceNll<true>(zSpan, fix.Target(), exposure);
        CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(expected, 1e-4));
    }

    SECTION("per-row exposure")
    {
        std::vector<Operon::Scalar> exposure { 1.0F, 1.5F, 0.5F, 2.0F, 1.2F };
        Operon::PoissonGradientCostFunction<true> cost { &interpreter, fix.Target(), fix.TrainingRange(), nullptr, 0,
            exposure };
        std::vector<Operon::Scalar> gradient(params.size());
        auto result = cost.Evaluate(params, gradient);
        REQUIRE(result.has_value());
        auto expected = ReferenceNll<true>(zSpan, fix.Target(), exposure);
        CHECK_THAT(static_cast<double>(*result), Catch::Matchers::WithinRel(expected, 1e-4));
    }
}

TEST_CASE("PoissonGradientCostFunction: gradient matches central finite differences", "[poisson-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    constexpr double Eps = 1e-3;

    auto checkFiniteDifference = [&](auto& cost, std::vector<Operon::Scalar> const& params) {
        std::vector<Operon::Scalar> gradient(params.size());
        auto result = cost.Evaluate(params, gradient);
        REQUIRE(result.has_value());

        for (std::size_t k = 0; k < params.size(); ++k) {
            auto plus = params;
            auto minus = params;
            plus[k] += static_cast<Operon::Scalar>(Eps);
            minus[k] -= static_cast<Operon::Scalar>(Eps);
            std::vector<Operon::Scalar> scratch(params.size());
            auto fPlus = cost.Evaluate(plus, scratch);
            auto fMinus = cost.Evaluate(minus, scratch);
            REQUIRE(fPlus.has_value());
            REQUIRE(fMinus.has_value());
            auto const fd = (static_cast<double>(*fPlus) - static_cast<double>(*fMinus)) / (2.0 * Eps);
            CHECK_THAT(static_cast<double>(gradient[k]), Catch::Matchers::WithinRel(fd, 1e-2));
        }
    };

    SECTION("LogInput=true")
    {
        Operon::PoissonGradientCostFunction<true> cost { &interpreter, fix.Target(), fix.TrainingRange() };
        checkFiniteDifference(cost, { 0.5F, -0.2F });
    }

    SECTION("LogInput=false")
    {
        Operon::PoissonGradientCostFunction<false> cost { &interpreter, fix.Target(), fix.TrainingRange() };
        checkFiniteDifference(cost, { 1.5F, 2.0F });
    }
}

TEST_CASE(
    "PoissonGradientCostFunction LogInput=false: a nonpositive mean is a typed evaluation error, not a silent NaN",
    "[poisson-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    Operon::PoissonGradientCostFunction<false> cost { &interpreter, fix.Target(), fix.TrainingRange() };

    // Both coefficients negative: since X1 > 0 for every row, the tree's
    // output is negative regardless of which slot multiplies X1 and which
    // is additive.
    std::vector<Operon::Scalar> params { -3.0F, -3.0F };
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().Code == Operon::GradientErrorCode::NonFiniteEvaluation);
    CHECK_FALSE(result.error().Cause.has_value());
    for (auto g : gradient) {
        CHECK(std::isnan(static_cast<double>(g)));
    }
}

TEST_CASE(
    "PoissonGradientCostFunction: interpreter failures are typed with the original cause", "[poisson-gradient-cost]")
{
    Fixture fix;
    constexpr auto missingVariable = Operon::Hash { 0xBADF00D };
    auto const variableTree = Operon::Tree({ Operon::Node { Operon::NodeType::Variable, missingVariable } });
    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &variableTree };

    Operon::PoissonGradientCostFunction<> cost { &interpreter, fix.Target(), fix.TrainingRange() };
    std::vector<Operon::Scalar> params(cost.NumParameters());
    std::vector<Operon::Scalar> gradient(params.size());
    auto result = cost.Evaluate(params, gradient);

    REQUIRE_FALSE(result.has_value());
    CHECK(result.error().Code == Operon::GradientErrorCode::EvaluationFailure);
    REQUIRE(result.error().Cause.has_value());
    CHECK(result.error().Cause->Kind == Operon::InterpreterError::Code::MissingVariable);
    CHECK(result.error().Cause->Hash == missingVariable);
    for (auto g : gradient) {
        CHECK(std::isnan(static_cast<double>(g)));
    }
}

TEST_CASE("PoissonGradientCostFunction: invalid exposure is a typed InvalidWeights error, not an assertion",
    "[poisson-gradient-cost]")
{
    Fixture fix;
    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    std::vector<Operon::Scalar> params { 0.5F, -0.2F };
    std::vector<Operon::Scalar> gradient(params.size());

    SECTION("wrong per-row size")
    {
        std::vector<Operon::Scalar> exposure { 1.0F, 1.0F };
        Operon::PoissonGradientCostFunction<true> cost { &interpreter, fix.Target(), fix.TrainingRange(), nullptr, 0,
            exposure };
        auto result = cost.Evaluate(params, gradient);
        REQUIRE_FALSE(result.has_value());
        CHECK(result.error().Code == Operon::GradientErrorCode::InvalidWeights);
        CHECK(result.error().Expected == fix.TrainingRange().Size());
        CHECK(result.error().Actual == exposure.size());
        for (auto g : gradient) {
            CHECK(std::isnan(static_cast<double>(g)));
        }
    }

    SECTION("negative or non-finite entry reports its row")
    {
        std::vector<Operon::Scalar> exposure { 1.0F, 1.0F, 1.0F, -0.5F, 1.0F };
        Operon::PoissonGradientCostFunction<true> negative { &interpreter, fix.Target(), fix.TrainingRange(), nullptr,
            0, exposure };
        auto negativeResult = negative.Evaluate(params, gradient);
        REQUIRE_FALSE(negativeResult.has_value());
        CHECK(negativeResult.error().Code == Operon::GradientErrorCode::InvalidWeights);
        CHECK(negativeResult.error().Row == 3);

        exposure[3] = std::numeric_limits<Operon::Scalar>::infinity();
        Operon::PoissonGradientCostFunction<true> infinite { &interpreter, fix.Target(), fix.TrainingRange(), nullptr,
            0, exposure };
        auto infiniteResult = infinite.Evaluate(params, gradient);
        REQUIRE_FALSE(infiniteResult.has_value());
        CHECK(infiniteResult.error().Code == Operon::GradientErrorCode::InvalidWeights);
        CHECK(infiniteResult.error().Row == 3);
    }
}
