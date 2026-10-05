// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <memory>
#include <vector>

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/optimizer/gaussian_gradient_cost.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/poisson_gradient_cost.hpp"
#include "operon/optimizer/solvers/sgd.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"
#if defined(HAVE_ASMJIT)
#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#endif

namespace Operon::Test {

// Test problem: y = X1 + X2 + X3 (linear, unique solution w1=w2=w3=1).
// Variable weights initialised to 0.1 — a well-conditioned linear least squares
// problem. LM and L-BFGS should reach near-zero SSE and w=1.0 in few iterations;
// SGD should at least significantly reduce the cost.
struct OptimizerFixture {
    static constexpr auto Nrow { 500 };
    static constexpr auto Ncol { 4 }; // X1, X2, X3, y

    Operon::RandomGenerator rng { 0 }; // NOLINT(readability-identifier-naming)
    Eigen::Array<Operon::Scalar, -1, -1> data { Nrow, Ncol }; // NOLINT(readability-identifier-naming)
    Operon::Dataset ds; // NOLINT(readability-identifier-naming)
    Operon::Tree tree; // NOLINT(readability-identifier-naming)
    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable; // NOLINT(readability-identifier-naming)
    Operon::Problem problem; // NOLINT(readability-identifier-naming)

    OptimizerFixture()
        : ds([&]() -> Operon::Dataset {
            for (auto i = 0; i < Ncol - 1; ++i) {
                auto col = data.col(i);
                std::generate(
                    col.begin(), col.end(), [&]() -> float { return Operon::Random::Uniform(rng, -1.0F, +1.0F); });
            }
            data.col(Ncol - 1) = data.col(0) + data.col(1) + data.col(2);
            std::vector<std::vector<Operon::Scalar>> cols(Ncol);
            for (auto j = 0; j < Ncol; ++j) {
                cols[j].assign(data.col(j).data(), data.col(j).data() + Nrow);
            }
            return Operon::Dataset(cols);
        }())
        , tree([&]() -> Tree {
            auto t = InfixParser::ParseOrThrow("X1 + X2 + X3", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) {
                    node.Value = static_cast<Operon::Scalar>(0.1);
                }
            }
            return t;
        }())
        , problem(&ds)
    {
        problem.SetTrainingRange({ 0, Nrow });
        problem.SetTestRange({ 0, Nrow });
        problem.SetTarget("X4"); // last column: X1+X2+X3
    }
};

TEST_CASE("Gaussian likelihood static methods", "[likelihood]")
{
    using Lik = GaussianLikelihood<Operon::Scalar>;
    constexpr auto n { 100 };

    SECTION("perfect prediction, scalar sigma=1: NLL = n/2 * log(2pi)")
    {
        std::vector<Operon::Scalar> pred(n, 1.0F);
        std::vector<Operon::Scalar> target(n, 1.0F); // zero residuals
        std::vector<Operon::Scalar> sigma(1, 1.0F);
        auto nll = Lik::ComputeLikelihood(pred, target, sigma);
        auto expected = n / 2.0 * std::log(Operon::Math::Tau);
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(expected, 1e-5));
    }

    SECTION("known residuals, scalar sigma: NLL = n/2 * log(2pi*s2) + SSR/(2*s2)")
    {
        // pred = 1, target = 0  =>  eᵢ = 1, SSR = n
        std::vector<Operon::Scalar> pred(n, 1.0F);
        std::vector<Operon::Scalar> target(n, 0.0F);
        constexpr double s { 2.0 };
        std::vector<Operon::Scalar> sigma(1, static_cast<Operon::Scalar>(s));
        auto expected = 0.5 * (n * std::log(Operon::Math::Tau * s * s) + n / (s * s));
        auto nll = Lik::ComputeLikelihood(pred, target, sigma);
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(expected, 1e-5));
    }

    SECTION("Fisher diagonal shape and values: identity jacobian, scalar sigma")
    {
        // J = I (n×n), sigma = 2  =>  diag(F) = 1 / sigma^2 = 1/4
        using Extents = std::dextents<std::size_t, 2>;
        using Mapping = std::layout_stride::mapping<Extents>;
        auto const rows = static_cast<std::size_t>(n);
        std::vector<Operon::Scalar> pred(rows, 0.0F);
        std::vector<Operon::Scalar> jac(rows * rows, 0.0F);
        for (std::size_t i = 0; i < rows; ++i) {
            jac[(i * rows) + i] = 1.0F;
        }
        Operon::ConstScalarMatrixView const view { jac.data(),
            Mapping { Extents { rows, rows }, std::array<std::size_t, 2> { rows, 1 } } };
        std::vector<Operon::Scalar> sigma(1, 2.0F);
        std::vector<Operon::Scalar> diagonal(rows);
        REQUIRE(Lik::ComputeFisherDiagonal(pred, view, sigma, diagonal).has_value());
        for (auto const d : diagonal) {
            CHECK_THAT(static_cast<double>(d), Catch::Matchers::WithinRel(0.25, 1e-5));
        }
    }
}

TEST_CASE("Parameter optimization", "[optimizer]") // NOLINT(readability-function-cognitive-complexity)
{
    OptimizerFixture fix;
    auto& rng = fix.rng;
    auto& tree = fix.tree;
    auto& dtable = fix.dtable;
    auto& problem = fix.problem;
    using DTable = OptimizerFixture::DTable;

    // Linear problem: unique solution at w=1.0, SSE=0.
    // LM and L-BFGS should converge tightly; SGD improves but may not reach zero.
    constexpr Operon::Scalar tightTol { 1e-3F };
    constexpr Operon::Scalar looseTol { 0.5F };
    constexpr Operon::Scalar paramTol { 0.01F };

    auto checkExact = [&](OptimizerBase& optimizer) -> void {
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        CHECK(summary->FinalCost < summary->InitialCost);
        CHECK(summary->FinalCost < tightTol);
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    };

    auto checkImproved = [&](OptimizerBase& optimizer) -> void {
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        CHECK(summary->FinalCost < summary->InitialCost);
        CHECK(summary->FinalCost < looseTol);
    };

    SECTION("tiny solver")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> optimizer { &dtable, &problem };
        checkExact(optimizer);
    }

    SECTION("eigen solver")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer { &dtable, &problem };
        checkExact(optimizer);
    }

    SECTION("lbfgs / gaussian")
    {
        LBFGSOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem };
        checkExact(optimizer);
    }

    SECTION("lbfgs / poisson")
    {
        // Poisson loss on a continuous target: just verify it runs and improves
        LBFGSOptimizer<DTable, PoissonGradientCostFunction<>> const optimizer { &dtable, &problem };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        CHECK(std::isfinite(summary->FinalCost));
        CHECK(!summary->FinalParameters.empty());
    }

    SECTION("sgd / gaussian")
    {
        auto const dim { tree.CoefficientsCount() };
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem, *rule };
        checkImproved(optimizer);
    }

    SECTION("sgd / poisson")
    {
        auto const dim { tree.CoefficientsCount() };
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, PoissonGradientCostFunction<>> const optimizer { &dtable, &problem, *rule };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        CHECK(std::isfinite(summary->FinalCost));
        CHECK(!summary->FinalParameters.empty());
    }

#if defined(HAVE_ASMJIT)
    SECTION("jit tiny solver")
    {
        Operon::JIT::JitZobrist zobrist { rng, 50, problem.GetInputs() };
        JIT::JitEvaluator jitEval { &problem, &zobrist };
        JitLevenbergMarquardtOptimizer<DTable> optimizer { &dtable, &problem, &jitEval };
        checkExact(optimizer);
    }
#endif
}

TEST_CASE("Optimizers return typed interpreter errors", "[optimizer][interpreter]")
{
    OptimizerFixture fix;
    using DTable = OptimizerFixture::DTable;
    constexpr auto missingVariable = Operon::Hash { 0xBADF00D };
    constexpr auto missingPrimitive = Operon::Hash { 0xDEADBEEF };
    auto const variableTree = Operon::Tree({ Operon::Node { Operon::NodeType::Variable, missingVariable } });
    auto const primitiveTree = Operon::Tree({
        Operon::Node::Constant(1),
        Operon::Node::Constant(2),
        Operon::Node::Function(missingPrimitive, 2),
    });

    auto check = [&](auto const& optimizer, Operon::Tree const& tree, InterpreterError::Code expected) {
        auto outcome = optimizer.Optimize(fix.rng, tree);
        REQUIRE_FALSE(outcome.has_value());
        auto const* error = EvaluationError(outcome);
        REQUIRE(error != nullptr);
        REQUIRE(error->Error.Cause.has_value());
        CHECK(error->Error.Cause->Kind == expected);
        CHECK(error->Error.Cause->Hash
            == (expected == InterpreterError::Code::MissingVariable ? missingVariable : missingPrimitive));
        using OptimizerType_ = std::remove_cvref_t<decltype(optimizer)>;
        if constexpr (std::same_as<OptimizerType_, LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny>>
            || std::same_as<OptimizerType_, LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen>>) {
            auto const& diag = Diagnostics(outcome);
            CHECK_FALSE(std::isfinite(diag.InitialCost));
            CHECK_FALSE(std::isfinite(diag.FinalCost));
            CHECK(diag.Iterations == 0);
        }
    };

    LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> tiny { &fix.dtable, &fix.problem };
    check(tiny, primitiveTree, InterpreterError::Code::MissingPrimitive);

    LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> eigen { &fix.dtable, &fix.problem };
    check(eigen, variableTree, InterpreterError::Code::MissingVariable);

    LBFGSOptimizer<DTable, GaussianGradientCostFunction> lbfgs { &fix.dtable, &fix.problem };
    check(lbfgs, primitiveTree, InterpreterError::Code::MissingPrimitive);

    auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(variableTree.CoefficientsCount());
    SGDOptimizer<DTable, GaussianGradientCostFunction> sgd { &fix.dtable, &fix.problem, *rule };
    check(sgd, variableTree, InterpreterError::Code::MissingVariable);

    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &primitiveTree };
    GaussianGradientCostFunction cost { &interpreter, fix.problem.TargetValues(), fix.problem.TrainingRange() };
    auto coeff = primitiveTree.GetCoefficients();
    std::vector<Operon::Scalar> gradient(coeff.size());
    auto value = cost.Evaluate(coeff, gradient);
    REQUIRE_FALSE(value.has_value());
    REQUIRE(value.error().Cause.has_value());
    CHECK(value.error().Cause->Kind == InterpreterError::Code::MissingPrimitive);
    for (auto g : gradient) {
        CHECK(std::isnan(static_cast<double>(g)));
    }
}

TEST_CASE("Optimizers report invalid training weights as typed configuration errors", "[optimizer]")
{
    for (auto const invalid : { Operon::Scalar { -1 }, std::numeric_limits<Operon::Scalar>::quiet_NaN(),
             std::numeric_limits<Operon::Scalar>::infinity() }) {
        OptimizerFixture fix;
        using DTable = OptimizerFixture::DTable;
        std::vector<Operon::Scalar> weights(OptimizerFixture::Nrow, Operon::Scalar { 1 });
        weights[1] = invalid;
        fix.ds.SetWeights(weights);

        auto check = [&](auto const& optimizer) {
            auto outcome = optimizer.Optimize(fix.rng, fix.tree);
            REQUIRE_FALSE(outcome.has_value());
            auto const* error = ConfigurationError(outcome);
            REQUIRE(error != nullptr);
            auto expected = WeightErrorCode::NegativeValue;
            if (std::isnan(static_cast<double>(invalid))) {
                expected = WeightErrorCode::NotANumber;
            } else if (std::isinf(static_cast<double>(invalid))) {
                expected = WeightErrorCode::Infinite;
            }
            CHECK(error->Error.Code == expected);
            CHECK(error->Error.Row == 1);
            CHECK(Diagnostics(outcome).FinalParameters == fix.tree.GetCoefficients());
        };

        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> tiny { &fix.dtable, &fix.problem };
        check(tiny);
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> eigen { &fix.dtable, &fix.problem };
        check(eigen);
        LBFGSOptimizer<DTable, GaussianGradientCostFunction> lbfgs { &fix.dtable, &fix.problem };
        check(lbfgs);
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(fix.tree.CoefficientsCount());
        SGDOptimizer<DTable, GaussianGradientCostFunction> sgd { &fix.dtable, &fix.problem, *rule };
        check(sgd);
#if defined(HAVE_ASMJIT)
        Operon::JIT::JitZobrist zobrist { fix.rng, 50, fix.problem.GetInputs() };
        JIT::JitEvaluator jitEval { &fix.problem, &zobrist };
        JitLevenbergMarquardtOptimizer<DTable> jit { &fix.dtable, &fix.problem, &jitEval };
        check(jit);
#endif
    }
}

// Test problem: a "clean" half of the rows has y = X1 exactly (X1 drawn from
// Uniform(-1,1), so c0=1 fits them perfectly); a "noisy" half is fixed at
// X1=1, y=6 - a point consistent with a totally different, unmodelable
// offset relationship. Those noisy rows deterministically drag an unweighted
// fit of "c0 * X1" away from c0=1 (unlike a random symmetric perturbation,
// whose contribution can wash out to ~0 depending on the RNG draw). Zeroing
// the noisy rows' weights should recover the clean-only solution c0=1 that
// an unweighted fit cannot reach.
struct WeightedOptimizerFixture {
    static constexpr auto Nrow { 400 };
    static constexpr auto Nclean { Nrow / 2 };

    Operon::RandomGenerator rng { 0 }; // NOLINT(readability-identifier-naming)
    Operon::Dataset ds; // NOLINT(readability-identifier-naming)
    Operon::Tree tree; // NOLINT(readability-identifier-naming)
    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable; // NOLINT(readability-identifier-naming)
    Operon::Problem problem; // NOLINT(readability-identifier-naming)

    WeightedOptimizerFixture()
        : ds([&]() -> Operon::Dataset {
            std::vector<Operon::Scalar> x(Nrow);
            std::vector<Operon::Scalar> y(Nrow);
            for (auto i = 0; i < Nclean; ++i) {
                x[i] = Operon::Random::Uniform(rng, -1.0F, +1.0F);
                y[i] = x[i];
            }
            for (auto i = Nclean; i < Nrow; ++i) {
                x[i] = Operon::Scalar { 1 };
                y[i] = Operon::Scalar { 6 };
            }
            std::vector<std::vector<Operon::Scalar>> cols { x, y };
            return Operon::Dataset(cols);
        }())
        , tree([&]() -> Tree {
            auto t = InfixParser::ParseOrThrow("X1", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) {
                    node.Value = static_cast<Operon::Scalar>(0.1);
                }
            }
            return t;
        }())
        , problem(&ds)
    {
        problem.SetTrainingRange({ 0, Nrow });
        problem.SetTestRange({ 0, Nrow });
        problem.SetTarget("X2");
        std::vector<Operon::Scalar> weights(Nrow, Operon::Scalar { 1 });
        std::fill(weights.begin() + Nclean, weights.end(), Operon::Scalar { 0 });
        ds.SetWeights(weights);
    }
};

TEST_CASE("Weighted parameter optimization", "[optimizer]")
{
    WeightedOptimizerFixture fix;
    auto& rng = fix.rng;
    auto& tree = fix.tree;
    auto& dtable = fix.dtable;
    auto& problem = fix.problem;
    using DTable = WeightedOptimizerFixture::DTable;

    constexpr Operon::Scalar paramTol { 0.01F };

    auto checkRecoversCleanSolution = [&](OptimizerBase& optimizer) -> void {
        auto summary = optimizer.Optimize(rng, tree);
        // The outcome must reflect the *weighted* objective actually
        // optimized - CoefficientOptimizer (local_search.cpp) only applies
        // FinalParameters when the outcome has a value, so a false failure
        // here would silently drop a real weighted improvement.
        CHECK(summary.has_value());
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    };

    SECTION("lm / eigen")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer { &dtable, &problem };
        checkRecoversCleanSolution(optimizer);
    }

    SECTION("lm / tiny")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> optimizer { &dtable, &problem };
        checkRecoversCleanSolution(optimizer);
    }

#if defined(HAVE_ASMJIT)
    SECTION("lm / jit")
    {
        // JitLevenbergMarquardtOptimizer previously ignored Problem::Weights()
        // entirely (both its interpreter-fallback and JIT-compiled cost
        // function paths), so weighted LM behavior silently differed by
        // backend. Same discriminative fixture as "lm / eigen" above -
        // recovering c0=1 here requires the zeroed-weight rows to actually
        // be down-weighted by the JIT path too.
        Operon::JIT::JitZobrist zobrist { rng, 50, problem.GetInputs() };
        JIT::JitEvaluator jitEval { &problem, &zobrist };
        JitLevenbergMarquardtOptimizer<DTable> optimizer { &dtable, &problem, &jitEval };
        checkRecoversCleanSolution(optimizer);
    }
#endif

    SECTION("lbfgs / gaussian")
    {
        LBFGSOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem };
        checkRecoversCleanSolution(optimizer);
    }

    SECTION("sgd / gaussian")
    {
        auto const dim { tree.CoefficientsCount() };
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem, *rule };
        auto summary = optimizer.Optimize(rng, tree);
        CHECK(summary.has_value());
        // SGD converges more slowly than LM/L-BFGS on this problem within
        // the default iteration budget, so use a looser tolerance - the
        // point is confirming weights are picked up at all, not tight
        // convergence.
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, 0.1F));
        }
    }

    SECTION("lbfgs / gaussian: reported cost matches the weighted objective, not raw SSE")
    {
        // Directly pins down the root cause rather than relying on Success
        // to flip (which only happens for adversarial coefficient
        // trajectories - not guaranteed by every dataset/tolerance
        // combination): InitialCost/FinalCost must equal the *weighted* SSE
        // GaussianGradientCostFunction actually optimizes, independently recomputed here,
        // not the unweighted SumOfSquaredErrors the cost lambda used before
        // the fix.
        LBFGSOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());

        auto const range = problem.TrainingRange();
        auto const target = problem.TargetValues(range);
        auto const weights = *problem.Weights(range);
        Operon::Interpreter<Operon::Scalar, DTable> interpreter { &dtable, &fix.ds, &tree };

        auto const pred0
            = interpreter.Evaluate(Operon::Span<Operon::Scalar const> { summary->InitialParameters }, range).value();
        auto const expectedInitialCost
            = 0.5 * Operon::SumOfSquaredErrors(pred0.begin(), pred0.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary->InitialCost), Catch::Matchers::WithinRel(expectedInitialCost, 1e-3));

        auto const pred1
            = interpreter.Evaluate(Operon::Span<Operon::Scalar const> { summary->FinalParameters }, range).value();
        auto const expectedFinalCost
            = 0.5 * Operon::SumOfSquaredErrors(pred1.begin(), pred1.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary->FinalCost), Catch::Matchers::WithinRel(expectedFinalCost, 1e-3));
    }

    SECTION("lm / eigen: reported cost matches the weighted objective, not raw SSE")
    {
        // Symmetric with the "lbfgs / gaussian" cost check above, but for the
        // LM path: LeastSquaresLMAdapter applies the sqrt(w)-residual trick (see
        // least_squares_lm_adapter.hpp), so Eigen::LevenbergMarquardt's
        // fnorm()^2 * 0.5 already equals the weighted SSE / 2 - pin that down
        // directly rather than relying on it transitively via Success/params.
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer { &dtable, &problem };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());

        auto const range = problem.TrainingRange();
        auto const target = problem.TargetValues(range);
        auto const weights = *problem.Weights(range);
        Operon::Interpreter<Operon::Scalar, DTable> interpreter { &dtable, &fix.ds, &tree };

        auto const pred0
            = interpreter.Evaluate(Operon::Span<Operon::Scalar const> { summary->InitialParameters }, range).value();
        auto const expectedInitialCost
            = 0.5 * Operon::SumOfSquaredErrors(pred0.begin(), pred0.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary->InitialCost), Catch::Matchers::WithinRel(expectedInitialCost, 1e-3));

        auto const pred1
            = interpreter.Evaluate(Operon::Span<Operon::Scalar const> { summary->FinalParameters }, range).value();
        auto const expectedFinalCost
            = 0.5 * Operon::SumOfSquaredErrors(pred1.begin(), pred1.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary->FinalCost), Catch::Matchers::WithinRel(expectedFinalCost, 1e-3));
    }

    SECTION("lbfgs / gaussian: CoefficientOptimizer actually applies the weighted-optimal coefficients")
    {
        // End-to-end check through the real call path (local_search.cpp),
        // not just Optimize() directly: CoefficientOptimizer gates
        // SetCoefficients on the outcome having a value, so a mis-scored
        // outcome would silently discard a genuine weighted improvement here.
        LBFGSOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem };
        Operon::CoefficientOptimizer const coeffOptimizer { &optimizer };
        auto [optimizedTree, summary] = coeffOptimizer(rng, tree);
        REQUIRE(summary.has_value());
        auto const coeffs = optimizedTree.GetCoefficients();
        REQUIRE(!coeffs.empty());
        for (auto const c : coeffs) {
            CHECK_THAT(c, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    }

    SECTION("unweighted sanity check: LM does NOT recover c0=1")
    {
        // Confirms the test problem is actually discriminative - not that
        // "any optimizer converges to 1 regardless of weights".
        std::vector<Operon::Scalar> ones(WeightedOptimizerFixture::Nrow, Operon::Scalar { 1 });
        fix.problem.GetDataset()->SetWeights(ones);
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer { &dtable, &problem };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        auto const p = summary->FinalParameters.front();
        CHECK(std::abs(p - 1.0F) > paramTol);
    }

    SECTION("unweighted sanity check: lbfgs does NOT recover c0=1")
    {
        std::vector<Operon::Scalar> ones(WeightedOptimizerFixture::Nrow, Operon::Scalar { 1 });
        fix.problem.GetDataset()->SetWeights(ones);
        LBFGSOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        auto const p = summary->FinalParameters.front();
        CHECK(std::abs(p - 1.0F) > paramTol);
    }

    SECTION("poisson never receives ordinary dataset sample weights as exposure")
    {
        // Poisson exposure is a deliberate, separate choice from Gaussian's WLS weights:
        // PoissonGradientCostFunction::UsesDatasetWeights is false, so the optimizers never forward dataset sample
        // weights to it. Verified by re-running with an all-ones weight vector (fresh rng, same seed) and checking the
        // result matches the zeroed-weight run to within float noise (not exact ==: defensive against benign future
        // changes to evaluation order/precision elsewhere in the RNG/opt path that wouldn't actually mean weights
        // started being applied).
        LBFGSOptimizer<DTable, PoissonGradientCostFunction<>> const optimizerZeroed { &dtable, &problem };
        Operon::RandomGenerator rngZeroed { 0 };
        auto summaryZeroed = optimizerZeroed.Optimize(rngZeroed, tree);
        REQUIRE(summaryZeroed.has_value());

        std::vector<Operon::Scalar> ones(WeightedOptimizerFixture::Nrow, Operon::Scalar { 1 });
        fix.problem.GetDataset()->SetWeights(ones);
        LBFGSOptimizer<DTable, PoissonGradientCostFunction<>> const optimizerOnes { &dtable, &problem };
        Operon::RandomGenerator rngOnes { 0 };
        auto summaryOnes = optimizerOnes.Optimize(rngOnes, tree);
        REQUIRE(summaryOnes.has_value());

        REQUIRE(summaryZeroed->FinalParameters.size() == summaryOnes->FinalParameters.size());
        for (auto i = 0UL; i < summaryZeroed->FinalParameters.size(); ++i) {
            CHECK_THAT(summaryZeroed->FinalParameters[i],
                Catch::Matchers::WithinRel(summaryOnes->FinalParameters[i], static_cast<Operon::Scalar>(1e-5)));
        }
        CHECK_THAT(static_cast<double>(summaryZeroed->FinalCost),
            Catch::Matchers::WithinRel(static_cast<double>(summaryOnes->FinalCost), 1e-5));
    }
}

TEST_CASE("Minibatch optimizer diagnostics charge at least one evaluation", "[optimizer]")
{
    OptimizerFixture fix;
    fix.problem.SetTrainingRange({ 0, OptimizerFixture::Nrow });
    fix.problem.SetTarget("X4");
    using DTable = OptimizerFixture::DTable;
    LBFGSOptimizer<DTable, GaussianGradientCostFunction> optimizer { &fix.dtable, &fix.problem };
    optimizer.SetBatchSize(1);
    optimizer.SetIterations(1);
    auto outcome = optimizer.Optimize(fix.rng, fix.tree);
    REQUIRE(EvaluationError(outcome) == nullptr);
    auto const& diagnostics = Diagnostics(outcome);
    CHECK(std::isfinite(static_cast<double>(diagnostics.InitialCost)));
    CHECK(std::isfinite(static_cast<double>(diagnostics.FinalCost)));
    CHECK(diagnostics.FunctionEvaluations >= 1);
    CHECK(diagnostics.JacobianEvaluations >= 1);
    auto const range = fix.problem.TrainingRange();
    auto const target = fix.problem.TargetValues(range);
    Operon::Interpreter<Operon::Scalar, DTable> interpreter { &fix.dtable, &fix.ds, &fix.tree };
    auto const pred0
        = interpreter.Evaluate(Operon::Span<Operon::Scalar const> { diagnostics.InitialParameters }, range).value();
    auto const pred1
        = interpreter.Evaluate(Operon::Span<Operon::Scalar const> { diagnostics.FinalParameters }, range).value();
    auto expectedCost = [&](auto const& prediction) {
        double cost = 0;
        for (std::size_t i = 0; i < prediction.size(); ++i) {
            auto const residual = static_cast<double>(prediction[i] - target[i]);
            cost += 0.5 * residual * residual;
        }
        return cost;
    };
    CHECK_THAT(static_cast<double>(diagnostics.InitialCost), Catch::Matchers::WithinRel(expectedCost(pred0), 1e-5));
    CHECK_THAT(static_cast<double>(diagnostics.FinalCost), Catch::Matchers::WithinRel(expectedCost(pred1), 1e-5));
}

// Same clean/noisy problem as WeightedOptimizerFixture, but padded with
// Npad unused rows so the training range starts at a non-zero offset
// (range_.Start() = Npad). GaussianGradientCostFunction::Evaluate previously indexed
// target_/weights_ by the *absolute* range.Start() instead of an offset
// relative to range_.Start(); since SelectRandomRange() returns range_
// itself whenever batchSize >= range_.Size() (the default), this reproduces
// the out-of-bounds read even without any random sub-batching - LBFGS/SGD
// with a non-zero training-range start alone was enough to trigger it.
// LM is unaffected (LeastSquaresLMAdapter always indexes 0..numResiduals_-1,
// never range.Start()), so this only needs to cover LBFGS/SGD.
struct WeightedOptimizerNonZeroStartFixture {
    static constexpr auto Npad { 100 };
    static constexpr auto Nrow { 400 };
    static constexpr auto Nclean { Nrow / 2 };
    static constexpr auto Ntotal { Npad + Nrow };

    Operon::RandomGenerator rng { 0 }; // NOLINT(readability-identifier-naming)
    Operon::Dataset ds; // NOLINT(readability-identifier-naming)
    Operon::Tree tree; // NOLINT(readability-identifier-naming)
    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable; // NOLINT(readability-identifier-naming)
    Operon::Problem problem; // NOLINT(readability-identifier-naming)

    WeightedOptimizerNonZeroStartFixture()
        : ds([&]() -> Operon::Dataset {
            std::vector<Operon::Scalar> x(Ntotal);
            std::vector<Operon::Scalar> y(Ntotal);
            for (auto i = 0; i < Npad; ++i) {
                x[i] = Operon::Scalar { -2 }; // never read - outside training/test range
                y[i] = Operon::Scalar { 100 };
            }
            for (auto i = 0; i < Nclean; ++i) {
                x[Npad + i] = Operon::Random::Uniform(rng, -1.0F, +1.0F);
                y[Npad + i] = x[Npad + i];
            }
            for (auto i = Nclean; i < Nrow; ++i) {
                x[Npad + i] = Operon::Scalar { 1 };
                y[Npad + i] = Operon::Scalar { 6 };
            }
            std::vector<std::vector<Operon::Scalar>> cols { x, y };
            return Operon::Dataset(cols);
        }())
        , tree([&]() -> Tree {
            auto t = InfixParser::ParseOrThrow("X1", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) {
                    node.Value = static_cast<Operon::Scalar>(0.1);
                }
            }
            return t;
        }())
        , problem(&ds)
    {
        problem.SetTrainingRange({ Npad, Ntotal });
        problem.SetTestRange({ Npad, Ntotal });
        problem.SetTarget("X2");
        std::vector<Operon::Scalar> weights(Ntotal, Operon::Scalar { 1 });
        std::fill(weights.begin() + Npad + Nclean, weights.end(), Operon::Scalar { 0 });
        ds.SetWeights(weights);
    }
};

TEST_CASE("Weighted parameter optimization with non-zero training range start", "[optimizer]")
{
    WeightedOptimizerNonZeroStartFixture fix;
    auto& rng = fix.rng;
    auto& tree = fix.tree;
    auto& dtable = fix.dtable;
    auto& problem = fix.problem;
    using DTable = WeightedOptimizerNonZeroStartFixture::DTable;

    constexpr Operon::Scalar paramTol { 0.01F };

    SECTION("lbfgs / gaussian")
    {
        LBFGSOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    }

    SECTION("sgd / gaussian")
    {
        auto const dim { tree.CoefficientsCount() };
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem, *rule };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, 0.1F));
        }
    }

    SECTION("lbfgs / gaussian: negative placeholder weights outside the training range don't trip validation")
    {
        // GaussianGradientCostFunction now receives the whole-dataset weights column (not a
        // slice pre-cut to the training range), so its weight-sign check
        // must only validate the in-range slice - rows outside range_ (e.g.
        // padding, or other splits' rows) are never read by SelectBatch and
        // may legitimately carry negative/sentinel values.
        std::vector<Operon::Scalar> weights(WeightedOptimizerNonZeroStartFixture::Ntotal, Operon::Scalar { 1 });
        std::fill(weights.begin(), weights.begin() + WeightedOptimizerNonZeroStartFixture::Npad, Operon::Scalar { -1 });
        std::fill(
            weights.begin() + WeightedOptimizerNonZeroStartFixture::Npad + WeightedOptimizerNonZeroStartFixture::Nclean,
            weights.end(), Operon::Scalar { 0 });
        fix.problem.GetDataset()->SetWeights(weights);

        LBFGSOptimizer<DTable, GaussianGradientCostFunction> optimizer { &dtable, &problem };
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    }
}

TEST_CASE("PoissonGradientCostFunction respects a non-zero training range start", "[optimizer]")
{
    // Same off-by-range_.Start() bug class fixed in Gaussian's gradient cost, but
    // exercised directly on PoissonGradientCostFunction::Evaluate rather than through a
    // full optimization run (Poisson doesn't have a clean, guaranteed-
    // converging target on this kind of problem). Two structurally
    // identical problems - one with range_.Start()==0, one padded so
    // range_.Start() > 0 - must produce identical loss/gradient for the
    // same fixed coefficients; a wrong offset would instead read
    // out-of-bounds/wrong data for the padded case.
    using DTable = DispatchTable<Operon::Scalar>;
    constexpr auto Npad { 37 };
    constexpr auto Nrow { 50 };

    auto build = [&](int pad) -> Operon::Dataset {
        std::vector<Operon::Scalar> x(pad + Nrow);
        std::vector<Operon::Scalar> y(pad + Nrow);
        for (auto i = 0; i < pad; ++i) {
            x[i] = Operon::Scalar { 999 };
            y[i] = Operon::Scalar { 999 };
        } // never read
        for (auto i = 0; i < Nrow; ++i) {
            x[pad + i] = static_cast<Operon::Scalar>(i + 1) * 0.1F;
            y[pad + i] = static_cast<Operon::Scalar>(i + 1);
        }
        std::vector<std::vector<Operon::Scalar>> cols { x, y };
        return Operon::Dataset(cols);
    };

    auto ds0 = build(0);
    auto dsPad = build(Npad);

    auto tree0 = InfixParser::ParseOrThrow("X1", ds0);
    auto treePad = InfixParser::ParseOrThrow("X1", dsPad);
    for (auto* t : { &tree0, &treePad }) {
        for (auto& node : t->Nodes()) {
            if (node.IsVariable()) {
                node.Value = Operon::Scalar { 1 };
            }
        }
    }

    DTable dtable;

    Operon::Problem problem0(&ds0);
    problem0.SetTrainingRange({ 0, Nrow });
    problem0.SetTarget("X2");

    Operon::Problem problemPad(&dsPad);
    problemPad.SetTrainingRange({ Npad, Npad + Nrow });
    problemPad.SetTarget("X2");

    Operon::Interpreter<Operon::Scalar, DTable> interp0 { &dtable, &ds0, &tree0 };
    Operon::Interpreter<Operon::Scalar, DTable> interpPad { &dtable, &dsPad, &treePad };

    // PoissonGradientCostFunction requires the whole dataset column
    // (absolute, dataset-row-indexed), not a slice pre-cut to the training range.
    auto target0 = problem0.TargetValues();
    auto targetPad = problemPad.TargetValues();

    PoissonGradientCostFunction<> cost0 { &interp0, target0, problem0.TrainingRange() };
    PoissonGradientCostFunction<> costPad { &interpPad, targetPad, problemPad.TrainingRange() };

    auto coeff = tree0.GetCoefficients();
    REQUIRE(!coeff.empty());
    std::vector<Operon::Scalar> grad0(coeff.size());
    std::vector<Operon::Scalar> gradPad(coeff.size());

    auto const c0 = cost0.Evaluate(coeff, grad0);
    auto const cPad = costPad.Evaluate(coeff, gradPad);
    REQUIRE(c0.has_value());
    REQUIRE(cPad.has_value());

    CHECK_THAT(static_cast<double>(*c0), Catch::Matchers::WithinRel(static_cast<double>(*cPad), 1e-5));
    for (std::size_t i = 0; i < grad0.size(); ++i) {
        CHECK_THAT(static_cast<double>(grad0[i]), Catch::Matchers::WithinRel(static_cast<double>(gradPad[i]), 1e-5));
    }
}

TEST_CASE("SGD update rules", "[optimizer]")
{
    OptimizerFixture fix;
    auto& rng = fix.rng;
    auto& tree = fix.tree;
    auto& dtable = fix.dtable;
    auto& problem = fix.problem;
    using DTable = OptimizerFixture::DTable;

    auto const dim { tree.CoefficientsCount() };

    Operon::Vector<std::unique_ptr<UpdateRule::LearningRateUpdateRule const>> rules;
    rules.emplace_back(new UpdateRule::Constant<Operon::Scalar>(dim, 1e-3)); // NOLINT
    rules.emplace_back(new UpdateRule::Momentum<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::RmsProp<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::AdaDelta<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::AdaMax<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::Adam<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::YamAdam<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::AmsGrad<Operon::Scalar>(dim));
    rules.emplace_back(new UpdateRule::Yogi<Operon::Scalar>(dim));

    for (auto const& rule : rules) {
        SGDOptimizer<DTable, GaussianGradientCostFunction> const optimizer { &dtable, &problem, *rule };
        auto summary = optimizer.Optimize(rng, tree);
        // Not gated on summary.has_value(): some rules (see YamAdam note
        // below) may not actually improve the cost on this fixture, but
        // diagnostics are populated either way (FitResult/FitFailure both
        // carry them) - this test only cares that the run itself is sane.
        auto const& diag = Diagnostics(summary);
        CHECK(diag.Iterations > 0);
        // YamAdam applies the raw (summed) gradient as its first step (step size ≈ 1),
        // which overshoots on unnormalized losses with large n. We only require finite output.
        CHECK(std::isfinite(diag.FinalCost));
    }
}

// Gaussian cost whose solver-facing instance (the one the optimizers build
// with a non-null rng) starts failing after SuccessfulEvaluations calls. The
// endpoint instance (rng == nullptr) always delegates, so initial and final
// endpoint costs stay valid and only the solve is disturbed. The static knobs
// are per-test configuration and are reset by each test case.
struct FailingGaussianCost {
    enum class Mode { Error, NonFiniteValue };

    using Scalar = Operon::Scalar;
    static inline Mode FailureMode { Mode::Error };
    static constexpr bool UsesDatasetWeights { GaussianGradientCostFunction::UsesDatasetWeights };
    static inline std::size_t SuccessfulEvaluations { 0 };
    static inline std::size_t InjectedFailures { 0 };

    FailingGaussianCost(gsl::not_null<InterpreterBase<Scalar> const*> interpreter, ConstScalarSpan target, Range range,
        RandomGenerator* rng, std::size_t batchSize, ConstScalarSpan weights)
        : inner_ { interpreter, target, range, rng, batchSize, weights }
        , solverFacing_ { rng != nullptr }
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t { return inner_.NumParameters(); }
    [[nodiscard]] auto FunctionEvaluations() const noexcept -> std::size_t { return inner_.FunctionEvaluations(); }
    [[nodiscard]] auto JacobianEvaluations() const noexcept -> std::size_t { return inner_.JacobianEvaluations(); }

    [[nodiscard]] auto Evaluate(ConstScalarSpan parameters, ScalarSpan gradient) const
        -> tl::expected<Scalar, GradientError>
    {
        if (solverFacing_ && calls_++ >= SuccessfulEvaluations) {
            ++InjectedFailures;
            std::fill(gradient.begin(), gradient.end(), std::numeric_limits<Scalar>::quiet_NaN());
            if (FailureMode == Mode::Error) {
                return tl::unexpected(GradientError { .Code = GradientErrorCode::NumericalFailure });
            }
            return std::numeric_limits<Scalar>::quiet_NaN();
        }
        return inner_.Evaluate(parameters, gradient);
    }

private:
    GaussianGradientCostFunction inner_;
    bool solverFacing_;
    mutable std::size_t calls_ { 0 };
};

static_assert(Concepts::InterpreterGradientCost<FailingGaussianCost>);

// Gaussian numerics behind a hand-written interpreter-backed cost surface.
// The variants below each add (or omit) exactly one piece of the
// InterpreterGradientCost contract on top of it.
struct InterpreterCostCore {
    using Scalar = Operon::Scalar;

    InterpreterCostCore(gsl::not_null<InterpreterBase<Scalar> const*> interpreter, ConstScalarSpan target, Range range,
        RandomGenerator* rng, std::size_t batchSize, ConstScalarSpan weights)
        : inner_ { interpreter, target, range, rng, batchSize, weights }
    {
    }

    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t { return inner_.NumParameters(); }
    [[nodiscard]] auto Evaluate(ConstScalarSpan parameters, ScalarSpan gradient) const
        -> tl::expected<Scalar, GradientError>
    {
        return inner_.Evaluate(parameters, gradient);
    }

    GaussianGradientCostFunction inner_;
};

// Conforming custom cost that records the weights span every constructed
// instance (solver-facing and endpoint) was given.
template <bool UseWeights> struct RecordingCost : InterpreterCostCore {
    static constexpr bool UsesDatasetWeights { UseWeights };
    static inline std::vector<ConstScalarSpan> Received;

    RecordingCost(gsl::not_null<InterpreterBase<Scalar> const*> interpreter, ConstScalarSpan target, Range range,
        RandomGenerator* rng, std::size_t batchSize, ConstScalarSpan weights)
        : InterpreterCostCore { interpreter, target, range, rng, batchSize, weights }
    {
        Received.push_back(weights);
    }

    [[nodiscard]] auto FunctionEvaluations() const noexcept -> std::size_t { return inner_.FunctionEvaluations(); }
    [[nodiscard]] auto JacobianEvaluations() const noexcept -> std::size_t { return inner_.JacobianEvaluations(); }
};

// Satisfies the solver-facing GradientCost contract but none of the
// interpreter-backed one: no constructor from a tree, no counters, no
// weight policy.
struct GradientOnlyCost {
    using Scalar = Operon::Scalar;
    [[nodiscard]] auto NumParameters() const noexcept -> std::size_t { return 1; }
    [[nodiscard]] auto Evaluate(ConstScalarSpan /*parameters*/, ScalarSpan /*gradient*/) const
        -> tl::expected<Scalar, GradientError>
    {
        return Scalar { 0 };
    }
};

struct CostWithoutWeightPolicy : InterpreterCostCore {
    using InterpreterCostCore::InterpreterCostCore;
    [[nodiscard]] auto FunctionEvaluations() const noexcept -> std::size_t { return 0; }
    [[nodiscard]] auto JacobianEvaluations() const noexcept -> std::size_t { return 0; }
};

struct CostWithoutCounters : InterpreterCostCore {
    using InterpreterCostCore::InterpreterCostCore;
    static constexpr bool UsesDatasetWeights { false };
};

struct CostWithNonBoolPolicy : InterpreterCostCore {
    using InterpreterCostCore::InterpreterCostCore;
    static constexpr int UsesDatasetWeights { 1 };
    [[nodiscard]] auto FunctionEvaluations() const noexcept -> std::size_t { return 0; }
    [[nodiscard]] auto JacobianEvaluations() const noexcept -> std::size_t { return 0; }
};

struct CostWithRuntimePolicy : InterpreterCostCore {
    using InterpreterCostCore::InterpreterCostCore;
    static inline bool UsesDatasetWeights { false };
    [[nodiscard]] auto FunctionEvaluations() const noexcept -> std::size_t { return 0; }
    [[nodiscard]] auto JacobianEvaluations() const noexcept -> std::size_t { return 0; }
};

struct CostWithoutWeightsParameter : InterpreterCostCore {
    static constexpr bool UsesDatasetWeights { false };
    CostWithoutWeightsParameter(InterpreterBase<Scalar> const* interpreter, ConstScalarSpan target, Range range,
        RandomGenerator* rng, std::size_t batchSize)
        : InterpreterCostCore { interpreter, target, range, rng, batchSize, {} }
    {
    }
    [[nodiscard]] auto FunctionEvaluations() const noexcept -> std::size_t { return 0; }
    [[nodiscard]] auto JacobianEvaluations() const noexcept -> std::size_t { return 0; }
};

static_assert(Concepts::GradientCost<GradientOnlyCost>);
static_assert(!Concepts::InterpreterGradientCost<GradientOnlyCost>);
static_assert(Concepts::GradientCost<CostWithoutWeightPolicy>);
static_assert(!Concepts::InterpreterGradientCost<CostWithoutWeightPolicy>);
static_assert(!Concepts::InterpreterGradientCost<CostWithoutCounters>);
static_assert(!Concepts::InterpreterGradientCost<CostWithNonBoolPolicy>);
static_assert(!Concepts::InterpreterGradientCost<CostWithRuntimePolicy>);
static_assert(Concepts::GradientCost<CostWithoutWeightsParameter>);
static_assert(!Concepts::InterpreterGradientCost<CostWithoutWeightsParameter>);
static_assert(Concepts::InterpreterGradientCost<RecordingCost<true>>);
static_assert(Concepts::InterpreterGradientCost<RecordingCost<false>>);

namespace {
    template <typename Cost> auto OptimizeWithLbfgsAndSgd(OptimizerFixture& fix) -> std::vector<FitOutcome>
    {
        using DTable = OptimizerFixture::DTable;
        std::vector<FitOutcome> outcomes;
        LBFGSOptimizer<DTable, Cost> lbfgs { &fix.dtable, &fix.problem };
        outcomes.push_back(lbfgs.Optimize(fix.rng, fix.tree));
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(fix.tree.CoefficientsCount());
        SGDOptimizer<DTable, Cost> sgd { &fix.dtable, &fix.problem, *rule };
        outcomes.push_back(sgd.Optimize(fix.rng, fix.tree));
        return outcomes;
    }
} // namespace

TEST_CASE("Optimizers run a conforming custom interpreter-backed cost", "[optimizer]")
{
    OptimizerFixture fix;
    RecordingCost<false>::Received.clear();

    auto const outcomes = OptimizeWithLbfgsAndSgd<RecordingCost<false>>(fix);
    REQUIRE(outcomes.size() == 2);
    for (auto const& outcome : outcomes) {
        REQUIRE(outcome.has_value());
        auto const& diag = Diagnostics(outcome);
        CHECK(diag.FinalCost < diag.InitialCost);
        // The counters come from the custom cost itself.
        CHECK(diag.FunctionEvaluations > 0);
        CHECK(diag.JacobianEvaluations > 0);
    }
}

TEST_CASE("Dataset weights reach a cost only when it declares UsesDatasetWeights", "[optimizer]")
{
    OptimizerFixture fix;
    std::vector<Operon::Scalar> weights(OptimizerFixture::Nrow, Operon::Scalar { 2 });
    fix.ds.SetWeights(weights);
    auto const column = *fix.ds.Weights();

    SECTION("declared: the whole dataset column is forwarded")
    {
        RecordingCost<true>::Received.clear();
        auto const outcomes = OptimizeWithLbfgsAndSgd<RecordingCost<true>>(fix);
        for (auto const& outcome : outcomes) {
            CHECK(outcome.has_value());
        }
        REQUIRE(!RecordingCost<true>::Received.empty());
        for (auto const received : RecordingCost<true>::Received) {
            CHECK(received.data() == column.data());
            CHECK(received.size() == column.size());
        }
    }

    SECTION("not declared: the cost receives no weights")
    {
        RecordingCost<false>::Received.clear();
        auto const outcomes = OptimizeWithLbfgsAndSgd<RecordingCost<false>>(fix);
        for (auto const& outcome : outcomes) {
            CHECK(outcome.has_value());
        }
        REQUIRE(!RecordingCost<false>::Received.empty());
        for (auto const received : RecordingCost<false>::Received) {
            CHECK(received.empty());
        }
    }

    SECTION("invalid dataset weights are validated only for a declaring cost")
    {
        weights[1] = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        fix.ds.SetWeights(weights);

        RecordingCost<true>::Received.clear();
        for (auto const& outcome : OptimizeWithLbfgsAndSgd<RecordingCost<true>>(fix)) {
            CHECK(ConfigurationError(outcome) != nullptr);
        }
        CHECK(RecordingCost<true>::Received.empty());

        RecordingCost<false>::Received.clear();
        for (auto const& outcome : OptimizeWithLbfgsAndSgd<RecordingCost<false>>(fix)) {
            CHECK(ConfigurationError(outcome) == nullptr);
            CHECK(outcome.has_value());
        }
    }
}

TEST_CASE("SGD stops at a mid-solve cost failure and keeps the last finite iterate", "[optimizer]")
{
    using DTable = OptimizerFixture::DTable;
    constexpr std::size_t successful { 3 };

    for (auto const mode : { FailingGaussianCost::Mode::Error, FailingGaussianCost::Mode::NonFiniteValue }) {
        OptimizerFixture fix;
        FailingGaussianCost::FailureMode = mode;
        FailingGaussianCost::SuccessfulEvaluations = successful;
        FailingGaussianCost::InjectedFailures = 0;

        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(fix.tree.CoefficientsCount());
        SGDOptimizer<DTable, FailingGaussianCost> optimizer { &fix.dtable, &fix.problem, *rule };
        auto outcome = optimizer.Optimize(fix.rng, fix.tree);

        CHECK(FailingGaussianCost::InjectedFailures == 1U);
        // The failed evaluation is not an optimizer failure: the iterate
        // reached before it is judged by the endpoint cost like any other.
        CHECK(EvaluationError(outcome) == nullptr);
        CHECK(ConfigurationError(outcome) == nullptr);
        auto const& diag = Diagnostics(outcome);
        CHECK(diag.Iterations == static_cast<int>(successful));
        CHECK(diag.FinalParameters != diag.InitialParameters);
        for (auto const p : diag.FinalParameters) {
            CHECK(std::isfinite(p));
        }
        CHECK(std::isfinite(diag.FinalCost));
        REQUIRE(outcome.has_value());
        CHECK(diag.FinalCost < diag.InitialCost);
    }
}

TEST_CASE("L-BFGS treats a mid-solve non-finite trial as a rejected line-search probe", "[optimizer]")
{
    using DTable = OptimizerFixture::DTable;

    for (auto const mode : { FailingGaussianCost::Mode::Error, FailingGaussianCost::Mode::NonFiniteValue }) {
        OptimizerFixture fix;
        FailingGaussianCost::FailureMode = mode;
        FailingGaussianCost::SuccessfulEvaluations = 2;
        FailingGaussianCost::InjectedFailures = 0;

        LBFGSOptimizer<DTable, FailingGaussianCost> optimizer { &fix.dtable, &fix.problem };
        auto outcome = optimizer.Optimize(fix.rng, fix.tree);

        CHECK(FailingGaussianCost::InjectedFailures >= 1U);
        // lbfgs reverts to the last accepted iterate and succeeds, so the
        // solver-facing failure is advisory: no typed error, and the endpoint
        // is a finite point that never costs more than the start.
        CHECK(EvaluationError(outcome) == nullptr);
        CHECK(ConfigurationError(outcome) == nullptr);
        auto const& diag = Diagnostics(outcome);
        for (auto const p : diag.FinalParameters) {
            CHECK(std::isfinite(p));
        }
        CHECK(std::isfinite(diag.FinalCost));
        CHECK(diag.FinalCost <= diag.InitialCost);
    }
}

TEST_CASE("Poisson optimizers ignore ordinary dataset weights instead of validating them", "[optimizer]")
{
    using DTable = OptimizerFixture::DTable;
    OptimizerFixture fix;
    std::vector<Operon::Scalar> weights(OptimizerFixture::Nrow, Operon::Scalar { 1 });
    weights[1] = std::numeric_limits<Operon::Scalar>::quiet_NaN();
    fix.ds.SetWeights(weights);

    LBFGSOptimizer<DTable, PoissonGradientCostFunction<>> lbfgs { &fix.dtable, &fix.problem };
    auto lbfgsOutcome = lbfgs.Optimize(fix.rng, fix.tree);
    CHECK(ConfigurationError(lbfgsOutcome) == nullptr);
    CHECK(lbfgsOutcome.has_value());

    auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(fix.tree.CoefficientsCount());
    SGDOptimizer<DTable, PoissonGradientCostFunction<>> sgd { &fix.dtable, &fix.problem, *rule };
    auto sgdOutcome = sgd.Optimize(fix.rng, fix.tree);
    CHECK(ConfigurationError(sgdOutcome) == nullptr);
    CHECK(sgdOutcome.has_value());
}

TEST_CASE("Zero-parameter fits report the endpoint cost with no iterations on every optimizer", "[optimizer]")
{
    using DTable = OptimizerFixture::DTable;
    constexpr Operon::Scalar factor { 0.5F };

    for (auto const weight : { Operon::Scalar { 1 }, Operon::Scalar { 2 } }) {
        OptimizerFixture fix;
        if (weight != Operon::Scalar { 1 }) {
            fix.ds.SetWeights(std::vector<Operon::Scalar>(OptimizerFixture::Nrow, weight));
        }
        // 0.5*(X1+X2+X3) against y = X1+X2+X3 leaves residual -0.5*y, with
        // every node fixed, so there is nothing to optimize.
        auto tree = fix.tree;
        for (auto& node : tree.Nodes()) {
            if (node.IsVariable()) {
                node.Value = factor;
            }
            node.Optimize = false;
        }
        REQUIRE(tree.CoefficientsCount() == 0);

        double expected { 0 };
        for (auto i = 0; i < OptimizerFixture::Nrow; ++i) {
            auto const y = static_cast<double>(fix.data(i, OptimizerFixture::Ncol - 1));
            expected += static_cast<double>(weight) * 0.25 * y * y;
        }
        expected *= 0.5;

        auto check = [&](OptimizerBase const& optimizer) {
            auto outcome = optimizer.Optimize(fix.rng, tree);
            // Nothing improved, but nothing failed either.
            REQUIRE_FALSE(outcome.has_value());
            CHECK(std::get_if<FitFailure>(&outcome.error()) != nullptr);
            auto const& diag = Diagnostics(outcome);
            CHECK(diag.InitialParameters.empty());
            CHECK(diag.FinalParameters.empty());
            CHECK(diag.Iterations == 0);
            CHECK(diag.FunctionEvaluations == 1);
            CHECK(diag.JacobianEvaluations == 0);
            CHECK_THAT(static_cast<double>(diag.InitialCost), Catch::Matchers::WithinRel(expected, 1e-3));
            CHECK(diag.FinalCost == diag.InitialCost);
        };

        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> tiny { &fix.dtable, &fix.problem };
        check(tiny);
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> eigen { &fix.dtable, &fix.problem };
        check(eigen);
        LBFGSOptimizer<DTable, GaussianGradientCostFunction> lbfgs { &fix.dtable, &fix.problem };
        check(lbfgs);
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(0);
        SGDOptimizer<DTable, GaussianGradientCostFunction> sgd { &fix.dtable, &fix.problem, *rule };
        check(sgd);
#if defined(HAVE_ASMJIT)
        Operon::JIT::JitZobrist zobrist { fix.rng, 50, fix.problem.GetInputs() };
        JIT::JitEvaluator jitEval { &fix.problem, &zobrist };
        JitLevenbergMarquardtOptimizer<DTable> jit { &fix.dtable, &fix.problem, &jitEval };
        check(jit);
#endif
    }
}

TEST_CASE("Zero-parameter fits of an unevaluable tree return the typed evaluation error", "[optimizer]")
{
    using DTable = OptimizerFixture::DTable;
    OptimizerFixture fix;
    constexpr auto missingPrimitive = Operon::Hash { 0xDEADBEEF };
    auto tree = Operon::Tree({
        Operon::Node::Constant(1),
        Operon::Node::Constant(2),
        Operon::Node::Function(missingPrimitive, 2),
    });
    for (auto& node : tree.Nodes()) {
        node.Optimize = false;
    }
    REQUIRE(tree.CoefficientsCount() == 0);

    auto check = [&](OptimizerBase const& optimizer) {
        auto outcome = optimizer.Optimize(fix.rng, tree);
        REQUIRE_FALSE(outcome.has_value());
        auto const* error = EvaluationError(outcome);
        REQUIRE(error != nullptr);
        REQUIRE(error->Error.Cause.has_value());
        CHECK(error->Error.Cause->Kind == InterpreterError::Code::MissingPrimitive);
        CHECK(Diagnostics(outcome).Iterations == 0);
    };

    LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> tiny { &fix.dtable, &fix.problem };
    check(tiny);
    LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> eigen { &fix.dtable, &fix.problem };
    check(eigen);
    LBFGSOptimizer<DTable, GaussianGradientCostFunction> lbfgs { &fix.dtable, &fix.problem };
    check(lbfgs);
    auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(0);
    SGDOptimizer<DTable, GaussianGradientCostFunction> sgd { &fix.dtable, &fix.problem, *rule };
    check(sgd);
}

} // namespace Operon::Test
