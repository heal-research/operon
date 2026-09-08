// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <exception>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/solvers/sgd.hpp"
#if defined(OPERON_HAVE_HIP)
#include "operon/optimizer/hip_context.hpp"
#include "operon/optimizer/population_encoding.hpp"
#include "operon/operators/evaluator.hpp"
#endif
#if defined(OPERON_HAVE_SYCL)
#include "operon/optimizer/sycl_context.hpp"
#include "operon/optimizer/population_encoding.hpp"
#endif
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

    Operon::RandomGenerator rng{0}; // NOLINT(readability-identifier-naming)
    Eigen::Array<Operon::Scalar, -1, -1> data{Nrow, Ncol}; // NOLINT(readability-identifier-naming)
    Operon::Dataset ds; // NOLINT(readability-identifier-naming)
    Operon::Tree tree; // NOLINT(readability-identifier-naming)
    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable; // NOLINT(readability-identifier-naming)
    Operon::Problem problem; // NOLINT(readability-identifier-naming)

    OptimizerFixture()
        : ds([&]() -> Operon::Dataset {
            for (auto i = 0; i < Ncol - 1; ++i) {
                auto col = data.col(i);
                std::generate(col.begin(), col.end(), [&]() -> float { return Operon::Random::Uniform(rng, -1.0F, +1.0F); });
            }
            data.col(Ncol - 1) = data.col(0) + data.col(1) + data.col(2);
            std::vector<std::vector<Operon::Scalar>> cols(Ncol);
            for (auto j = 0; j < Ncol; ++j) {
                cols[j].assign(data.col(j).data(), data.col(j).data() + Nrow);
            }
            return Operon::Dataset(cols);
        }())
        , tree([&]() -> Tree {
            auto t = InfixParser::Parse("X1 + X2 + X3", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) { node.Value = static_cast<Operon::Scalar>(0.1); }
            }
            return t;
        }())
        , problem(&ds)
    {
        problem.SetTrainingRange({0, Nrow});
        problem.SetTestRange({0, Nrow});
        problem.SetTarget("X4"); // last column: X1+X2+X3
    }
};

TEST_CASE("Gaussian likelihood static methods", "[likelihood]")
{
    using Lik = GaussianLikelihood<Operon::Scalar>;
    constexpr auto n { 100 };

    SECTION("perfect prediction, scalar sigma=1: NLL = n/2 * log(2pi)") {
        std::vector<Operon::Scalar> pred(n, 1.0F);
        std::vector<Operon::Scalar> target(n, 1.0F); // zero residuals
        std::vector<Operon::Scalar> sigma(1, 1.0F);
        auto nll = Lik::ComputeLikelihood(pred, target, sigma);
        auto expected = n / 2.0 * std::log(Operon::Math::Tau);
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(expected, 1e-5));
    }

    SECTION("known residuals, scalar sigma: NLL = n/2 * log(2pi*s2) + SSR/(2*s2)") {
        // pred = 1, target = 0  =>  eᵢ = 1, SSR = n
        std::vector<Operon::Scalar> pred(n, 1.0F);
        std::vector<Operon::Scalar> target(n, 0.0F);
        constexpr double s { 2.0 };
        std::vector<Operon::Scalar> sigma(1, static_cast<Operon::Scalar>(s));
        auto expected = 0.5 * (n * std::log(Operon::Math::Tau * s * s) + n / (s * s));
        auto nll = Lik::ComputeLikelihood(pred, target, sigma);
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(expected, 1e-5));
    }

    SECTION("GaussianLoss::ComputeLikelihood delegates to GaussianLikelihood") {
        std::vector<Operon::Scalar> pred(n, 1.0F);
        std::vector<Operon::Scalar> target(n, 0.0F);
        std::vector<Operon::Scalar> sigma(1, 1.0F);
        CHECK(GaussianLoss<Operon::Scalar>::ComputeLikelihood(pred, target, sigma)
           == GaussianLikelihood<Operon::Scalar>::ComputeLikelihood(pred, target, sigma));
    }

    SECTION("FisherMatrix shape and values: identity jacobian, scalar sigma") {
        // J = I (n×n), sigma = 2  =>  F = J^T J / sigma^2 = I / 4
        std::vector<Operon::Scalar> pred(n, 0.0F);
        Eigen::Matrix<Operon::Scalar, -1, -1> jac = Eigen::Matrix<Operon::Scalar, -1, -1>::Identity(n, n);
        std::vector<Operon::Scalar> sigma(1, 2.0F);
        auto fisher = Lik::ComputeFisherMatrix(pred, {jac.data(), static_cast<std::size_t>(jac.size())}, sigma);
        REQUIRE(fisher.rows() == n);
        REQUIRE(fisher.cols() == n);
        CHECK_THAT(static_cast<double>(fisher.diagonal().minCoeff()),
                   Catch::Matchers::WithinRel(0.25, 1e-5));
        CHECK_THAT(static_cast<double>(fisher.diagonal().maxCoeff()),
                   Catch::Matchers::WithinRel(0.25, 1e-5));
    }
}

TEST_CASE("Poisson optimizer diagnostics use Poisson NLL", "[optimizer][likelihood]")
{
    using Loss = PoissonLoss<Operon::Scalar>;
    std::vector<Operon::Scalar> const prediction{0.0F, std::log(2.0F)};
    std::vector<Operon::Scalar> const target{1.0F, 3.0F};

    CHECK(Loss::Cost(prediction, target, {})
        == PoissonLikelihood<Operon::Scalar>::ComputeLikelihood(prediction, target, {}));
}

TEST_CASE("Parameter optimization", "[optimizer]") // NOLINT(readability-function-cognitive-complexity)
{
    OptimizerFixture fix;
    auto& rng     = fix.rng;
    auto& tree    = fix.tree;
    auto& dtable  = fix.dtable;
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

    SECTION("tiny solver") {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> optimizer{&dtable, &problem};
        checkExact(optimizer);
    }

    SECTION("eigen solver") {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer{&dtable, &problem};
        checkExact(optimizer);
    }

    SECTION("lbfgs / gaussian") {
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        checkExact(optimizer);
    }

    SECTION("lbfgs / poisson") {
        // Poisson loss on a continuous target: just verify it runs and improves
        LBFGSOptimizer<DTable, PoissonLoss<Operon::Scalar>> const optimizer{&dtable, &problem};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        CHECK(std::isfinite(summary->FinalCost));
        CHECK(!summary->FinalParameters.empty());
    }

    SECTION("sgd / gaussian") {
        auto const dim{tree.CoefficientsCount()};
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        checkImproved(optimizer);
    }

    SECTION("sgd / poisson") {
        auto const dim{tree.CoefficientsCount()};
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, PoissonLoss<Operon::Scalar>> const optimizer{&dtable, &problem, *rule};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        CHECK(std::isfinite(summary->FinalCost));
        CHECK(!summary->FinalParameters.empty());
    }

#if defined(HAVE_ASMJIT)
    SECTION("jit tiny solver") {
        Operon::JIT::JitZobrist zobrist{rng, 50, problem.GetInputs()};
        JIT::JitEvaluator jitEval{&problem, &zobrist};
        JitLevenbergMarquardtOptimizer<DTable> optimizer{&dtable, &problem, &jitEval};
        checkExact(optimizer);
    }
#endif

    SECTION("ComputeLikelihood virtual dispatch: pred==target => NLL = n/2 * log(2pi)") {
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> const optimizer{&dtable, &problem};
        auto range = problem.TrainingRange();
        auto target = problem.TargetValues(range);
        std::vector<Operon::Scalar> sigma(1, 1.0F);
        auto nll = optimizer.ComputeLikelihood(target, target, sigma);
        auto expected = static_cast<double>(range.Size()) / 2.0 * std::log(Operon::Math::Tau);
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(expected, 1e-4));
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

    Operon::RandomGenerator rng{0}; // NOLINT(readability-identifier-naming)
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
                x[i] = Operon::Scalar{1};
                y[i] = Operon::Scalar{6};
            }
            std::vector<std::vector<Operon::Scalar>> cols{x, y};
            return Operon::Dataset(cols);
        }())
        , tree([&]() -> Tree {
            auto t = InfixParser::Parse("X1", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) { node.Value = static_cast<Operon::Scalar>(0.1); }
            }
            return t;
        }())
        , problem(&ds)
    {
        problem.SetTrainingRange({0, Nrow});
        problem.SetTestRange({0, Nrow});
        problem.SetTarget("X2");
        std::vector<Operon::Scalar> weights(Nrow, Operon::Scalar{1});
        std::fill(weights.begin() + Nclean, weights.end(), Operon::Scalar{0});
        ds.SetWeights(weights);
    }
};

TEST_CASE("Weighted parameter optimization", "[optimizer]")
{
    WeightedOptimizerFixture fix;
    auto& rng     = fix.rng;
    auto& tree    = fix.tree;
    auto& dtable  = fix.dtable;
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

    SECTION("lm / eigen") {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer{&dtable, &problem};
        checkRecoversCleanSolution(optimizer);
    }

    SECTION("lm / tiny") {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> optimizer{&dtable, &problem};
        checkRecoversCleanSolution(optimizer);
    }

#if defined(HAVE_ASMJIT)
    SECTION("lm / jit") {
        // JitLevenbergMarquardtOptimizer previously ignored Problem::Weights()
        // entirely (both its interpreter-fallback and JIT-compiled cost
        // function paths), so weighted LM behavior silently differed by
        // backend. Same discriminative fixture as "lm / eigen" above -
        // recovering c0=1 here requires the zeroed-weight rows to actually
        // be down-weighted by the JIT path too.
        Operon::JIT::JitZobrist zobrist{rng, 50, problem.GetInputs()};
        JIT::JitEvaluator jitEval{&problem, &zobrist};
        JitLevenbergMarquardtOptimizer<DTable> optimizer{&dtable, &problem, &jitEval};
        checkRecoversCleanSolution(optimizer);
    }
#endif

    SECTION("lbfgs / gaussian") {
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        checkRecoversCleanSolution(optimizer);
    }

    SECTION("sgd / gaussian") {
        auto const dim{tree.CoefficientsCount()};
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
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

    SECTION("lbfgs / gaussian: reported cost matches the weighted objective, not raw SSE") {
        // Directly pins down the root cause rather than relying on Success
        // to flip (which only happens for adversarial coefficient
        // trajectories - not guaranteed by every dataset/tolerance
        // combination): InitialCost/FinalCost must equal the *weighted* SSE
        // GaussianLoss actually optimizes, independently recomputed here,
        // not the unweighted SumOfSquaredErrors the cost lambda used before
        // the fix.
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());

        auto const range = problem.TrainingRange();
        auto const target = problem.TargetValues(range);
        auto const weights = *problem.Weights(range);
        Operon::Interpreter<Operon::Scalar, DTable> interpreter{&dtable, &fix.ds, &tree};

        auto const pred0 = interpreter.Evaluate(Operon::Span<Operon::Scalar const>{summary->InitialParameters}, range);
        auto const expectedInitialCost = 0.5 * Operon::SumOfSquaredErrors(pred0.begin(), pred0.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary->InitialCost), Catch::Matchers::WithinRel(expectedInitialCost, 1e-3));

        auto const pred1 = interpreter.Evaluate(Operon::Span<Operon::Scalar const>{summary->FinalParameters}, range);
        auto const expectedFinalCost = 0.5 * Operon::SumOfSquaredErrors(pred1.begin(), pred1.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary->FinalCost), Catch::Matchers::WithinRel(expectedFinalCost, 1e-3));
    }

    SECTION("lm / eigen: reported cost matches the weighted objective, not raw SSE") {
        // Symmetric with the "lbfgs / gaussian" cost check above, but for the
        // LM path: LMCostFunction applies the sqrt(w)-residual trick (see
        // lm_cost_function_base.hpp), so Eigen::LevenbergMarquardt's
        // fnorm()^2 * 0.5 already equals the weighted SSE / 2 - pin that down
        // directly rather than relying on it transitively via Success/params.
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer{&dtable, &problem};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());

        auto const range = problem.TrainingRange();
        auto const target = problem.TargetValues(range);
        auto const weights = *problem.Weights(range);
        Operon::Interpreter<Operon::Scalar, DTable> interpreter{&dtable, &fix.ds, &tree};

        auto const pred0 = interpreter.Evaluate(Operon::Span<Operon::Scalar const>{summary->InitialParameters}, range);
        auto const expectedInitialCost = 0.5 * Operon::SumOfSquaredErrors(pred0.begin(), pred0.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary->InitialCost), Catch::Matchers::WithinRel(expectedInitialCost, 1e-3));

        auto const pred1 = interpreter.Evaluate(Operon::Span<Operon::Scalar const>{summary->FinalParameters}, range);
        auto const expectedFinalCost = 0.5 * Operon::SumOfSquaredErrors(pred1.begin(), pred1.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary->FinalCost), Catch::Matchers::WithinRel(expectedFinalCost, 1e-3));
    }

    SECTION("lbfgs / gaussian: CoefficientOptimizer actually applies the weighted-optimal coefficients") {
        // End-to-end check through the real call path (local_search.cpp),
        // not just Optimize() directly: CoefficientOptimizer gates
        // SetCoefficients on the outcome having a value, so a mis-scored
        // outcome would silently discard a genuine weighted improvement here.
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        Operon::CoefficientOptimizer const coeffOptimizer{&optimizer};
        auto [optimizedTree, summary] = coeffOptimizer(rng, tree);
        REQUIRE(summary.has_value());
        auto const coeffs = optimizedTree.GetCoefficients();
        REQUIRE(!coeffs.empty());
        for (auto const c : coeffs) {
            CHECK_THAT(c, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    }

    SECTION("unweighted sanity check: LM does NOT recover c0=1") {
        // Confirms the test problem is actually discriminative - not that
        // "any optimizer converges to 1 regardless of weights".
        std::vector<Operon::Scalar> ones(WeightedOptimizerFixture::Nrow, Operon::Scalar{1});
        fix.problem.GetDataset()->SetWeights(ones);
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer{&dtable, &problem};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        auto const p = summary->FinalParameters.front();
        CHECK(std::abs(p - 1.0F) > paramTol);
    }

    SECTION("unweighted sanity check: lbfgs does NOT recover c0=1") {
        std::vector<Operon::Scalar> ones(WeightedOptimizerFixture::Nrow, Operon::Scalar{1});
        fix.problem.GetDataset()->SetWeights(ones);
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        auto const p = summary->FinalParameters.front();
        CHECK(std::abs(p - 1.0F) > paramTol);
    }

    SECTION("poisson ignores weights (documented limitation, not yet implemented)") {
        // PoissonLoss's constructor accepts a weights span only to share
        // LBFGSOptimizer's generic call site with GaussianLoss; it must
        // have zero effect on the result until the exposure-vs-precision
        // weight semantics are reconciled. Verified by re-running with an
        // all-ones weight vector (fresh rng, same seed) and checking the
        // result matches the zeroed-weight run to within float noise (not
        // exact ==: defensive against benign future changes to evaluation
        // order/precision elsewhere in the RNG/opt path that wouldn't
        // actually mean weights started being applied).
        LBFGSOptimizer<DTable, PoissonLoss<Operon::Scalar>> const optimizerZeroed{&dtable, &problem};
        Operon::RandomGenerator rngZeroed{0};
        auto summaryZeroed = optimizerZeroed.Optimize(rngZeroed, tree);
        REQUIRE(summaryZeroed.has_value());

        std::vector<Operon::Scalar> ones(WeightedOptimizerFixture::Nrow, Operon::Scalar{1});
        fix.problem.GetDataset()->SetWeights(ones);
        LBFGSOptimizer<DTable, PoissonLoss<Operon::Scalar>> const optimizerOnes{&dtable, &problem};
        Operon::RandomGenerator rngOnes{0};
        auto summaryOnes = optimizerOnes.Optimize(rngOnes, tree);
        REQUIRE(summaryOnes.has_value());

        REQUIRE(summaryZeroed->FinalParameters.size() == summaryOnes->FinalParameters.size());
        for (auto i = 0UL; i < summaryZeroed->FinalParameters.size(); ++i) {
            CHECK_THAT(summaryZeroed->FinalParameters[i], Catch::Matchers::WithinRel(summaryOnes->FinalParameters[i], 1e-5F));
        }
        CHECK_THAT(static_cast<double>(summaryZeroed->FinalCost), Catch::Matchers::WithinRel(static_cast<double>(summaryOnes->FinalCost), 1e-5));
    }
}

// Same clean/noisy problem as WeightedOptimizerFixture, but padded with
// Npad unused rows so the training range starts at a non-zero offset
// (range_.Start() = Npad). GaussianLoss::operator() previously indexed
// target_/weights_ by the *absolute* range.Start() instead of an offset
// relative to range_.Start(); since SelectRandomRange() returns range_
// itself whenever batchSize >= range_.Size() (the default), this reproduces
// the out-of-bounds read even without any random sub-batching - LBFGS/SGD
// with a non-zero training-range start alone was enough to trigger it.
// LM is unaffected (LMCostFunction always indexes 0..numResiduals_-1,
// never range.Start()), so this only needs to cover LBFGS/SGD.
struct WeightedOptimizerNonZeroStartFixture {
    static constexpr auto Npad { 100 };
    static constexpr auto Nrow { 400 };
    static constexpr auto Nclean { Nrow / 2 };
    static constexpr auto Ntotal { Npad + Nrow };

    Operon::RandomGenerator rng{0}; // NOLINT(readability-identifier-naming)
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
                x[i] = Operon::Scalar{-2}; // never read - outside training/test range
                y[i] = Operon::Scalar{100};
            }
            for (auto i = 0; i < Nclean; ++i) {
                x[Npad + i] = Operon::Random::Uniform(rng, -1.0F, +1.0F);
                y[Npad + i] = x[Npad + i];
            }
            for (auto i = Nclean; i < Nrow; ++i) {
                x[Npad + i] = Operon::Scalar{1};
                y[Npad + i] = Operon::Scalar{6};
            }
            std::vector<std::vector<Operon::Scalar>> cols{x, y};
            return Operon::Dataset(cols);
        }())
        , tree([&]() -> Tree {
            auto t = InfixParser::Parse("X1", ds);
            for (auto& node : t.Nodes()) {
                if (node.IsVariable()) { node.Value = static_cast<Operon::Scalar>(0.1); }
            }
            return t;
        }())
        , problem(&ds)
    {
        problem.SetTrainingRange({Npad, Ntotal});
        problem.SetTestRange({Npad, Ntotal});
        problem.SetTarget("X2");
        std::vector<Operon::Scalar> weights(Ntotal, Operon::Scalar{1});
        std::fill(weights.begin() + Npad + Nclean, weights.end(), Operon::Scalar{0});
        ds.SetWeights(weights);
    }
};

TEST_CASE("Weighted parameter optimization with non-zero training range start", "[optimizer]")
{
    WeightedOptimizerNonZeroStartFixture fix;
    auto& rng     = fix.rng;
    auto& tree    = fix.tree;
    auto& dtable  = fix.dtable;
    auto& problem = fix.problem;
    using DTable = WeightedOptimizerNonZeroStartFixture::DTable;

    constexpr Operon::Scalar paramTol { 0.01F };

    SECTION("lbfgs / gaussian") {
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    }

    SECTION("sgd / gaussian") {
        auto const dim{tree.CoefficientsCount()};
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, 0.1F));
        }
    }

    SECTION("lbfgs / gaussian: negative placeholder weights outside the training range don't trip validation") {
        // GaussianLoss now receives the whole-dataset weights column (not a
        // slice pre-cut to the training range), so its weight-sign check
        // must only validate the in-range slice - rows outside range_ (e.g.
        // padding, or other splits' rows) are never read by SelectBatch and
        // may legitimately carry negative/sentinel values.
        std::vector<Operon::Scalar> weights(WeightedOptimizerNonZeroStartFixture::Ntotal, Operon::Scalar{1});
        std::fill(weights.begin(), weights.begin() + WeightedOptimizerNonZeroStartFixture::Npad, Operon::Scalar{-1});
        std::fill(weights.begin() + WeightedOptimizerNonZeroStartFixture::Npad + WeightedOptimizerNonZeroStartFixture::Nclean, weights.end(), Operon::Scalar{0});
        fix.problem.GetDataset()->SetWeights(weights);

        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        auto summary = optimizer.Optimize(rng, tree);
        REQUIRE(summary.has_value());
        for (auto const p : summary->FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    }
}

TEST_CASE("PoissonLoss respects a non-zero training range start", "[optimizer]")
{
    // Same off-by-range_.Start() bug class fixed in GaussianLoss, but
    // exercised directly on PoissonLoss::operator() rather than through a
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
        for (auto i = 0; i < pad; ++i) { x[i] = Operon::Scalar{999}; y[i] = Operon::Scalar{999}; } // never read
        for (auto i = 0; i < Nrow; ++i) {
            x[pad + i] = static_cast<Operon::Scalar>(i + 1) * 0.1F;
            y[pad + i] = static_cast<Operon::Scalar>(i + 1);
        }
        std::vector<std::vector<Operon::Scalar>> cols{x, y};
        return Operon::Dataset(cols);
    };

    auto ds0 = build(0);
    auto dsPad = build(Npad);

    auto tree0 = InfixParser::Parse("X1", ds0);
    auto treePad = InfixParser::Parse("X1", dsPad);
    for (auto* t : {&tree0, &treePad}) {
        for (auto& node : t->Nodes()) {
            if (node.IsVariable()) { node.Value = Operon::Scalar{1}; }
        }
    }

    DTable dtable;

    Operon::Problem problem0(&ds0);
    problem0.SetTrainingRange({0, Nrow});
    problem0.SetTarget("X2");

    Operon::Problem problemPad(&dsPad);
    problemPad.SetTrainingRange({Npad, Npad + Nrow});
    problemPad.SetTarget("X2");

    Operon::Interpreter<Operon::Scalar, DTable> interp0{&dtable, &ds0, &tree0};
    Operon::Interpreter<Operon::Scalar, DTable> interpPad{&dtable, &dsPad, &treePad};

    Operon::RandomGenerator rng0{0};
    Operon::RandomGenerator rngPad{0};

    // PoissonLoss now requires the whole dataset column (absolute,
    // dataset-row-indexed), not a slice pre-cut to the training range - see
    // its constructor comment.
    auto target0 = problem0.TargetValues();
    auto targetPad = problemPad.TargetValues();

    PoissonLoss<Operon::Scalar> loss0{&rng0, &interp0, target0, problem0.TrainingRange()};
    PoissonLoss<Operon::Scalar> lossPad{&rngPad, &interpPad, targetPad, problemPad.TrainingRange()};

    auto coeff = tree0.GetCoefficients();
    REQUIRE(!coeff.empty());
    Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));
    Eigen::Matrix<Operon::Scalar, -1, 1> grad0(coeff.size());
    Eigen::Matrix<Operon::Scalar, -1, 1> gradPad(coeff.size());

    auto const c0 = loss0(x0, grad0);
    auto const cPad = lossPad(x0, gradPad);

    CHECK_THAT(static_cast<double>(c0), Catch::Matchers::WithinRel(static_cast<double>(cPad), 1e-5));
    for (auto i = 0; i < grad0.size(); ++i) {
        CHECK_THAT(static_cast<double>(grad0(i)), Catch::Matchers::WithinRel(static_cast<double>(gradPad(i)), 1e-5));
    }
}

#if defined(OPERON_HAVE_HIP)
TEST_CASE("HIP population local search preserves the delivery contract", "[optimizer][hip][population-local-search]")
{
    OptimizerFixture fix;
    std::vector<Operon::Individual> population(2, Operon::Individual{1});
    population[0].Genotype = fix.tree;
    population[1].Genotype = fix.tree;
    auto const alternateCoefficients = std::vector<Operon::Scalar>{0.2F, 0.2F, 0.2F};
    population[1].Genotype.SetCoefficients(alternateCoefficients);
    auto const selected = std::vector<std::size_t>{1, 0};
    auto encoded = Operon::PopulationOptimization::EncodePopulation(population, selected);
    REQUIRE(encoded.has_value());

    auto const variables = fix.problem.GetInputs();
    std::vector<Operon::Scalar> columns;
    columns.reserve(variables.size() * OptimizerFixture::Nrow);
    for (auto const hash : variables) {
        auto const values = fix.ds.GetValues(hash);
        columns.insert(columns.end(), values.begin(), values.end());
    }

    Operon::PopulationOptimization::Hip::Context hip;
    REQUIRE(hip.Supports(fix.tree, variables));
    auto const result = hip.Optimize(*encoded, variables, columns, variables.size(), OptimizerFixture::Nrow,
                                     fix.problem.TargetValues(fix.problem.TrainingRange()), {}, 8);

    REQUIRE(result.Status.size() == selected.size());
    REQUIRE(result.InitialCosts.size() == selected.size());
    REQUIRE(result.FinalCosts.size() == selected.size());
    REQUIRE(result.Iterations.size() == selected.size());
    REQUIRE(result.AcceptedSteps.size() == selected.size());
    REQUIRE(result.Coefficients.size() == encoded->Coefficients.size());
    for (auto i = std::size_t{}; i < selected.size(); ++i) {
        CHECK(result.Status[i] == Operon::PopulationLocalSearchStatus::Improved);
        CHECK(std::isfinite(result.InitialCosts[i]));
        CHECK(std::isfinite(result.FinalCosts[i]));
        CHECK(result.FinalCosts[i] < result.InitialCosts[i]);
        auto const& tree = encoded->Trees[i];
        for (auto const coefficient : std::span{result.Coefficients}.subspan(tree.CoefficientOffset, tree.CoefficientCount)) {
            CHECK_THAT(coefficient, Catch::Matchers::WithinAbs(1.0F, 0.01F));
        }
    }
}

TEST_CASE("HIP resident population preserves topology and updates coefficients", "[optimizer][hip][resident]")
{
    OptimizerFixture fix;
    std::vector<Operon::Individual> population(1, Operon::Individual{1});
    population.front().Genotype = fix.tree;
    auto encoded = Operon::PopulationOptimization::EncodePopulation(population, std::vector<std::size_t>{0});
    REQUIRE(encoded.has_value());
    auto const variables = fix.problem.GetInputs();
    std::vector<Operon::Scalar> columns;
    for (auto const hash : variables) {
        auto const values = fix.ds.GetValues(hash);
        columns.insert(columns.end(), values.begin(), values.end());
    }
    Operon::PopulationOptimization::Hip::Context hip;
    hip.Upload(*encoded, variables);
    auto const initial = hip.EvaluateResident(columns, variables.size(), OptimizerFixture::Nrow);
    auto updated = encoded->Coefficients;
    std::fill(updated.begin(), updated.end(), Operon::Scalar{1});
    hip.UpdateCoefficients(updated);
    auto const resident = hip.EvaluateResident(columns, variables.size(), OptimizerFixture::Nrow);
    auto const explicitResult = hip.Evaluate(columns, variables.size(), OptimizerFixture::Nrow, updated);
    REQUIRE(resident.size() == explicitResult.size());
    CHECK(resident != initial);
    for (std::size_t row = 0; row < resident.size(); ++row) {
        CHECK_THAT(resident[row], Catch::Matchers::WithinAbs(explicitResult[row], 1e-5F));
    }
    auto const optimized = hip.OptimizeGaussian(columns, variables.size(), OptimizerFixture::Nrow,
                                                fix.problem.TargetValues(fix.problem.TrainingRange()), {}, 8);
    auto const optimizedResident = hip.EvaluateResident(columns, variables.size(), OptimizerFixture::Nrow);
    auto const optimizedExplicit = hip.Evaluate(columns, variables.size(), OptimizerFixture::Nrow, optimized.Coefficients);
    REQUIRE(optimizedResident.size() == optimizedExplicit.size());
    for (std::size_t row = 0; row < optimizedResident.size(); ++row) {
        CHECK_THAT(optimizedResident[row], Catch::Matchers::WithinAbs(optimizedExplicit[row], 1e-5F));
    }
}

TEST_CASE("HIP Gaussian population scorer matches scalar evaluation", "[optimizer][hip][population-scorer]")
{
    OptimizerFixture fix;
    Operon::PopulationOptimization::Hip::Context context;
    fix.problem.SetLinearScalingEnabled(false);
    Operon::Evaluator<OptimizerFixture::DTable> evaluator{&fix.problem, &fix.dtable, Operon::MSE{}};
    Operon::PopulationOptimization::Hip::GaussianPopulationOffspringScorer scorer{context};
    std::vector<Operon::Individual> candidates(1, Operon::Individual{1});
    candidates.front().Genotype = fix.tree;
    std::vector<Operon::RandomGenerator> random;
    random.emplace_back(1234);
    std::vector<Operon::Vector<Operon::Scalar>> scratch(1);
    scorer.Score(candidates, random, evaluator, nullptr, 0.0, 1.0, scratch);

    std::vector<Operon::Scalar> scalarScratch(OptimizerFixture::Nrow);
    Operon::Individual expected{1};
    expected.Genotype = fix.tree;
    Operon::RandomGenerator scalarRandom{1234};
    Operon::ScoreIndividual(scalarRandom, expected, evaluator, nullptr, 0.0, 1.0, scalarScratch);
    REQUIRE(candidates.front().Fitness.size() == 1);
    CHECK_THAT(candidates.front().Fitness.front(), Catch::Matchers::WithinAbs(expected.Fitness.front(), 1e-5F));
}

TEST_CASE("HIP Gaussian costs match scalar reduction across tiled row boundaries", "[optimizer][hip][population-scorer]")
{
    constexpr auto TreeCount = std::size_t{2};
    for (auto const rows : {std::size_t{4096}, std::size_t{4097}}) {
        DYNAMIC_SECTION("rows=" << rows) {
            std::vector<Operon::Scalar> x1(rows);
            std::vector<Operon::Scalar> x2(rows);
            std::vector<Operon::Scalar> target(rows);
            std::vector<Operon::Scalar> weights(rows);
            for (auto row = std::size_t{}; row < rows; ++row) {
                x1[row] = static_cast<Operon::Scalar>(row % 13) * 0.1F;
                x2[row] = static_cast<Operon::Scalar>(row % 17) * -0.05F;
                target[row] = x1[row] + x2[row];
                weights[row] = row % 3 == 0 ? 0.25F : 1.0F;
            }
            Operon::Dataset dataset{std::vector<std::vector<Operon::Scalar>>{x1, x2, target}};
            auto tree = InfixParser::Parse("X1 + X2", dataset);
            for (auto& node : tree.Nodes()) {
                if (node.IsVariable()) { node.Value = 0.1F; }
            }
            std::vector<Operon::Individual> population(TreeCount, Operon::Individual{1});
            for (auto& individual : population) { individual.Genotype = tree; }
            auto encoded = Operon::PopulationOptimization::EncodePopulation(population, std::vector<std::size_t>{0, 1});
            REQUIRE(encoded.has_value());

            std::vector<Operon::Scalar> columns;
            columns.reserve(rows * 2);
            columns.insert(columns.end(), x1.begin(), x1.end());
            columns.insert(columns.end(), x2.begin(), x2.end());
            Operon::PopulationOptimization::Hip::Context hip;
            hip.Upload(*encoded, std::array{dataset.GetVariable("X1")->Hash, dataset.GetVariable("X2")->Hash});
            auto const [costs, valid] = hip.GaussianCosts(columns, 2, rows, target, weights);
            REQUIRE(costs.size() == TreeCount);
            REQUIRE(valid == std::vector<uint8_t>(TreeCount, 1));

            Operon::Scalar expected{};
            for (auto row = std::size_t{}; row < rows; ++row) {
                auto const residual = 0.9F * (x1[row] + x2[row]);
                expected += 0.5F * weights[row] * residual * residual;
            }
            for (auto const cost : costs) { CHECK_THAT(cost, Catch::Matchers::WithinRel(expected, 1e-5F)); }
        }
    }
}

TEST_CASE("HIP local search replays the shared corpus", "[optimizer][hip][population-local-search][corpus]")
{
    Operon::Dataset dataset{"./data/PopulationLocalSearch.csv", /*hasHeader=*/true};
    Operon::Problem problem{&dataset};
    problem.SetTarget("Y");
    problem.SetInputs(std::vector<std::string>{"X1", "X2", "X3"});
    problem.SetTrainingRange({0, dataset.Rows<std::size_t>()});
    problem.SetLinearScalingEnabled(false);

    auto tree = InfixParser::Parse("X1 + X2 + X3", dataset);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) { node.Value = 0.1F; }
    }
    std::vector<Operon::Individual> population(1, Operon::Individual{1});
    population.front().Genotype = tree;
    auto const selected = std::vector<std::size_t>{0};
    auto encoded = Operon::PopulationOptimization::EncodePopulation(population, selected);
    REQUIRE(encoded.has_value());

    auto const variables = problem.GetInputs();
    std::vector<Operon::Scalar> columns;
    columns.reserve(variables.size() * problem.TrainingRange().Size());
    for (auto const hash : variables) {
        auto const values = dataset.GetValues(hash);
        columns.insert(columns.end(), values.begin(), values.end());
    }

    Operon::PopulationOptimization::Hip::Context hip;
    auto const result = hip.Optimize(*encoded, variables, columns, variables.size(), problem.TrainingRange().Size(),
                                     problem.TargetValues(problem.TrainingRange()), {}, 8);
    REQUIRE(result.Status == std::vector{Operon::PopulationLocalSearchStatus::Improved});
    REQUIRE(result.InitialCosts.size() == 1);
    REQUIRE(result.FinalCosts.size() == 1);
    CHECK(result.FinalCosts.front() < result.InitialCosts.front());
}
#endif

#if defined(OPERON_HAVE_SYCL)
TEST_CASE("SYCL local search replays the shared corpus", "[optimizer][sycl][population-local-search][corpus]")
{
    Operon::Dataset dataset{"./data/PopulationLocalSearch.csv", /*hasHeader=*/true};
    Operon::Problem problem{&dataset};
    problem.SetTarget("Y");
    problem.SetInputs(std::vector<std::string>{"X1", "X2", "X3"});
    problem.SetTrainingRange({0, dataset.Rows<std::size_t>()});
    problem.SetLinearScalingEnabled(false);

    auto tree = InfixParser::Parse("X1 + X2 + X3", dataset);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) { node.Value = 0.1F; }
    }
    std::vector<Operon::Individual> population(1, Operon::Individual{1});
    population.front().Genotype = tree;
    auto const selected = std::vector<std::size_t>{0};
    auto encoded = Operon::PopulationOptimization::EncodePopulation(population, selected);
    REQUIRE(encoded.has_value());

    auto const variables = problem.GetInputs();
    std::vector<Operon::Scalar> columns;
    columns.reserve(variables.size() * problem.TrainingRange().Size());
    for (auto const hash : variables) {
        auto const values = dataset.GetValues(hash);
        columns.insert(columns.end(), values.begin(), values.end());
    }

    try {
        Operon::PopulationOptimization::Sycl::Context sycl;
        REQUIRE(sycl.Supports(tree, variables));
        auto const result = sycl.Optimize(*encoded, variables, columns, variables.size(), problem.TrainingRange().Size(),
                                          problem.TargetValues(problem.TrainingRange()), {}, 8);
        REQUIRE(result.Status == std::vector{Operon::PopulationLocalSearchStatus::Improved});
        REQUIRE(result.InitialCosts.size() == 1);
        REQUIRE(result.FinalCosts.size() == 1);
        CHECK(result.FinalCosts.front() < result.InitialCosts.front());
    } catch (std::exception const& error) {
        if (std::string_view{error.what()} == "No matching device") { SKIP("requires a SYCL GPU device"); }
        throw;
    }
}
#endif
#if defined(OPERON_HAVE_HIP) || defined(OPERON_HAVE_SYCL)
TEST_CASE("Accelerator population evaluation matches the shared corpus", "[optimizer][population-evaluation]")
{
    Operon::Dataset dataset{"./data/PopulationLocalSearch.csv", /*hasHeader=*/true};
    Operon::Problem problem{&dataset};
    problem.SetTarget("Y");
    problem.SetInputs(std::vector<std::string>{"X1", "X2", "X3"});
    problem.SetTrainingRange({0, dataset.Rows<std::size_t>()});

    auto tree = InfixParser::Parse("X1 + X2 + X3", dataset);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) { node.Value = 0.1F; }
    }
    std::vector<Operon::Individual> population(1, Operon::Individual{1});
    population.front().Genotype = tree;
    auto encoded = Operon::PopulationOptimization::EncodePopulation(population, std::vector<std::size_t>{0});
    REQUIRE(encoded.has_value());

    auto const variables = problem.GetInputs();
    std::vector<Operon::Scalar> columns;
    columns.reserve(variables.size() * problem.TrainingRange().Size());
    for (auto const hash : variables) {
        auto const values = dataset.GetValues(hash);
        columns.insert(columns.end(), values.begin(), values.end());
    }
    Operon::ScalarDispatch dtable;
    Operon::Interpreter<Operon::Scalar, Operon::ScalarDispatch> interpreter{&dtable, &dataset, &tree};
    auto const expected = interpreter.Evaluate(tree.GetCoefficients(), problem.TrainingRange());

#if defined(OPERON_HAVE_HIP)
    SECTION("HIP") {
        Operon::PopulationOptimization::Hip::Context hip;
        hip.Upload(*encoded, variables);
        auto const output = hip.Evaluate(columns, variables.size(), problem.TrainingRange().Size(), encoded->Coefficients);
        REQUIRE(output.size() == expected.size());
        for (std::size_t row = 0; row < output.size(); ++row) {
            CHECK_THAT(output[row], Catch::Matchers::WithinAbs(expected[row], 1e-5F));
        }
    }
#endif
#if defined(OPERON_HAVE_SYCL)
    SECTION("SYCL") {
        Operon::PopulationOptimization::Sycl::Context sycl;
        sycl.Upload(*encoded, variables);
        auto const output = sycl.Evaluate(columns, variables.size(), problem.TrainingRange().Size(), encoded->Coefficients);
        REQUIRE(output.size() == expected.size());
        for (std::size_t row = 0; row < output.size(); ++row) {
            CHECK_THAT(output[row], Catch::Matchers::WithinAbs(expected[row], 1e-5F));
        }
    }
#endif
}
#endif

TEST_CASE("SGD update rules", "[optimizer]")
{
    OptimizerFixture fix;
    auto& rng     = fix.rng;
    auto& tree    = fix.tree;
    auto& dtable  = fix.dtable;
    auto& problem = fix.problem;
    using DTable = OptimizerFixture::DTable;

    auto const dim{tree.CoefficientsCount()};

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
        SGDOptimizer<DTable, GaussianLoss<Operon::Scalar>> const optimizer{&dtable, &problem, *rule};
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

} // namespace Operon::Test
