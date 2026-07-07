// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
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
        CHECK(summary.FinalCost < summary.InitialCost);
        CHECK(summary.FinalCost < tightTol);
        for (auto const p : summary.FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    };

    auto checkImproved = [&](OptimizerBase& optimizer) -> void {
        auto summary = optimizer.Optimize(rng, tree);
        CHECK(summary.FinalCost < summary.InitialCost);
        CHECK(summary.FinalCost < looseTol);
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
        CHECK(std::isfinite(summary.FinalCost));
        CHECK(!summary.FinalParameters.empty());
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
        CHECK(std::isfinite(summary.FinalCost));
        CHECK(!summary.FinalParameters.empty());
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
        // Success must reflect the *weighted* objective actually optimized -
        // CoefficientOptimizer (local_search.cpp) only applies FinalParameters
        // when Success is true, so a false negative here would silently drop
        // a real weighted improvement.
        CHECK(summary.Success);
        for (auto const p : summary.FinalParameters) {
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

    SECTION("lbfgs / gaussian") {
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        checkRecoversCleanSolution(optimizer);
    }

    SECTION("sgd / gaussian") {
        auto const dim{tree.CoefficientsCount()};
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        auto summary = optimizer.Optimize(rng, tree);
        CHECK(summary.Success);
        // SGD converges more slowly than LM/L-BFGS on this problem within
        // the default iteration budget, so use a looser tolerance - the
        // point is confirming weights are picked up at all, not tight
        // convergence.
        for (auto const p : summary.FinalParameters) {
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

        auto const range = problem.TrainingRange();
        auto const target = problem.TargetValues(range);
        auto const weights = *problem.Weights(range);
        Operon::Interpreter<Operon::Scalar, DTable> interpreter{&dtable, &fix.ds, &tree};

        auto const pred0 = interpreter.Evaluate(Operon::Span<Operon::Scalar const>{summary.InitialParameters}, range);
        auto const expectedInitialCost = 0.5 * Operon::SumOfSquaredErrors(pred0.begin(), pred0.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary.InitialCost), Catch::Matchers::WithinRel(expectedInitialCost, 1e-3));

        auto const pred1 = interpreter.Evaluate(Operon::Span<Operon::Scalar const>{summary.FinalParameters}, range);
        auto const expectedFinalCost = 0.5 * Operon::SumOfSquaredErrors(pred1.begin(), pred1.end(), target.begin(), weights.begin());
        CHECK_THAT(static_cast<double>(summary.FinalCost), Catch::Matchers::WithinRel(expectedFinalCost, 1e-3));
    }

    SECTION("lbfgs / gaussian: CoefficientOptimizer actually applies the weighted-optimal coefficients") {
        // End-to-end check through the real call path (local_search.cpp),
        // not just Optimize() directly: CoefficientOptimizer gates
        // SetCoefficients on summary.Success, so a mis-scored Success would
        // silently discard a genuine weighted improvement here.
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        Operon::CoefficientOptimizer const coeffOptimizer{&optimizer};
        auto [optimizedTree, summary] = coeffOptimizer(rng, tree);
        REQUIRE(summary.Success);
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
        auto const p = summary.FinalParameters.front();
        CHECK(std::abs(p - 1.0F) > paramTol);
    }

    SECTION("unweighted sanity check: lbfgs does NOT recover c0=1") {
        std::vector<Operon::Scalar> ones(WeightedOptimizerFixture::Nrow, Operon::Scalar{1});
        fix.problem.GetDataset()->SetWeights(ones);
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        auto summary = optimizer.Optimize(rng, tree);
        auto const p = summary.FinalParameters.front();
        CHECK(std::abs(p - 1.0F) > paramTol);
    }

    SECTION("poisson ignores weights (documented limitation, not yet implemented)") {
        // PoissonLoss's constructor accepts a weights span only to share
        // LBFGSOptimizer's generic call site with GaussianLoss; it must
        // have zero effect on the result until the exposure-vs-precision
        // weight semantics are reconciled. Verified by re-running with an
        // all-ones weight vector (fresh rng, same seed) and checking the
        // result is bit-for-bit identical to the zeroed-weight run.
        LBFGSOptimizer<DTable, PoissonLoss<Operon::Scalar>> const optimizerZeroed{&dtable, &problem};
        Operon::RandomGenerator rngZeroed{0};
        auto summaryZeroed = optimizerZeroed.Optimize(rngZeroed, tree);

        std::vector<Operon::Scalar> ones(WeightedOptimizerFixture::Nrow, Operon::Scalar{1});
        fix.problem.GetDataset()->SetWeights(ones);
        LBFGSOptimizer<DTable, PoissonLoss<Operon::Scalar>> const optimizerOnes{&dtable, &problem};
        Operon::RandomGenerator rngOnes{0};
        auto summaryOnes = optimizerOnes.Optimize(rngOnes, tree);

        REQUIRE(summaryZeroed.FinalParameters.size() == summaryOnes.FinalParameters.size());
        for (auto i = 0UL; i < summaryZeroed.FinalParameters.size(); ++i) {
            CHECK(summaryZeroed.FinalParameters[i] == summaryOnes.FinalParameters[i]);
        }
        CHECK(summaryZeroed.FinalCost == summaryOnes.FinalCost);
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
        for (auto const p : summary.FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, paramTol));
        }
    }

    SECTION("sgd / gaussian") {
        auto const dim{tree.CoefficientsCount()};
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        auto summary = optimizer.Optimize(rng, tree);
        for (auto const p : summary.FinalParameters) {
            CHECK_THAT(p, Catch::Matchers::WithinAbs(1.0F, 0.1F));
        }
    }
}

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
        CHECK(summary.Iterations > 0);
        // YamAdam applies the raw (summed) gradient as its first step (step size ≈ 1),
        // which overshoots on unnormalized losses with large n. We only require finite output.
        CHECK(std::isfinite(summary.FinalCost));
    }
}

} // namespace Operon::Test
