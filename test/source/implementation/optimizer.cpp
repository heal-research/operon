// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include "../operon_test.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/solvers/sgd.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"
#include "operon/operators/evaluator.hpp"

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
            return Operon::Dataset(data);
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

TEST_CASE("Parameter optimization", "[optimizer]")
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

    SECTION("ceres solver") {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Ceres> optimizer{&dtable, &problem};
        checkExact(optimizer);
    }

    SECTION("lbfgs / gaussian") {
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        checkExact(optimizer);
    }

    SECTION("lbfgs / poisson") {
        // Poisson loss on a continuous target: just verify it runs and improves
        LBFGSOptimizer<DTable, PoissonLoss<Operon::Scalar>> optimizer{&dtable, &problem};
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
        SGDOptimizer<DTable, PoissonLoss<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        auto summary = optimizer.Optimize(rng, tree);
        CHECK(std::isfinite(summary.FinalCost));
        CHECK(!summary.FinalParameters.empty());
    }

    SECTION("ComputeLikelihood virtual dispatch: pred==target => NLL = n/2 * log(2pi)") {
        LBFGSOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem};
        auto range = problem.TrainingRange();
        auto target = problem.TargetValues(range);
        std::vector<Operon::Scalar> sigma(1, 1.0F);
        auto nll = optimizer.ComputeLikelihood(target, target, sigma);
        auto expected = static_cast<double>(range.Size()) / 2.0 * std::log(Operon::Math::Tau);
        CHECK_THAT(static_cast<double>(nll), Catch::Matchers::WithinRel(expected, 1e-4));
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
        SGDOptimizer<DTable, GaussianLoss<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        auto summary = optimizer.Optimize(rng, tree);
        CHECK(summary.Iterations > 0);
        // YamAdam applies the raw (summed) gradient as its first step (step size ≈ 1),
        // which overshoots on unnormalized losses with large n. We only require finite output.
        CHECK(std::isfinite(summary.FinalCost));
    }
}

} // namespace Operon::Test
