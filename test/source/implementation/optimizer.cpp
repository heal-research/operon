// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

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

TEST_CASE("Parameter optimization", "[optimizer]")
{
    Operon::RandomGenerator rng{0};
    constexpr auto nrow{500};
    constexpr auto ncol{7};
    auto range = Range{0, nrow};

    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (auto i = 0; i < ncol; ++i) {
        auto col = data.col(i);
        std::generate(col.begin(), col.end(), [&]() { return Operon::Random::Uniform(rng, -1.0F, +1.0F); });
    }

    auto x1 = data.col(0);
    auto x2 = data.col(1);
    auto x3 = data.col(2);
    auto x4 = data.col(3);
    auto x5 = data.col(4);
    auto x6 = data.col(5);

    data.col(ncol - 1) = x1 * x2 + x3 * x4 + x5 * x6;

    Operon::Dataset ds(data);
    Operon::Map<std::string, Operon::Hash> vars;
    for (auto v : ds.GetVariables()) {
        vars[v.Name] = v.Hash;
    }

    auto tree = InfixParser::Parse("X1 * X2 + X3 * X4 + X5 * X6", vars);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) {
            node.Value = static_cast<Operon::Scalar>(0.01);
        } // NOLINT
    }

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;

    Operon::Problem problem{&ds};
    problem.SetTrainingRange(range);
    problem.SetTestRange(range);

    auto testOptimizer = [&](OptimizerBase& optimizer) {
        auto summary = optimizer.Optimize(rng, tree);
        CHECK(std::isfinite(summary.FinalCost));
        CHECK(!summary.FinalParameters.empty());
    };

    SECTION("tiny solver") {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> optimizer{&dtable, &problem};
        testOptimizer(optimizer);
    }

    SECTION("eigen solver") {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer{&dtable, &problem};
        testOptimizer(optimizer);
    }

    SECTION("ceres solver") {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Ceres> optimizer{&dtable, &problem};
        testOptimizer(optimizer);
    }

    SECTION("lbfgs / gaussian") {
        LBFGSOptimizer<DTable, GaussianLikelihood<Operon::Scalar>> optimizer{&dtable, &problem};
        testOptimizer(optimizer);
    }

    SECTION("lbfgs / poisson") {
        LBFGSOptimizer<DTable, PoissonLikelihood<Operon::Scalar>> optimizer{&dtable, &problem};
        testOptimizer(optimizer);
    }

    SECTION("sgd / gaussian") {
        auto const dim{tree.CoefficientsCount()};
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, GaussianLikelihood<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        testOptimizer(optimizer);
    }

    SECTION("sgd / poisson") {
        auto const dim{tree.CoefficientsCount()};
        auto rule = std::make_unique<UpdateRule::Adam<Operon::Scalar>>(dim);
        SGDOptimizer<DTable, PoissonLikelihood<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        testOptimizer(optimizer);
    }
}

TEST_CASE("SGD update rules", "[optimizer]")
{
    Operon::RandomGenerator rng{42};
    constexpr auto nrow{500};
    constexpr auto ncol{7};
    auto range = Range{0, nrow};

    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (auto i = 0; i < ncol; ++i) {
        auto col = data.col(i);
        std::generate(col.begin(), col.end(), [&]() { return Operon::Random::Uniform(rng, -1.0F, +1.0F); });
    }
    data.col(ncol - 1) = data.col(0) * data.col(1) + data.col(2) * data.col(3) + data.col(4) * data.col(5);

    Operon::Dataset ds(data);
    Operon::Map<std::string, Operon::Hash> vars;
    for (auto v : ds.GetVariables()) {
        vars[v.Name] = v.Hash;
    }

    auto tree = InfixParser::Parse("X1 * X2 + X3 * X4 + X5 * X6", vars);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) { node.Value = static_cast<Operon::Scalar>(0.01); }
    }

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;

    Operon::Problem problem{&ds};
    problem.SetTrainingRange(range);
    problem.SetTestRange(range);

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
        SGDOptimizer<DTable, GaussianLikelihood<Operon::Scalar>> optimizer{&dtable, &problem, *rule};
        auto summary = optimizer.Optimize(rng, tree);
        CHECK(summary.Iterations > 0);
    }
}

} // namespace Operon::Test
