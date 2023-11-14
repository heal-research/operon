// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
//
#include "../operon_test.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/solvers/sgd.hpp"
#include "operon/parser/infix.hpp"
#include <doctest/doctest.h>
#include <utility>

namespace Operon::Test {

TEST_CASE("Evaluation correctness")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };

    using DTable = DispatchTable<Operon::Scalar>;
    auto const& X = ds.Values(); // NOLINT

    Operon::Map<std::string, Operon::Hash> vars;
    for (auto const& v : ds.GetVariables()) {
        fmt::print("{} : {} {}\n", v.Name, v.Hash, v.Index);
        vars[v.Name] = v.Hash;
    }

    std::vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    DTable dtable;

    SUBCASE("Basic operations")
    {
        const auto eps = 1e-6;

        auto tree = InfixParser::Parse("X1 + X2 + X3", vars);
        auto coeff = tree.GetCoefficients();
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(coeff, range);
        auto res1 = X.col(0) + X.col(1) + X.col(2);

        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res1(i)) < eps; }));

        tree = InfixParser::Parse("X1 - X2 + X3", vars);
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        auto res2 = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res2(i)) < eps; }));

        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        auto res3 = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res3(i)) < eps; }));

        Operon::Node node(Operon::NodeType::Fmax);
        node.Arity = 3;
        auto a = Operon::Node::Constant(2);
        auto b = Operon::Node::Constant(3);
        auto c = Operon::Node::Constant(4);
        tree = Operon::Tree({ a, b, c, node });
        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        CHECK(estimatedValues[0] == 4);

        node = Operon::Node(Operon::NodeType::Sub);
        node.Arity = 1;
        tree = Operon::Tree({ Operon::Node::Constant(2), node });
        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = Interpreter<Operon::Scalar, DTable>(dtable, ds, tree).Evaluate(tree.GetCoefficients(), range);
        CHECK(estimatedValues[0] == -2);
    }
}

TEST_CASE("Batch evaluation")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };

    Operon::Problem problem{ds, range, range};
    Operon::PrimitiveSet pset{PrimitiveSet::Arithmetic};
    Operon::BalancedTreeCreator creator{pset, ds.VariableHashes()};

    Operon::RandomGenerator rng{0};
    auto constexpr n{10};

    std::vector<Operon::Tree> trees;
    std::vector<Operon::Scalar> result(range.Size() * n);    for (auto i = 0; i < n; ++i) {
        trees.push_back(creator(rng, 20, 10, 20));
    }

    Operon::EvaluateTrees(trees, ds, range, {result.data(), result.size()});
    Operon::EvaluateTrees(trees, ds, range);
}

TEST_CASE("parameter optimization")
{
    Operon::RandomGenerator rng{0};
    constexpr auto nrow{500};
    constexpr auto ncol{7};
    auto range = Range { 0, nrow };

    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (auto i = 0; i < ncol; ++i) {
        auto col = data.col(i);
        std::generate(col.begin(), col.end(), [&](){ return Operon::Random::Uniform(rng, -1.0F, +1.0F); });
    }

    // input variables
    auto x1 = data.col(0);
    auto x2 = data.col(1);
    auto x3 = data.col(2);
    auto x4 = data.col(3);
    auto x5 = data.col(4);
    auto x6 = data.col(5);

    // target variable
    data.col(ncol-1) = x1 * x2 + x3 * x4 + x5  * x6;

    Operon::Dataset ds(data);
    Operon::Map<std::string, Operon::Hash> vars;
    for (auto v : ds.GetVariables()) {
        // fmt::print("{} : {}\n", v.Name, v.Hash);
        vars[v.Name] = v.Hash;
    }

    Eigen::Array<Operon::Scalar, -1, 1> res = x1 * x2 + x3 * x4 + x5 * x6;
    Operon::Span<Operon::Scalar> target(res.data(), res.size());
    auto tree = InfixParser::Parse("X1 * X2 + X3 * X4 + X5 * X6", vars);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) {
            node.Value = static_cast<Operon::Scalar>(0.01);
        } // NOLINT
    }

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;

    Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, ds, tree };

    Operon::Problem problem{ds, range, range};

    auto constexpr batchSize { 32 };

#if defined(HAVE_CERES)
    SUBCASE("ceres autodiff")
    {
        LevenbergMarquardtOptimizer<DerivativeCalculator, OptimizerType::Ceres> optimizer(dc, tree, ds);
        OptimizerSummary summary {};
        auto coeff = optimizer.Optimize(target, range, iterations, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
        tree.SetCoefficients(coeff);
        fmt::print("final tree: {}\n", InfixFormatter::Format(tree, ds));
    }
#endif
    // auto const dim { tree.CoefficientsCount() };

    // std::vector<std::unique_ptr<UpdateRule::LearningRateUpdateRule<Operon::Scalar> const>> rules;
    // rules.emplace_back(new UpdateRule::Constant<Operon::Scalar>(dim, 1e-3)); // NOLINT
    // rules.emplace_back(new UpdateRule::Momentum<Operon::Scalar>(dim));
    // rules.emplace_back(new UpdateRule::RmsProp<Operon::Scalar>(dim));
    // rules.emplace_back(new UpdateRule::AdaDelta<Operon::Scalar>(dim));
    // rules.emplace_back(new UpdateRule::AdaMax<Operon::Scalar>(dim));
    // rules.emplace_back(new UpdateRule::Adam<Operon::Scalar>(dim));
    // rules.emplace_back(new UpdateRule::YamAdam<Operon::Scalar>(dim));
    // rules.emplace_back(new UpdateRule::AmsGrad<Operon::Scalar>(dim));
    // rules.emplace_back(new UpdateRule::Yogi<Operon::Scalar>(dim));

    auto testOptimizer = [&](OptimizerBase<DTable>& optimizer, std::string const& name) {
        fmt::print(fmt::fg(fmt::color::orange), "=== {} Solver ===\n", name);
        auto summary = optimizer.Optimize(rng, tree);
        fmt::print("batch size: {}\n", batchSize);
        fmt::print("expression: {}\n", InfixFormatter::Format(tree, ds));
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
        fmt::print("final parameters: {}\n\n", summary.FinalParameters);
    };

    SUBCASE("tiny")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Tiny> optimizer{dtable, problem};
        testOptimizer(optimizer, "tiny solver");
    }

    SUBCASE("eigen")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> optimizer { dtable, problem };
        testOptimizer(optimizer, "eigen solver");
    }

    SUBCASE("ceres")
    {
        LevenbergMarquardtOptimizer<DTable, OptimizerType::Ceres> optimizer { dtable, problem };
        testOptimizer(optimizer, "ceres solver");
    }

    SUBCASE("lbfgs / gaussian")
    {
        LBFGSOptimizer<DTable, GaussianLikelihood<Operon::Scalar>> optimizer { dtable, problem };
        testOptimizer(optimizer, "l-bfgs / gaussian");
    }

    SUBCASE("lbfgs / poisson")
    {
        LBFGSOptimizer<DTable, PoissonLikelihood<Operon::Scalar>> optimizer { dtable, problem };
        testOptimizer(optimizer, "l-bfgs / poisson");
    }

    SUBCASE("sgd / gaussian")
    {
        // for (auto const& rule : rules) {
        //     SGDOptimizer<DTable, GaussianLikelihood<Operon::Scalar>> optimizer { dtable, problem, *(rule) };
        //     testOptimizer(optimizer, fmt::format("sgd / gaussian / {}", rule->Name()));
        // }
    }

    SUBCASE("sgd / poisson")
    {
        // for (auto const& rule : rules) {
        //     SGDOptimizer<DTable, PoissonLikelihood<Operon::Scalar>> optimizer { dtable, problem, *(rule) };
        //     testOptimizer(optimizer, fmt::format("sgd / poisson / {}", rule->Name()));
        // }
    }
}
} // namespace Operon::Test
