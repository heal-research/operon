// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
//
#include <doctest/doctest.h>
#include "operon/autodiff/autodiff.hpp"
#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {

TEST_CASE("Evaluation correctness")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows() };

    Interpreter interpreter;
    auto const& X = ds.Values(); // NOLINT

    Operon::Map<std::string, Operon::Hash> vars;
    for (auto const& v : ds.Variables()) {
        fmt::print("{} : {} {}\n", v.Name, v.Hash, v.Index);
        vars[v.Name] = v.Hash;
    }

    std::vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    SUBCASE("Basic operations")
    {
        const auto eps = 1e-6;

        auto tree = InfixParser::Parse("X1 + X2 + X3", vars);
        auto estimatedValues = interpreter.operator()<Operon::Scalar>(tree, ds, range);
        auto res1 = X.col(0) + X.col(1) + X.col(2);

        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res1(i)) < eps; }));

        tree = InfixParser::Parse("X1 - X2 + X3", vars);
        estimatedValues = interpreter.operator()<Operon::Scalar>(tree, ds, range);
        auto res2 = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res2(i)) < eps; }));

        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = interpreter.operator()<Operon::Scalar>(tree, ds, range);
        auto res3 = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res3(i)) < eps; }));
    }
}

TEST_CASE("parameter optimization")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows() };

    auto const& X = ds.Values(); // NOLINT

    Operon::Map<std::string, Operon::Hash> vars;
    for (auto v : ds.Variables()) {
        fmt::print("{} : {}\n", v.Name, v.Hash);
        vars[v.Name] = v.Hash;
    }

    auto s1 = ds.GetValues("X1"); 
    auto s2 = ds.GetValues("X2"); 
    auto s3 = ds.GetValues("X3"); 
    auto s4 = ds.GetValues("X4");

    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> x1(s1.data(), std::ssize(s1));
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> x2(s2.data(), std::ssize(s2));
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> x3(s3.data(), std::ssize(s3));
    Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> x4(s4.data(), std::ssize(s4));

    Eigen::Array<Operon::Scalar, -1, 1> res = x1 * x2 + x3 * x4;
    Operon::Span<Operon::Scalar> target(res.data(), res.size());
    auto tree = InfixParser::Parse("X1 * X2 + X3 * X4", vars);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) { node.Value = static_cast<Operon::Scalar>(0.0001); } // NOLINT
    }
    fmt::print("initial tree: {}\n", InfixFormatter::Format(tree, ds));

#if false 
    using Interpreter = Operon::GenericInterpreter<Operon::Scalar, Operon::Dual>;
    Interpreter interpreter;
    using DerivativeCalculator = Operon::Autodiff::Forward::ForwardAutodiffCalculator<Interpreter>;
    DerivativeCalculator dc{ interpreter };
#else
    using Interpreter = Operon::GenericInterpreter<Operon::Scalar>;
    Interpreter interpreter;
    using DerivativeCalculator = Operon::Autodiff::Reverse::DerivativeCalculator<Interpreter>;
    DerivativeCalculator dc{ interpreter };
#endif

    auto constexpr iterations{50};

#if defined(HAVE_CERES)
    SUBCASE("ceres autodiff") {
        NonlinearLeastSquaresOptimizer<DerivativeCalculator, OptimizerType::Ceres> optimizer(dc, tree, ds);
        OptimizerSummary summary{};
        auto coeff = optimizer.Optimize(target, range, iterations, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
        tree.SetCoefficients(coeff);
        fmt::print("final tree: {}\n", InfixFormatter::Format(tree, ds));
    }
#endif

    SUBCASE("tiny") {
        NonlinearLeastSquaresOptimizer<DerivativeCalculator, OptimizerType::Tiny> optimizer(dc, tree, ds);
        OptimizerSummary summary{};
        auto coeff = optimizer.Optimize(target, range, iterations, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
        tree.SetCoefficients(coeff);
        fmt::print("final tree: {}\n", InfixFormatter::Format(tree, ds));
    }

    SUBCASE("eigen") {
        NonlinearLeastSquaresOptimizer<DerivativeCalculator, OptimizerType::Eigen> optimizer(dc, tree, ds);
        OptimizerSummary summary{};
        auto coeff = optimizer.Optimize(target, range, iterations, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
        tree.SetCoefficients(coeff);
        fmt::print("final tree: {}\n", InfixFormatter::Format(tree, ds));
    }

    SUBCASE("ceres") {
        NonlinearLeastSquaresOptimizer<DerivativeCalculator, OptimizerType::Ceres> optimizer(dc, tree, ds);
        OptimizerSummary summary{};
        auto coeff = optimizer.Optimize(target, range, iterations, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
        tree.SetCoefficients(coeff);
        fmt::print("final tree: {}\n", InfixFormatter::Format(tree, ds));
    }
}
} // namespace Operon::Test

