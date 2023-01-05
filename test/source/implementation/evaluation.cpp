// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
//
#include <doctest/doctest.h>
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

    //auto decimals = [](auto v) {
    //    auto s = fmt::format("{:.50f}", v);
    //    size_t d = 0;
    //    auto p = s.find('.');
    //    while(s[++p] == '0')
    //        ++d;
    //    return d;
    //};

    Interpreter interpreter;
    auto const& X = ds.Values(); // NOLINT

    Operon::Map<std::string, Operon::Hash> vars;
    for (auto v : ds.Variables()) {
        fmt::print("{} : {} {}\n", v.Name, v.Hash, v.Index);
        vars[v.Name] = v.Hash;
    }

    std::vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    SUBCASE("Basic operations")
    {
        const auto eps = 1e-6;

        auto tree = InfixParser::Parse("X1 + X2 + X3", vars);
        auto estimatedValues = interpreter.Evaluate<Operon::Scalar>(tree, ds, range);
        auto res1 = X.col(0) + X.col(1) + X.col(2);

        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res1(i)) < eps; }));

        tree = InfixParser::Parse("X1 - X2 + X3", vars);
        estimatedValues = interpreter.Evaluate<Operon::Scalar>(tree, ds, range);
        auto res2 = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res2(i)) < eps; }));

        fmt::print("tree: {}\n", InfixFormatter::Format(tree, ds));
        estimatedValues = interpreter.Evaluate<Operon::Scalar>(tree, ds, range);
        auto res3 = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res3(i)) < eps; }));
    }
}

TEST_CASE("Numeric optimization")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows() };

    Interpreter interpreter;
    auto const& X = ds.Values(); // NOLINT

    Operon::Map<std::string, Operon::Hash> vars;
    for (auto v : ds.Variables()) {
        fmt::print("{} : {}\n", v.Name, v.Hash);
        vars[v.Name] = v.Hash;
    }

    Eigen::Array<Operon::Scalar, -1, 1> res = X.col(0) + X.col(1);
    Operon::Span<Operon::Scalar> target(res.data(), res.size());
    auto tree = InfixParser::Parse("X1 + X2", vars);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) node.Value = static_cast<Operon::Scalar>(0.0001);
    }

    SUBCASE("ceres autodiff") {
        auto treeCopy = tree;
        NonlinearLeastSquaresOptimizer<OptimizerType::CERES> optimizer(interpreter, treeCopy, ds);
        OptimizerSummary summary;
        auto coeff = optimizer.Optimize(target, range, 10, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
    }

    //SUBCASE("ceres numeric diff") {
    //    auto tree_copy = tree;
    //    NonlinearLeastSquaresOptimizer<OptimizerType::CERES> optimizer(interpreter, tree_copy, ds);
    //    auto summary = optimizer.Optimize<DerivativeMethod::NUMERIC>(target, range, 10, true, true);
    //    fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
    //}

    SUBCASE("tiny") {
        auto tree_copy = tree;
        NonlinearLeastSquaresOptimizer<OptimizerType::TINY> optimizer(interpreter, tree_copy, ds);
        OptimizerSummary summary;
        auto coeff = optimizer.Optimize(target, range, 10, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
    }

    SUBCASE("eigen") {
        auto tree_copy = tree;
        NonlinearLeastSquaresOptimizer<OptimizerType::EIGEN> optimizer(interpreter, tree_copy, ds);
        OptimizerSummary summary;
        auto coeff = optimizer.Optimize(target, range, 10, summary);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
    }
}

TEST_CASE("tiny bug")
{
    auto ds = Dataset("../data/Pagie-1.csv", true);
    auto infix = "((((10.31296 / 4.01705) + ((-27.05388) - 23.68143)) / ((-148.00854) - ((78.81192 * Y) + ((-30.19245) * X)))) / (((((-6.40791) * Y) - (4.72377 * Y)) - (((-76.46925) * X) + 403.50482)) / (14.26075 - (-14.37711))))";
    Operon::Map<std::string, Operon::Hash> vars;
    for (auto const& v : ds.Variables()) {
        vars.insert({ v.Name, v.Hash });
    }

    auto tree = InfixParser::Parse(infix, vars);

    Interpreter interpreter;

    auto range = Range { 0, ds.Rows() };
    auto target = ds.GetValues("F");

    NonlinearLeastSquaresOptimizer<OptimizerType::TINY> optimizer(interpreter, tree, ds);
    OptimizerSummary summary;
    auto coeff = optimizer.Optimize(target, range, 10, summary);
    fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
}


} // namespace Operon::Test

