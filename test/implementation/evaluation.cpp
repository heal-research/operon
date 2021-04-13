/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#include "core/dataset.hpp"
#include "interpreter/interpreter.hpp"
#include "nnls/nnls.hpp"
#include "core/format.hpp"
#include "core/stats.hpp"
#include "core/metrics.hpp"
#include "parser/infix.hpp"

#include <doctest/doctest.h>

#include "nnls/tiny_optimizer.hpp"
#include <ceres/tiny_solver.h>

namespace Operon {
namespace Test {
TEST_CASE("Evaluation correctness")
{
    auto ds = Dataset("../data/Poly-10.csv", true);
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
    auto const& X = ds.Values();

    robin_hood::unordered_map<std::string, Operon::Hash> map;
    for (auto v : ds.Variables()) {
        fmt::print("{} : {}\n", v.Name, v.Hash);
        map[v.Name] = v.Hash;
    }
    
    std::vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    SUBCASE("Basic operations")
    {
        auto eps = 1e-6;

        auto tree = InfixParser::Parse("X1 + X2", map);
        auto estimatedValues = interpreter.Evaluate<Operon::Scalar>(tree, ds, range);
        auto res1 = X.col(0) + X.col(1);
        
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res1(i)) < eps; }));

        tree = InfixParser::Parse("X1 - X2", map);
        estimatedValues = interpreter.Evaluate<Operon::Scalar>(tree, ds, range);
        auto res2 = X.col(0) - X.col(1);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - res2(i)) < eps; }));
    }
}

TEST_CASE("Numeric optimization")
{
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto range = Range { 0, ds.Rows() };

    Interpreter interpreter;
    auto const& X = ds.Values();


    robin_hood::unordered_map<std::string, Operon::Hash> map;
    for (auto v : ds.Variables()) {
        fmt::print("{} : {}\n", v.Name, v.Hash);
        map[v.Name] = v.Hash;
    }

    Eigen::Array<Operon::Scalar, -1, 1> res = X.col(0) + X.col(1);
    gsl::span<Operon::Scalar> target(res.data(), res.size());
    auto tree = InfixParser::Parse("X1 + X2", map);
    for (auto& node : tree.Nodes()) {
        if (node.IsVariable()) node.Value = static_cast<Operon::Scalar>(0.0001);
    }

    SUBCASE("ceres autodiff") {
        auto tree_copy = tree;
        NonlinearLeastSquaresOptimizer<OptimizerType::CERES> optimizer(interpreter, tree_copy, ds);
        auto summary = optimizer.Optimize(target, range, 10, true, true);
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
        auto summary = optimizer.Optimize(target, range, 10, true, true);
        fmt::print("iterations: {}, initial cost: {}, final cost: {}\n", summary.Iterations, summary.InitialCost, summary.FinalCost);
    }
}

} // namespace Test
} // namespace Operon

