// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include "../operon_test.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"
#include "operon/interpreter/functions.hpp"

namespace Operon::Test {

TEST_CASE("Evaluation correctness", "[interpreter]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, ds.Rows<std::size_t>()};

    using DTable = DispatchTable<Operon::Scalar>;
    auto const& X = ds.Values(); // NOLINT

    Operon::Vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    DTable dtable;
    auto const eps = 1e-3;

    SECTION("X1 + X2 + X3") {
        auto tree = InfixParser::Parse("X1 + X2 + X3", ds);
        auto coeff = tree.GetCoefficients();
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(coeff, range);
        Eigen::Array<Operon::Scalar, -1, 1> expected = X.col(0) + X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - expected(i)) < eps; }));
    }

    SECTION("X1 - X2 + X3") {
        auto tree = InfixParser::Parse("X1 - X2 + X3", ds);
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range);
        auto expected = X.col(0) - X.col(1) + X.col(2);
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - expected(i)) < eps; }));
    }

    SECTION("log(abs(X1))") {
        auto tree = InfixParser::Parse("log(abs(X1))", ds);
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range);
        Eigen::Array<Operon::Scalar, -1, 1> expected = X.col(0).abs().log();
        CHECK(std::all_of(indices.begin(), indices.end(), [&](auto i) { return std::abs(estimatedValues[i] - expected(i)) < eps; }));
    }

    SECTION("log of constant") {
        auto tree = InfixParser::Parse("log(0.12485691905021667)", ds);
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range);
        CHECK(std::abs(estimatedValues[0] - std::log(0.12485691905021667)) < eps);
    }

    SECTION("N-ary fmax") {
        Operon::Node node(Operon::NodeType::Fmax);
        node.Arity = 3;
        auto a = Operon::Node::Constant(2);
        auto b = Operon::Node::Constant(3);
        auto c = Operon::Node::Constant(4);
        auto tree = Operon::Tree({a, b, c, node});
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range);
        CHECK(estimatedValues[0] == 4);
    }

    SECTION("Unary sub (negation)") {
        auto node = Operon::Node(Operon::NodeType::Sub);
        node.Arity = 1;
        auto tree = Operon::Tree({Operon::Node::Constant(2), node});
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range);
        CHECK(estimatedValues[0] == -2);
    }
}

TEST_CASE("Batch evaluation", "[interpreter]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range{0, ds.Rows<std::size_t>()};

    Operon::Problem problem{&ds};
    problem.SetTrainingRange(range);
    problem.SetTestRange(range);

    Operon::PrimitiveSet pset{PrimitiveSet::Arithmetic};
    constexpr size_t maxLength = 20;
    Operon::BalancedTreeCreator creator{&pset, ds.VariableHashes(), /* bias= */ 0.0, maxLength};

    Operon::RandomGenerator rng{0};
    auto constexpr n{10};

    Operon::Vector<Operon::Tree> trees;
    Operon::Vector<Operon::Scalar> result(range.Size() * n);
    for (auto i = 0; i < n; ++i) {
        trees.push_back(creator(rng, 20, 10, 20));
    }

    // Should not throw
    REQUIRE_NOTHROW(Operon::EvaluateTrees(trees, &ds, range, {result.data(), result.size()}));
    REQUIRE_NOTHROW(Operon::EvaluateTrees(trees, &ds, range));
}

} // namespace Operon::Test
