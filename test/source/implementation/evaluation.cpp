// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <fmt/format.h>
#include <functional>
#include <numeric>

#include "../operon_test.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/types.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/functions.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"

namespace Operon::Test {

TEST_CASE("Evaluation correctness", "[interpreter]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };

    using DTable = DispatchTable<Operon::Scalar>;
    auto const x0 = ds.GetValues(int64_t { 0 });
    auto const x1 = ds.GetValues(int64_t { 1 });
    auto const x2 = ds.GetValues(int64_t { 2 });

    Operon::Vector<size_t> indices(range.Size());
    std::iota(indices.begin(), indices.end(), 0);

    DTable dtable;
    auto const eps = 1e-3;

    SECTION("X1 + X2 + X3")
    {
        auto tree = InfixParser::ParseOrThrow("X1 + X2 + X3", ds);
        auto coeff = tree.GetCoefficients();
        auto estimatedValues = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(coeff, range).value();
        CHECK(std::all_of(indices.begin(), indices.end(),
            [&](auto i) -> auto { return std::abs(estimatedValues[i] - (x0[i] + x1[i] + x2[i])) < eps; }));
    }

    SECTION("X1 - X2 + X3")
    {
        auto tree = InfixParser::ParseOrThrow("X1 - X2 + X3", ds);
        auto estimatedValues
            = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range).value();
        CHECK(std::all_of(indices.begin(), indices.end(),
            [&](auto i) -> auto { return std::abs(estimatedValues[i] - (x0[i] - x1[i] + x2[i])) < eps; }));
    }

    SECTION("log(abs(X1))")
    {
        auto tree = InfixParser::ParseOrThrow("log(abs(X1))", ds);
        auto estimatedValues
            = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range).value();
        CHECK(std::all_of(indices.begin(), indices.end(),
            [&](auto i) -> auto { return std::abs(estimatedValues[i] - std::log(std::abs(x0[i]))) < eps; }));
    }

    SECTION("log of constant")
    {
        auto tree = InfixParser::ParseOrThrow("log(0.12485691905021667)", ds);
        auto estimatedValues
            = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range).value();
        CHECK(std::abs(estimatedValues[0] - std::log(0.12485691905021667)) < eps);
    }

    SECTION("N-ary fmax")
    {
        auto node = Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Fmax), 3);
        auto a = Operon::Node::Constant(2);
        auto b = Operon::Node::Constant(3);
        auto c = Operon::Node::Constant(4);
        auto tree = Operon::Tree({ a, b, c, node });
        auto estimatedValues
            = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range).value();
        CHECK(estimatedValues[0] == 4);
    }

    SECTION("Unary sub (negation)")
    {
        auto node = Operon::Node::Function(static_cast<Operon::Hash>(Operon::BuiltinOp::Sub), 1);
        auto tree = Operon::Tree({ Operon::Node::Constant(2), node });
        auto estimatedValues
            = Interpreter<Operon::Scalar, DTable>(&dtable, &ds, &tree).Evaluate(tree.GetCoefficients(), range).value();
        CHECK(estimatedValues[0] == -2);
    }
}

TEST_CASE("Batch evaluation", "[interpreter]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };

    Operon::Problem problem { &ds };
    problem.SetTrainingRange(range);
    problem.SetTestRange(range);

    Operon::PrimitiveSet pset { PrimitiveSet::Arithmetic };
    constexpr size_t maxLength = 20;
    Operon::BalancedTreeCreator const creator { &pset, ds.VariableHashes(), /* bias= */ 0.0, maxLength };

    Operon::RandomGenerator rng { 0 };
    auto constexpr n { 10 };

    Operon::Vector<Operon::Tree> trees;
    Operon::Vector<Operon::Scalar> result(range.Size() * n);
    for (auto i = 0; i < n; ++i) {
        trees.push_back(creator(rng, 20, 10, 20));
    }

    // Should succeed
    REQUIRE(Operon::EvaluateTrees(trees, &ds, range, { result.data(), result.size() }));
    REQUIRE(Operon::EvaluateTrees(trees, &ds, range));
}

TEST_CASE("Batch evaluation reports typed failures", "[interpreter]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };

    constexpr auto missingPrimitive = Operon::Hash { 0xBADF00D };
    auto tree = Operon::Tree({
        Operon::Node::Constant(1),
        Operon::Node::Constant(2),
        Operon::Node::Function(missingPrimitive, 2),
    });
    Operon::Vector<Operon::Tree> trees { tree };
    Operon::Vector<Operon::Scalar> result(range.Size());

    auto intoBuffer = Operon::EvaluateTrees(trees, &ds, range, { result.data(), result.size() });
    REQUIRE_FALSE(intoBuffer.has_value());
    CHECK(intoBuffer.error().Index == 0);
    CHECK(intoBuffer.error().Error.Kind == InterpreterError::Code::MissingPrimitive);
    CHECK(intoBuffer.error().Error.Hash == missingPrimitive);

    auto allocated = Operon::EvaluateTrees(trees, &ds, range);
    REQUIRE_FALSE(allocated.has_value());
    CHECK(allocated.error().Index == 0);
    CHECK(allocated.error().Error.Kind == InterpreterError::Code::MissingPrimitive);
    CHECK(allocated.error().Error.Hash == missingPrimitive);
}

TEST_CASE("Batch evaluation validates output size", "[interpreter]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);
    auto range = Range { 0, ds.Rows<std::size_t>() };
    auto trees = Operon::Vector<Operon::Tree> { Operon::Tree { Operon::Node::Constant(1) } };
    Operon::Vector<Operon::Scalar> result(range.Size() - 1);

    auto evaluated = Operon::EvaluateTrees(trees, &ds, range, { result.data(), result.size() });
    REQUIRE_FALSE(evaluated);
    CHECK(evaluated.error().Error.Kind == InterpreterError::Code::InvalidOutputSize);
    CHECK(evaluated.error().Error.ExpectedSize == range.Size());
    CHECK(evaluated.error().Error.ActualSize == result.size());
}

// An n-ary node's weight multiplies the node's result exactly once, regardless of how many argument chunks the
// dispatcher needs (arity > 4 is evaluated in chunks of up to four arguments).
TEST_CASE("N-ary function weight is applied once", "[interpreter]")
{
    auto const maxArity = 9;
    std::vector<std::string> names;
    std::vector<std::vector<Operon::Scalar>> columns;
    for (auto i = 0; i < maxArity; ++i) {
        names.push_back(fmt::format("v{}", i));
        // keep every value away from zero so Div is well-conditioned; scatter them so the extrema of Fmin/Fmax land in
        // different argument chunks
        columns.push_back({ 1.5F + static_cast<Scalar>((i * 5) % 9), 2.0F + (0.25F * static_cast<Scalar>((i * 4) % 7)),
            1.25F, 3.0F });
    }
    Dataset ds(names, columns);
    Range const range { 0, ds.Rows<std::size_t>() };
    DispatchTable<Operon::Scalar> dtable;

    auto const expectedFor = [&](BuiltinOp op, int arity, double weight, std::size_t row) {
        // postfix layout: the argument adjacent to the function node is the first operand
        std::vector<double> x;
        for (auto i = arity - 1; i >= 0; --i) {
            x.push_back(static_cast<double>(columns[i][row]));
        }
        auto const rest = [&](auto init, auto f) {
            auto acc = init;
            for (auto i = 1UL; i < x.size(); ++i) {
                acc = f(acc, x[i]);
            }
            return acc;
        };
        switch (op) {
        case BuiltinOp::Add:
            return weight * std::accumulate(x.begin(), x.end(), 0.0);
        case BuiltinOp::Mul:
            return weight * std::accumulate(x.begin(), x.end(), 1.0, std::multiplies<> {});
        case BuiltinOp::Sub:
            return arity == 1 ? -weight * x[0] : weight * (x[0] - rest(0.0, std::plus<> {}));
        case BuiltinOp::Div:
            return arity == 1 ? weight / x[0] : weight * x[0] / rest(1.0, std::multiplies<> {});
        case BuiltinOp::Fmin:
            return weight * *std::ranges::min_element(x);
        case BuiltinOp::Fmax:
            return weight * *std::ranges::max_element(x);
        default:
            FAIL("unexpected op");
            return 0.0;
        }
    };

    for (auto const op :
        { BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Sub, BuiltinOp::Div, BuiltinOp::Fmin, BuiltinOp::Fmax }) {
        for (auto const arity : { 1, 2, 3, 4, 5, 6, 7, 8, 9 }) {
            for (auto const weight : { 1.0, 2.0, -1.5, 0.0 }) {
                DYNAMIC_SECTION("op " << static_cast<int>(op) << " arity " << arity << " weight " << weight)
                {
                    Operon::Vector<Node> nodes;
                    for (auto i = 0; i < arity; ++i) {
                        Node variable(NodeType::Variable);
                        variable.HashValue = variable.CalculatedHashValue = ds.GetVariable(names[i])->Hash;
                        variable.Value = 1.0F;
                        variable.Optimize = false;
                        nodes.push_back(variable);
                    }
                    auto fn = Node::Function(static_cast<Hash>(op), static_cast<uint16_t>(arity));
                    fn.Value = static_cast<Scalar>(weight);
                    nodes.push_back(fn);
                    Tree tree { nodes };
                    tree.UpdateNodes();

                    auto const result
                        = Interpreter<Scalar, DispatchTable<Scalar>>(&dtable, &ds, &tree).Evaluate({}, range);
                    REQUIRE(result);
                    for (auto row = 0UL; row < range.Size(); ++row) {
                        auto const expected = expectedFor(op, arity, weight, row);
                        CHECK((*result)[row] == Catch::Approx(expected).epsilon(1e-4).margin(1e-5));
                    }
                }
            }
        }
    }
}

} // namespace Operon::Test
