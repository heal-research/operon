// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

// Exercises MakeComposedCallable/MakeComposedCallableDiff (composed-function
// derivation, step 2 of operon-planning/designs/composed-functions.md)
// directly against DispatchTable::RegisterFunction — the higher-level
// RegisterComposedFunction orchestration (arity/diff-coverage validation,
// wiring into the other 3 backends) is a later step and not exercised here.

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <cmath>

#include "../operon_test.hpp"
#include "operon/core/composed_function.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/parser/infix.hpp"

namespace Operon::Test {

namespace {
    auto MakeComposedNode(std::string const& name, uint16_t arity) -> Operon::Node
    {
        auto const hash = Operon::Hasher{}(name);
        return Operon::Node::Function(hash, arity);
    }
} // namespace

TEST_CASE("Composed function: Callable derivation", "[composed-function]")
{
    Operon::Dataset ds(std::vector<std::string>{"x", "y"},
                       std::vector<std::vector<Operon::Scalar>>{{1.3F, -0.6F, 2.0F}, {0.4F, 1.1F, -0.9F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};
    auto const xHash = ds.GetVariable("x")->Hash;
    auto const yHash = ds.GetVariable("y")->Hash;

    SECTION("Unary body: numeric eval matches the manually-expanded expression") {
        auto body = InfixParser::ParseFunctionBody("1 / (1 + exp(-x))", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("logistic", 1);
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto result = interpreter.Evaluate(coeff, range);
        auto const xs = ds.GetValues("x");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto const expected = 1.0 / (1.0 + std::exp(-static_cast<double>(xs[i])));
            CHECK(static_cast<double>(result[i]) == Catch::Approx(expected).margin(1e-5));
        }
    }

    SECTION("Binary body: argument order matches Tree::Indices binding (order-sensitive op)") {
        // Sub is order-sensitive — this would fail silently (wrong sign/value)
        // if BindArgIndices bound params in the wrong order.
        auto body = InfixParser::ParseFunctionBody("a - b", std::vector<std::string>{"a", "b"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 2);
        auto composedNode = MakeComposedNode("sub2", 2);
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Node vy(Operon::NodeType::Variable);
        vy.HashValue = vy.CalculatedHashValue = yHash;
        vy.Value = 1.0F;
        // Postfix: first-arg subtree, second-arg subtree, then the call —
        // i.e. sub2(x, y).
        Operon::Vector<Operon::Node> nodes{vx, vy, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto result = interpreter.Evaluate(coeff, range);
        auto const xs = ds.GetValues("x");
        auto const ys = ds.GetValues("y");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            CHECK(static_cast<double>(result[i]) == Catch::Approx(static_cast<double>(xs[i]) - static_cast<double>(ys[i])).margin(1e-5));
        }
    }

    SECTION("Composed node's own weight is applied once (not baked into the body)") {
        auto body = InfixParser::ParseFunctionBody("sin(x)", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("sinComposed", 1);
        composedNode.Value = 3.7F;
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto result = interpreter.Evaluate(coeff, range);
        auto const xs = ds.GetValues("x");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto const expected = 3.7 * std::sin(static_cast<double>(xs[i]));
            CHECK(static_cast<double>(result[i]) == Catch::Approx(expected).margin(1e-5));
        }
    }
}

TEST_CASE("Composed function: CallableDiff derivation", "[composed-function]")
{
    Operon::Dataset ds(std::vector<std::string>{"x", "y"},
                       std::vector<std::vector<Operon::Scalar>>{{1.3F, -0.6F, 2.0F}, {0.4F, 1.1F, -0.9F}});
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Range const range{0, ds.Rows<std::size_t>()};
    auto const xHash = ds.GetVariable("x")->Hash;

    SECTION("Weighted composed node: d/dx(w*sin(x)) = w*cos(x) — the exact case the double-weighting bug would break") {
        auto body = InfixParser::ParseFunctionBody("sin(x)", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto diff = MakeComposedCallableDiff<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("sinComposedW", 1);
        composedNode.Value = 3.7F;
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable, diff);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto rev = interpreter.JacRevVariable(coeff, range, xHash);
        auto fwd = interpreter.JacFwdVariable(coeff, range, xHash);

        auto const xs = ds.GetValues("x");
        for (std::size_t i = 0; i < range.Size(); ++i) {
            auto const expected = 3.7 * std::cos(static_cast<double>(xs[i]));
            CHECK(static_cast<double>(rev[i]) == Catch::Approx(expected).margin(1e-4));
            CHECK(static_cast<double>(fwd[i]) == Catch::Approx(expected).margin(1e-4));
        }
    }

    SECTION("Repeated parameter occurrence, nonzero derivative: d/dx(x + x) = 2") {
        auto body = InfixParser::ParseFunctionBody("x + x", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto diff = MakeComposedCallableDiff<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("doubleX", 1);
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable, diff);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto rev = interpreter.JacRevVariable(coeff, range, xHash);
        auto fwd = interpreter.JacFwdVariable(coeff, range, xHash);

        for (std::size_t i = 0; i < range.Size(); ++i) {
            CHECK(static_cast<double>(rev[i]) == Catch::Approx(2.0).margin(1e-4));
            CHECK(static_cast<double>(fwd[i]) == Catch::Approx(2.0).margin(1e-4));
        }
    }

    SECTION("Repeated parameter occurrence, cancelling derivative: d/dx(x - x) = 0") {
        auto body = InfixParser::ParseFunctionBody("x - x", std::vector<std::string>{"x"});
        auto callable = MakeComposedCallable<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto diff = MakeComposedCallableDiff<Operon::DispatchTable<Operon::Scalar>, Operon::Scalar>(dtable, body, 1);
        auto composedNode = MakeComposedNode("zeroX", 1);
        dtable.RegisterFunction<Operon::Scalar>(composedNode.HashValue, callable, diff);

        Operon::Node vx(Operon::NodeType::Variable);
        vx.HashValue = vx.CalculatedHashValue = xHash;
        vx.Value = 1.0F;
        Operon::Vector<Operon::Node> nodes{vx, composedNode};
        Operon::Tree tree{nodes};

        auto coeff = tree.GetCoefficients();
        Operon::Interpreter<Operon::Scalar, Operon::DispatchTable<Operon::Scalar>> const interpreter{&dtable, &ds, &tree};
        auto rev = interpreter.JacRevVariable(coeff, range, xHash);
        auto fwd = interpreter.JacFwdVariable(coeff, range, xHash);

        for (std::size_t i = 0; i < range.Size(); ++i) {
            CHECK(static_cast<double>(rev[i]) == Catch::Approx(0.0).margin(1e-4));
            CHECK(static_cast<double>(fwd[i]) == Catch::Approx(0.0).margin(1e-4));
        }
    }
}

} // namespace Operon::Test
