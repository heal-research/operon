// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <algorithm>
#include <cstddef>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/types.hpp"

namespace Operon::Test {

TEST_CASE("Node type traits", "[core]")
{
    SECTION("Node is trivial") {
        CHECK(std::is_trivial_v<Operon::Node>);
    }

    SECTION("Node is trivially copyable") {
        CHECK(std::is_trivially_copyable_v<Operon::Node>);
    }

    SECTION("Node is standard layout") {
        CHECK(std::is_standard_layout_v<Operon::Node>);
    }

    SECTION("Node size is at most 64 bytes") {
        CHECK(sizeof(Node) <= size_t{64});
    }

    SECTION("Ref node is not optimizable") {
        Node ref(NodeType::Ref);
        CHECK(ref.IsLeaf());
        CHECK(ref.IsRef());
        CHECK_FALSE(ref.Optimize);
    }
}

TEST_CASE("Tree construction and access", "[core]")
{
    Operon::Vector<Node> nodes;
    std::generate_n(std::back_inserter(nodes), 50, []() { return Node(NodeType::Add); }); // NOLINT
    Tree tree{nodes};

    SECTION("Tree stores correct number of nodes") {
        CHECK(tree.Length() == 50);
    }

    SECTION("Individual holds a tree") {
        Individual ind(1);
        ind.Genotype = std::move(tree);
        CHECK(ind.Genotype.Length() == 50);
    }
}

TEST_CASE("Tree coefficients", "[core]")
{
    Node c1(NodeType::Constant); c1.Value = 3.14F; c1.Optimize = true;
    Node c2(NodeType::Constant); c2.Value = 2.71F; c2.Optimize = true;
    Node const add(NodeType::Add);
    Tree tree({c1, c2, add});
    tree.UpdateNodes();

    auto coeff = tree.GetCoefficients();
    REQUIRE(coeff.size() == 2);
    CHECK(coeff[0] == Catch::Approx(3.14F));
    CHECK(coeff[1] == Catch::Approx(2.71F));

    coeff[0] = 1.0F;
    coeff[1] = 2.0F;
    tree.SetCoefficients(coeff);
    auto coeff2 = tree.GetCoefficients();
    CHECK(coeff2[0] == Catch::Approx(1.0F));
    CHECK(coeff2[1] == Catch::Approx(2.0F));
}

TEST_CASE("Dataset loading and access", "[core]")
{
    auto ds = Dataset("./data/Poly-10.csv", /*hasHeader=*/true);

    SECTION("Row and column counts") {
        CHECK(ds.Rows<std::size_t>() > 0);
        CHECK(ds.Cols<std::size_t>() > 0);
    }

    SECTION("Variable listing") {
        auto variables = ds.GetVariables();
        CHECK(!variables.empty());
    }

    SECTION("Column access by variable hash") {
        auto variables = ds.GetVariables();
        REQUIRE(!variables.empty());
        auto values = ds.GetValues(variables[0].Hash);
        CHECK(values.size() == ds.Rows<std::size_t>());
    }

    SECTION("Target variable lookup") {
        auto result = ds.GetVariable("Y");
        CHECK(result.has_value());
    }
}

TEST_CASE("PrimitiveSet configuration", "[core]")
{
    PrimitiveSet pset;
    pset.SetConfig(PrimitiveSet::Arithmetic);

    auto enabled = pset.EnabledPrimitives();
    CHECK(!enabled.empty());

    // Arithmetic should include Add, Sub, Mul, Div, Constant, Variable
    CHECK(pset.IsEnabled(Node(NodeType::Add).HashValue));
    CHECK(pset.IsEnabled(Node(NodeType::Sub).HashValue));
    CHECK(pset.IsEnabled(Node(NodeType::Mul).HashValue));
    CHECK(pset.IsEnabled(Node(NodeType::Div).HashValue));
}

TEST_CASE("Tree::Simplify", "[core][simplify]")
{
    using NT = Operon::NodeType;
    using S  = Operon::Scalar;

    auto Const = [](S v) { return Node::Constant(v); };
    auto Var   = []()    { return Node(NT::Variable); };

    SECTION("constant folding: Add(2, 3) -> Const(5)") {
        Operon::Vector<Node> ns{ Const(2), Const(3), Node(NT::Add) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        REQUIRE(tree[0].IsConstant());
        CHECK(tree[0].Value == Catch::Approx(5.0));
    }

    SECTION("constant folding: Mul(2, 3) -> Const(6)") {
        Operon::Vector<Node> ns{ Const(2), Const(3), Node(NT::Mul) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        REQUIRE(tree[0].IsConstant());
        CHECK(tree[0].Value == Catch::Approx(6.0));
    }

    SECTION("constant folding: nested Add(Mul(2,3), 4) -> Const(10)") {
        // [Const(2), Const(3), Mul, Const(4), Add]
        Operon::Vector<Node> ns{ Const(2), Const(3), Node(NT::Mul), Const(4), Node(NT::Add) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        REQUIRE(tree[0].IsConstant());
        CHECK(tree[0].Value == Catch::Approx(10.0));
    }

    SECTION("identity: x + 0 -> x") {
        Operon::Vector<Node> ns{ Const(0), Var(), Node(NT::Add) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
    }

    SECTION("identity: x * 1 -> x") {
        Operon::Vector<Node> ns{ Const(1), Var(), Node(NT::Mul) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
    }

    SECTION("annihilator: x * 0 -> 0") {
        Operon::Vector<Node> ns{ Const(0), Var(), Node(NT::Mul) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        REQUIRE(tree[0].IsConstant());
        CHECK(tree[0].Value == Catch::Approx(0.0));
    }

    SECTION("identity: x - 0 -> x") {
        // post-order Sub(x, 0) = [Const(0), Var, Sub]: Var is i-1 (minuend), Const(0) is k (subtrahend)
        Operon::Vector<Node> ns{ Const(0), Var(), Node(NT::Sub) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
    }

    SECTION("identity: x / 1 -> x") {
        // post-order Div(x, 1) = [Const(1), Var, Div]: Var is i-1 (numerator), Const(1) is k (denominator)
        Operon::Vector<Node> ns{ Const(1), Var(), Node(NT::Div) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
    }

    SECTION("Pow: x^0 -> 1") {
        // [Const(0), Var, Pow]: base=Var (j=i-1), exp=Const(0) (k)
        Operon::Vector<Node> ns{ Const(0), Var(), Node(NT::Pow) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        REQUIRE(tree[0].IsConstant());
        CHECK(tree[0].Value == Catch::Approx(1.0));
    }

    SECTION("Pow: x^1 -> x") {
        Operon::Vector<Node> ns{ Const(1), Var(), Node(NT::Pow) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
    }

    SECTION("Pow: 1^x -> 1") {
        // [Var, Const(1), Pow]: base=Const(1) (j=i-1), exp=Var (k)
        Operon::Vector<Node> ns{ Var(), Const(1), Node(NT::Pow) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        REQUIRE(tree[0].IsConstant());
        CHECK(tree[0].Value == Catch::Approx(1.0));
    }

    SECTION("n-ary: Add(x, 0, 0) -> x (two zero children removed)") {
        // Manually set arity=3 for a 3-child Add
        Node addNode(NT::Add);
        addNode.Arity = 3;
        Operon::Vector<Node> ns{ Const(0), Const(0), Var(), addNode };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
    }

    SECTION("constant folding: Exp(Const(0)) -> Const(1)") {
        Operon::Vector<Node> ns{ Const(0), Node(NT::Exp) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        REQUIRE(tree[0].IsConstant());
        CHECK(tree[0].Value == Catch::Approx(1.0));
    }

    SECTION("strength reduction: Pow(x, 2) -> Square(x)") {
        // [Const(2), Var, Pow]: base=Var (ch[0]=i-1), exp=Const(2) (ch[1])
        Operon::Vector<Node> ns{ Const(2), Var(), Node(NT::Pow) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 2);
        CHECK(tree[1].Type == NT::Square);
        CHECK(tree[0].IsVariable());
    }

    SECTION("strength reduction: Pow(x, 0.5) -> Sqrt(x)") {
        Operon::Vector<Node> ns{ Const(0.5), Var(), Node(NT::Pow) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 2);
        CHECK(tree[1].Type == NT::Sqrt);
        CHECK(tree[0].IsVariable());
    }

    SECTION("structural inverse: Log(Exp(x)) -> x") {
        // [Var, Exp, Log] in post-order
        Operon::Vector<Node> ns{ Var(), Node(NT::Exp), Node(NT::Log) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
    }

    SECTION("structural inverse: Logabs(Exp(x)) -> x") {
        Operon::Vector<Node> ns{ Var(), Node(NT::Exp), Node(NT::Logabs) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 1);
        CHECK(tree[0].IsVariable());
    }

    SECTION("structural inverse: Sqrt(Square(x)) -> Abs(x)") {
        // [Var, Square, Sqrt] in post-order
        Operon::Vector<Node> ns{ Var(), Node(NT::Square), Node(NT::Sqrt) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 2);
        CHECK(tree[1].Type == NT::Abs);
        CHECK(tree[0].IsVariable());
    }

    SECTION("structural inverse: Sqrtabs(Square(x)) -> Abs(x)") {
        Operon::Vector<Node> ns{ Var(), Node(NT::Square), Node(NT::Sqrtabs) };
        auto tree = Tree(std::move(ns)).UpdateNodes().Simplify();
        REQUIRE(tree.Length() == 2);
        CHECK(tree[1].Type == NT::Abs);
        CHECK(tree[0].IsVariable());
    }
}

} // namespace Operon::Test
