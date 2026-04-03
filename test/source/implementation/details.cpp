// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

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

} // namespace Operon::Test
