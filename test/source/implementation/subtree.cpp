// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026 Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <ranges>
#include <vector>

#include "operon/core/tree.hpp"

namespace Operon::Test {

namespace {
    auto Constant(Scalar value) -> Node
    {
        return Node::Constant(value);
    }

    auto Add() -> Node
    {
        return Node::Function(static_cast<Hash>(BuiltinOp::Add), 2);
    }
} // namespace

TEST_CASE("Subtree child ranges handle leaves and preserve postfix child order", "[core]")
{
    auto rootLeaf = Tree({Constant(1)}).UpdateNodes();
    CHECK(std::ranges::empty(rootLeaf.Children(0)));
    CHECK(std::ranges::empty(rootLeaf.Indices(0)));

    auto tree = Tree({Constant(1), Constant(2), Add(), Constant(3), Add()}).UpdateNodes();
    std::vector<std::size_t> indices;
    for (auto const index : tree.Indices(4)) {
        indices.push_back(index);
    }
    CHECK(std::ranges::empty(tree.Indices(0)));

    CHECK(indices == std::vector<std::size_t>{3, 2});

    std::vector<Scalar> children;
    for (auto const& child : tree.Children(4)) {
        children.push_back(child.Value);
    }
    CHECK(children == std::vector<Scalar>{3, 1});
}

TEST_CASE("Subtree child ranges enumerate indices and mutable nodes", "[core]")
{
    auto tree = Tree({Constant(1), Constant(2), Add()}).UpdateNodes();

    std::vector<std::tuple<std::size_t, std::size_t>> indexed;
    for (auto const [position, child] : Tree::EnumerateIndices(tree.Nodes(), 2)) {
        indexed.emplace_back(position, child);
    }
    CHECK(indexed == std::vector<std::tuple<std::size_t, std::size_t>>{{0, 1}, {1, 0}});

    for (auto [position, child] : Tree::EnumerateNodes(tree.Nodes(), 2)) {
        child.Value = static_cast<Scalar>(position + 10);
    }
    CHECK(tree.Nodes()[1].Value == 10);
    CHECK(tree.Nodes()[0].Value == 11);
}

} // namespace Operon::Test
