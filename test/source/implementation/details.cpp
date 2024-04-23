// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <cstddef>
#include <doctest/doctest.h>
#include <fmt/core.h>
#include <iterator>
#include <type_traits>
#include <utility>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace dt = doctest;

namespace Operon::Test {

    TEST_CASE("Node is trivial" * dt::test_suite("[detail]"))
    {
        CHECK(std::is_trivial_v<Operon::Node>);
    }

    TEST_CASE("Node is trivially copyable" * dt::test_suite("[detail]"))
    {
        CHECK(std::is_trivially_copyable_v<Operon::Node>);
    }

    TEST_CASE("Node is standard layout" * dt::test_suite("[detail]"))
    {
        CHECK(std::is_standard_layout_v<Operon::Node>);
    }

    TEST_CASE("Node is small" * dt::test_suite("[detail]"))
    {
        // this test case basically wants to ensure that,
        // for memory efficiency purposes, the Node struct
        // is kept as small as possible
        auto *node = static_cast<Node*>(nullptr);
        auto szType = sizeof(node->Type);
        auto szArity = sizeof(node->Arity);
        auto szLength = sizeof(node->Length);
        auto szDepth = sizeof(node->Depth);
        auto szLevel = sizeof(node->Level);
        auto szEnabled = sizeof(node->IsEnabled);
        auto szOptimize = sizeof(node->Optimize);
        auto szHashValue = sizeof(node->HashValue);
        auto szCalculatedHashValue = sizeof(node->CalculatedHashValue);
        auto szValue = sizeof(node->Value);
        auto szParent = sizeof(node->Parent);
        auto szTotal = szType + szArity + szLength + szDepth + szLevel + szEnabled + szOptimize + szHashValue + szParent + szCalculatedHashValue + szValue;
        fmt::print("Size breakdown of the Node class:\n");
        fmt::print("Type                {:>2}\n", szType);
        fmt::print("Arity               {:>2}\n", szArity);
        fmt::print("Length              {:>2}\n", szLength);
        fmt::print("Depth               {:>2}\n", szDepth);
        fmt::print("Level               {:>2}\n", szLevel);
        fmt::print("Parent              {:>2}\n", szParent);
        fmt::print("Enabled             {:>2}\n", szEnabled);
        fmt::print("Optimize            {:>2}\n", szOptimize);
        fmt::print("Value               {:>2}\n", szValue);
        fmt::print("HashValue           {:>2}\n", szHashValue);
        fmt::print("CalculatedHashValue {:>2}\n", szCalculatedHashValue);
        fmt::print("-------------------------\n");
        fmt::print("Total               {:>2}\n", szTotal);
        fmt::print("Total + padding     {:>2}\n", sizeof(Node));
        fmt::print("-------------------------\n");
        Operon::Vector<Node> nodes;
        std::generate_n(std::back_inserter(nodes), 50, []() { return Node(NodeType::Add); }); // NOLINT
        Tree tree { nodes };
        fmt::print("sizeof(Tree)        {:>2}\n", sizeof(tree));
        fmt::print("sizeof(vector<Node>) {:>2}\n", sizeof(nodes)); // NOLINT
        Individual ind(1);
        ind.Genotype = std::move(tree);

        fmt::print("sizeof(Individual)  {:>2}\n", sizeof(ind));

        CHECK(sizeof(Node) <= size_t { 64 });
    }
} // namespace Operon::Test
