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

#include <doctest/doctest.h>

#include "core/common.hpp"
#include "core/operator.hpp"
#include "core/tree.hpp"

#include <ceres/ceres.h>

namespace dt = doctest;

namespace Operon {
namespace Test {
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

    TEST_CASE("Node is pod" * dt::test_suite("[detail]"))
    {
        CHECK(std::is_pod_v<Operon::Node>);
    }

    TEST_CASE("Node is small" * dt::test_suite("[detail]"))
    {
        // this test case basically wants to ensure that,
        // for memory efficiency purposes, the Node struct
        // is kept as small as possible
        auto node = static_cast<Node*>(nullptr);
        auto szType = sizeof(node->Type);
        auto szArity = sizeof(node->Arity);
        auto szLength = sizeof(node->Length);
        auto szDepth = sizeof(node->Depth);
        auto szLevel = sizeof(node->Level);
        auto szEnabled = sizeof(node->IsEnabled);
        auto szHashValue = sizeof(node->HashValue);
        auto szCalculatedHashValue = sizeof(node->CalculatedHashValue);
        auto szValue = sizeof(node->Value);
        auto szParent = sizeof(node->Parent);
        auto szTotal = szType + szArity + szLength + szDepth + szLevel + szEnabled + szHashValue + szParent + szCalculatedHashValue + szValue;
        fmt::print("Size breakdown of the Node class:\n");
        fmt::print("Type                {:>2}\n", szType);
        fmt::print("Arity               {:>2}\n", szArity);
        fmt::print("Length              {:>2}\n", szLength);
        fmt::print("Depth               {:>2}\n", szDepth);
        fmt::print("Level               {:>2}\n", szLevel);
        fmt::print("Parent              {:>2}\n", szParent);
        fmt::print("Enabled             {:>2}\n", szEnabled);
        fmt::print("Value               {:>2}\n", szValue);
        fmt::print("HashValue           {:>2}\n", szHashValue);
        fmt::print("CalculatedHashValue {:>2}\n", szCalculatedHashValue);
        fmt::print("-------------------------\n");
        fmt::print("Total               {:>2}\n", szTotal);
        fmt::print("Total + padding     {:>2}\n", sizeof(Node));
        fmt::print("-------------------------\n");
        Operon::Vector<Node> nodes;
        std::generate_n(std::back_inserter(nodes), 50, []() { return Node(NodeType::Add); });
        Tree tree { nodes };
        fmt::print("sizeof(Tree)        {:>2}\n", sizeof(tree));
        fmt::print("sizeof(vector<Node>) {:>2}\n", sizeof(nodes));
        Individual ind(1);
        ind.Genotype = std::move(tree);

        fmt::print("sizeof(Individual)  {:>2}\n", sizeof(ind));

        CHECK(sizeof(Node) <= size_t { 64 });
    }
} // namespace Test
} // namespace Operon
