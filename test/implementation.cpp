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

#include <catch2/catch.hpp>

#include "core/node.hpp"
#include "core/grammar.hpp"
#include "operators/initialization.hpp"

#include "random/jsf.hpp"
#include "random/sfc64.hpp"

namespace Operon::Test {
TEST_CASE("Node is trivial")
{
    REQUIRE(std::is_trivial_v<Operon::Node>);
}

TEST_CASE("Node is small")
{
    // this test case basically wants to ensure that, 
    // for memory efficiency purposes, the Node struct
    // is kept as small as possible
    auto node                  = static_cast<Node*>(nullptr);
    auto szType                = sizeof(node->Type);
    auto szArity               = sizeof(node->Arity);
    auto szLength              = sizeof(node->Length);
    auto szDepth               = sizeof(node->Depth);
    auto szEnabled             = sizeof(node->IsEnabled);
    auto szHashValue           = sizeof(node->HashValue);
    auto szCalculatedHashValue = sizeof(node->CalculatedHashValue);
    auto szValue               = sizeof(node->Value);
    auto szParent              = sizeof(node->Parent);
    fmt::print("Size breakdown of the Node class:\n");
    fmt::print("Type                {:>2}\n", szType);
    fmt::print("Arity               {:>2}\n", szArity);
    fmt::print("Length              {:>2}\n", szLength);
    fmt::print("Depth               {:>2}\n", szDepth);
    fmt::print("Parent              {:>2}\n", szParent);
    fmt::print("Enabled             {:>2}\n", szEnabled);
    fmt::print("Value               {:>2}\n", szValue);
    fmt::print("HashValue           {:>2}\n", szHashValue);
    fmt::print("CalculatedHashValue {:>2}\n", szCalculatedHashValue);
    fmt::print("-------------------------\n");
    auto szTotal = szType + szArity + szLength + szDepth + szEnabled + szHashValue + szParent + szCalculatedHashValue + szValue;
    fmt::print("Total               {:>2}\n", szTotal); 
    fmt::print("Total + padding     {:>2}\n", sizeof(Node));
    fmt::print("-------------------------\n");
    fmt::print("sizeof(Tree)        {:>2}\n", sizeof(Tree));
    fmt::print("sizeof(vector<Node>) {:>2}\n", sizeof(std::vector<Node>));

    REQUIRE(sizeof(Node) <= size_t{64});
}

TEST_CASE("Jsf is copyable") 
{
    Random::JsfRand<64> jsf(1234);
    jsf();
    jsf();
    auto tmp = jsf;

    REQUIRE(tmp() == jsf());
}

TEST_CASE("Sfc64 is copyable")
{
    Random::Sfc64 sfc(1234);
    sfc();
    sfc();
    auto tmp = sfc;

    REQUIRE(tmp() == sfc());
}
} // namespace Operon::Test
