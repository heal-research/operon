/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
 */

#include <catch2/catch.hpp>

#include "core/node.hpp"
#include "core/grammar.hpp"
#include "operators/initialization.hpp"

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
} // namespace Operon::Test
