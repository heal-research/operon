// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>
#include <random>

#include "../operon_test.hpp"

#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/core/tree.hpp"

namespace nb = ankerl::nanobench;

namespace Operon::Test {

TEST_CASE("BTC creation throughput", "[performance]")
{
    constexpr auto nrow{10};
    constexpr auto ncol{10};
    constexpr auto maxd{10};
    constexpr auto maxl{100};

    Operon::PrimitiveSet pset{Operon::PrimitiveSet::Arithmetic};

    Operon::RandomGenerator rd(1234UL);
    auto ds = Util::RandomDataset(rd, nrow, ncol);
    auto inputs = ds.VariableHashes();

    BalancedTreeCreator creator{&pset, inputs, /* bias= */ 0.0, maxl};
    std::uniform_int_distribution<size_t> dist(1, maxl);

    nb::Bench bench;
    bench.run("btc", [&]() {
        return creator(rd, dist(rd), 0, maxd);
    });

    // Just verify it produces valid trees
    auto tree = creator(rd, dist(rd), 0, maxd);
    CHECK(tree.Length() > 0);
}

} // namespace Operon::Test
