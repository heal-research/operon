// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include "../operon_test.hpp"

#include "operon/core/pset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/parser/infix.hpp"

namespace nb = ankerl::nanobench;

namespace Operon::Test {

TEST_CASE("Parser throughput", "[performance]")
{
    constexpr auto nTrees{1000};
    constexpr auto maxLength{50};
    constexpr auto nrow{10};
    constexpr auto ncol{10};

    Operon::RandomGenerator rng(1234UL);
    auto ds = Util::RandomDataset(rng, nrow, ncol);

    Operon::PrimitiveSet pset;
    pset.SetConfig(PrimitiveSet::Arithmetic | NodeType::Exp | NodeType::Log | NodeType::Variable);

    BalancedTreeCreator creator{&pset, ds.VariableHashes(), /* bias= */ 0.0, maxLength};
    std::uniform_int_distribution<size_t> dist(1, maxLength);

    // Pre-generate infix strings
    std::vector<std::string> strings;
    strings.reserve(nTrees);
    size_t totalNodes{0};
    for (auto i = 0; i < nTrees; ++i) {
        auto tree = creator(rng, dist(rng), 0, 10);
        totalNodes += tree.Length();
        strings.push_back(InfixFormatter::Format(tree, ds, 20));
    }

    size_t idx{0};
    nb::Bench bench;
    bench.run("parse", [&]() -> void {
        auto tree = InfixParser::Parse(strings[idx], ds);
        nb::doNotOptimizeAway(tree);
        idx = (idx + 1) % nTrees;
    });

    // Derive nodes/second from nanobench's median ns/iteration
    auto const nsPerIter = bench.results().front().median(nb::Result::Measure::elapsed);
    auto const avgNodes = static_cast<double>(totalNodes) / nTrees;
    auto const nodesPerSec = avgNodes / nsPerIter * 1e9;
    fmt::print("Parser throughput: {:.2e} nodes/s (avg {:.1f} nodes/tree)\n", nodesPerSec, avgNodes);

    CHECK(nodesPerSec > 0);
}

} // namespace Operon::Test
