/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
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

#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/stats.hpp"
#include "core/metrics.hpp"
#include "core/distance.hpp"
#include "core/pset.hpp"
#include "analyzers/diversity.hpp"
#include "operators/creator.hpp"

#include "nanobench.h"

namespace Operon {
namespace Test {
TEST_CASE("Intersection performance") 
{
    size_t n = 1000;
    size_t maxLength = 100;
    size_t maxDepth = 1000;

    Operon::RandomGenerator rd(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(maxLength, maxLength);

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic | NodeType::Exp | NodeType::Log);

    std::vector<Tree> trees(n);
    auto btc = BalancedTreeCreator { grammar, inputs };
    std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return btc(rd, sizeDistribution(rd), 0, maxDepth); });

    std::vector<Operon::Distance::HashVector> hashesStrict(trees.size());
    std::vector<Operon::Distance::HashVector> hashesStruct(trees.size());

    const auto F = Operon::HashFunction::XXHash;
    std::transform(trees.begin(), trees.end(), hashesStrict.begin(), [](Tree tree) { return MakeHashes<F>(tree, Operon::HashMode::Strict); });
    std::transform(trees.begin(), trees.end(), hashesStruct.begin(), [](Tree tree) { return MakeHashes<F>(tree, Operon::HashMode::Relaxed); });

    std::uniform_int_distribution<size_t> dist(0u, trees.size()-1);

    SUBCASE("Performance") {
        ankerl::nanobench::Bench b;
        b.performanceCounters(true).relative(true).minEpochIterations(100000);
        b.run("intersect strict",        [&](){ ankerl::nanobench::doNotOptimizeAway(Operon::Distance::CountIntersect(hashesStrict[dist(rd)], hashesStrict[dist(rd)])); });
        b.run("intersect struct",        [&](){ ankerl::nanobench::doNotOptimizeAway(Operon::Distance::CountIntersect(hashesStruct[dist(rd)], hashesStruct[dist(rd)])); });
        b.run("jaccard distance",        [&](){ ankerl::nanobench::doNotOptimizeAway(Operon::Distance::Jaccard(hashesStrict[dist(rd)], hashesStrict[dist(rd)])); });
        b.run("jaccard distance",        [&](){ ankerl::nanobench::doNotOptimizeAway(Operon::Distance::Jaccard(hashesStruct[dist(rd)], hashesStruct[dist(rd)])); });
        b.run("sorensen-dice distance",  [&](){ ankerl::nanobench::doNotOptimizeAway(Operon::Distance::SorensenDice(hashesStrict[dist(rd)], hashesStrict[dist(rd)])); });
        b.run("sorensen-dice distance",  [&](){ ankerl::nanobench::doNotOptimizeAway(Operon::Distance::SorensenDice(hashesStruct[dist(rd)], hashesStruct[dist(rd)])); });
    }
}
} // namespace Test
} // namespace Operon

