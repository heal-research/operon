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

#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/grammar.hpp"
#include "core/stats.hpp"
#include "operators/creator.hpp"
#include "operators/selection.hpp"
#include <algorithm>
#include <catch2/catch.hpp>
#include <execution>

namespace Operon::Test {
TEST_CASE("Selection Distribution")
{
    size_t nTrees = 1'000;
    size_t maxLength = 100;
    size_t maxDepth = 12;

    auto random = Operon::Random(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto creator = BalancedTreeCreator { sizeDistribution, maxDepth, maxLength };

    std::vector<Individual<1>> individuals(nTrees);
    Grammar grammar;
    for (size_t i = 0; i < nTrees; ++i) {
        individuals[i].Genotype = creator(random, grammar, inputs);
        individuals[i][0] = std::uniform_real_distribution(0.0, 1.0)(random);
    }

    using Ind = Individual<1>;
    constexpr gsl::index Idx = 0;

    ProportionalSelector<Ind, Idx> proportionalSelector;
    proportionalSelector.Prepare(individuals);

    TournamentSelector<Ind, Idx> tournamentSelector(2);
    tournamentSelector.Prepare(individuals);

    RankTournamentSelector<Ind, Idx> rankedSelector(2);
    rankedSelector.Prepare(individuals);

    auto plotHist = [&](SelectorBase<Ind, Idx>& selector)
    {
        std::vector<size_t> hist(individuals.size());

        for (size_t i = 0; i < 100 * nTrees; ++i)
        {
            hist[selector(random)]++;
        }
        std::sort(hist.begin(), hist.end(), std::greater<>{});
        for (size_t i = 0; i < nTrees; ++i)
        {
            //auto qty = std::string(hist[i], '*');
            auto qty = hist[i];
            fmt::print("{:>5}\t{}\n", i, qty / 100.0);
        }
    };

    SECTION("Proportional")
    {
        plotHist(proportionalSelector);
    }

    SECTION("Tournament Size 2")
    {
        plotHist(tournamentSelector);
    }

    SECTION("Rank Tournament Size 2")
    {
        plotHist(rankedSelector);
    }
    
    SECTION("Tournament Size 3")
    {
        tournamentSelector.TournamentSize(3);
        plotHist(tournamentSelector);
    }

    SECTION("Rank Tournament Size 3")
    {
        rankedSelector.TournamentSize(3);
        plotHist(rankedSelector);
    }
}
}
