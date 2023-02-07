// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
//
#include <algorithm>
#include <catch2/catch.hpp>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/core/grammar.hpp"
#include "operon/core/stats.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/selection.hpp"

namespace Operon::Test {
TEST_CASE("Selection Distribution")
{
    size_t nTrees = 1'000;
    size_t maxLength = 100;
    size_t maxDepth = 12;

    auto random = Operon::Random(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.GetVariables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    std::vector<Individual<1>> individuals(nTrees);
    PrimitiveSet grammar;
    auto creator = BalancedTreeCreator { grammar, inputs };
    for (size_t i = 0; i < nTrees; ++i) {
        individuals[i].Genotype = creator(random, sizeDistribution(random), maxDepth);
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
