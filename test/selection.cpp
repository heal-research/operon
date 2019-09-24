#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/grammar.hpp"
#include "core/stats.hpp"
#include "operators/initialization.hpp"
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

    auto random = Random::JsfRand<64>();
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto creator = GrowTreeCreator { sizeDistribution, maxDepth, maxLength };

    std::vector<Individual<1>> individuals(nTrees);
    Grammar grammar;
    for (size_t i = 0; i < nTrees; ++i) {
        individuals[i].Genotype = creator(random, grammar, inputs);
        individuals[i][0] = std::uniform_real_distribution(0.0, 1.0)(random);
    }

    constexpr bool Maximization = true;

    ProportionalSelector<Individual<1>, 0, Maximization> proportionalSelector;
    proportionalSelector.Prepare(individuals);

    TournamentSelector<Individual<1>, 0, Maximization> tournamentSelector(2);
    tournamentSelector.Prepare(individuals);

    RankedTournamentSelector<Individual<1>, 0, Maximization> rankedSelector(2);
    rankedSelector.Prepare(individuals);

    auto plotHist = [&](SelectorBase<Individual<1>, 0, Maximization>& selector)
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

    SECTION("Tournament New Size 2")
    {
        plotHist(rankedSelector);
    }

    SECTION("Tournament Size 3")
    {
        tournamentSelector.TournamentSize(3);
        plotHist(tournamentSelector);
    }

    SECTION("Tournament New Size 3")
    {
        rankedSelector.TournamentSize(3);
        plotHist(rankedSelector);
    }
}
}
