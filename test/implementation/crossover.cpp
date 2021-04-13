#include <core/types.hpp>
#include <doctest/doctest.h>
#include <random>
#include <stat/meanvariance.hpp>

#include "core/dataset.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/pset.hpp"
#include "core/stats.hpp"
#include "operators/creator.hpp"
#include "operators/crossover.hpp"

namespace Operon::Test {
TEST_CASE("Crossover")
{
    auto target = "Y";
    auto ds = Dataset("../data/Poly-10.csv", true);
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });

    Range range { 0, 250 };

    PrimitiveSet grammar;
    grammar.SetConfig(PrimitiveSet::Arithmetic);

    BalancedTreeCreator btc { grammar, inputs, /* bias= */ 0.0 };

    Operon::RandomGenerator random(1234);

    SUBCASE("Simple swap")
    {
        Operon::RandomGenerator rng(std::random_device{}());
        size_t maxDepth{1000}, maxLength{100};
        Operon::SubtreeCrossover cx(0.9, maxDepth, maxLength);
        auto p1 = btc(rng, 7, 1, maxDepth);
        auto p2 = btc(rng, 5, 1, maxDepth);
        auto child = cx(rng, p1, p2); 

        fmt::print("parent 1\n{}\n", TreeFormatter::Format(p1, ds, 2));
        fmt::print("parent 2\n{}\n", TreeFormatter::Format(p2, ds, 2));
        fmt::print("child\n{}\n", TreeFormatter::Format(child, ds, 2));
    }

    SUBCASE("Distribution of swap locations")
    {
        Operon::RandomGenerator rng(std::random_device{}());

        size_t maxDepth{1000}, maxLength{20};
        Operon::SubtreeCrossover cx(1.0, maxDepth, maxLength);

        std::vector<double> c1(maxLength);
        std::vector<double> c2(maxLength);

        uint64_t p1_term{0}, p1_func{0};
        uint64_t p2_term{0}, p2_func{0};

        for (int n = 0; n < 100000; ++n) {
            auto p1 = btc(rng, maxLength, 1, maxDepth);
            //auto p2 = btc(rng, maxLength, 1, maxDepth);
            auto p2 = p1;

            auto [i, j] = cx.FindCompatibleSwapLocations(rng, p1, p2);
            c1[i]++;
            c2[j]++;

            p1_term += p1[i].IsLeaf();
            p1_func += !p1[i].IsLeaf();

            p2_term += p2[i].IsLeaf();
            p2_func += !p2[i].IsLeaf();
        }

        fmt::print("p1_term: {}, p1_func: {}\n", p1_term, p1_func);
        fmt::print("p2_term: {}, p2_func: {}\n", p2_term, p2_func);

        fmt::print("parents swap location sampling counts:\n");
        for (size_t i = 0; i < maxLength; ++i) {
            fmt::print("{} {} {}\n", i, c1[i], c2[i]);
        }
    }

    SUBCASE("Child size") {
        const int n = 100000;
        std::vector<Tree> trees;
        size_t maxDepth{1000}, maxLength{100};
        auto sizeDistribution = std::uniform_int_distribution<size_t>(1, maxLength);
        for (int i = 0; i < n; ++i) trees.push_back(btc(random, sizeDistribution(random), 1, maxDepth));

        std::uniform_int_distribution<size_t> dist(0, n-1);
        for (auto p : { 0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0 }) {
            PearsonsRCalculator calc;
            MeanVarianceCalculator mv;
            Operon::SubtreeCrossover cx(p, maxDepth, maxLength);
            for (int i = 0; i < n; ++i) {
                auto p1 = dist(random);
                //auto p2 = dist(random);
                auto c = cx(random, trees[p1], trees[p1]);
                calc.Add(trees[p1].Length(), trees[p1].Length());
                mv.Add((double)c.Length());
            }
            fmt::print("p: {:.1f}, parent1: {:.2f}, parent2: {:.2f}, child: {:.2f}\n", p, calc.MeanX(), calc.MeanY(), mv.Mean());
        }
    }
}
}

