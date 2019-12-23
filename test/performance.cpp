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
#include <execution>

#include "core/eval.hpp"
#include "core/grammar.hpp"
#include "core/metrics.hpp"
#include "core/stats.hpp"
#include "core/distance.hpp"
#include "operators/selection.hpp"
#include "operators/creator.hpp"
#include "random/jsf.hpp"
#include "analyzers/diversity.hpp"

#include <tbb/task_scheduler_init.h>
#include <unordered_set>

namespace Operon::Test {

TEST_CASE("Evaluation performance", "[performance]")
{
    size_t n = 10'000;
    size_t maxLength = 50;
    size_t maxDepth = 1000;

    auto rd = Operon::Random();
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    Range range = { 0, ds.Rows() };

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto creator = BalancedTreeCreator { sizeDistribution, maxDepth, maxLength };

    std::vector<Tree> trees(n);
    std::vector<Operon::Scalar> fit(n);

    auto evaluate = [&](auto& tree) -> size_t {
        auto estimated = Evaluate<Operon::Scalar>(tree, ds, range);
        return estimated.size();
    };

    Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;

    auto print_performance = [&](auto d) {
#ifdef _MSC_VER
        auto totalNodes = std::reduce(trees.begin(), trees.end(), 0UL, [](size_t partial, const auto& t) { return partial + t.Length(); });
#else
        auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
#endif
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(d);
        fmt::print("\nTotal nodes: {}, elapsed: {} s, performance: {:.4e} nodes/s\n", totalNodes, elapsed.count() / 1000.0, totalNodes * ds.Rows() * 1000.0 / elapsed.count());
    };

    Grammar grammar;

    auto measurePerformance = [&]()
    {
        size_t k = 0;
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });
        // [+, -, *, /]
        model.start();
        BENCHMARK("Sequential")
        {
            ++k;
            std::transform(std::execution::seq, trees.begin(), trees.end(), fit.begin(), evaluate);
        };
        model.finish();
        print_performance(model.elapsed() / k);

        k = 0;
        model.start();
        BENCHMARK("Parallel")
        {
            ++k;
            std::transform(std::execution::par_unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
        };
        model.finish();
        print_performance(model.elapsed() / k);
    };

    SECTION("Arithmetic")
    {
        grammar.SetConfig(Grammar::Arithmetic);
        measurePerformance();
    }

    SECTION("Arithmetic + Exp + Log")
    {
        // [+, -, *, /, exp, log]
        grammar.SetConfig(Grammar::Arithmetic | NodeType::Exp | NodeType::Log);
        measurePerformance();
    }

    SECTION("Arithmetic + Sin + Cos")
    {
        // [+, -, *, /, exp, log]
        grammar.SetConfig(Grammar::Arithmetic | NodeType::Sin | NodeType::Cos);
        measurePerformance();
    }

    SECTION("Arithmetic + Exp + Log + Sin + Cos")
    {
        grammar.SetConfig(Grammar::Arithmetic | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos);
        measurePerformance();
    }

    SECTION("Arithmetic + Sqrt + Cbrt + Square")
    {
        // [+, -, *, /, exp, log]
        grammar.SetConfig(Grammar::Arithmetic | NodeType::Sqrt | NodeType::Cbrt | NodeType::Square);
        measurePerformance();
    }

    SECTION("Arithmetic + Exp + Log + Sin + Cos + Tan + Sqrt + Cbrt + Square")
    {
        // [+, -, *, /, exp, log]
        grammar.SetConfig(Grammar::Full);
        measurePerformance();
    }
}

TEST_CASE("Tree creation performance")
{
    size_t n = 5000;
    size_t maxLength = 100;
    size_t maxDepth = 100;

    auto rd = Operon::Random();
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    std::vector<Tree> trees(n);

    Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;

    auto print_performance = [&](auto d) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(d);
        fmt::print("\nElapsed: {} s, performance: {:.4e} trees/s, \n", elapsed.count() / 1000.0, trees.size() * 1000.0 / elapsed.count());
    };
    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic);
    double k = 0;
    auto btc = BalancedTreeCreator { sizeDistribution, maxDepth, maxLength };
    SECTION("Balanced tree creator")
    {
        k = 0;
        model.start();
        BENCHMARK("Sequential")
        {
            ++k;
            std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return btc(rd, grammar, inputs); });
        };
        model.finish();
        print_performance(model.elapsed() / k);

        k = 0;
        model.start();
        BENCHMARK("Parallel")
        {
            ++k;
            std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return btc(rd, grammar, inputs); });
        };
        model.finish();
        print_performance(model.elapsed() / k);
    }

    auto utc = UniformTreeCreator{ sizeDistribution, maxDepth, maxLength };
    SECTION("Uniform tree creator")
    {
        k = 0;
        model.start();
        BENCHMARK("Sequential")
        {
            ++k;
            std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return utc(rd, grammar, inputs); });
        };
        model.finish();
        print_performance(model.elapsed() / k);

        k = 0;
        model.start();
        BENCHMARK("Parallel")
        {
            ++k;
            std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return utc(rd, grammar, inputs); });
        };
        model.finish();
        print_performance(model.elapsed() / k);
    }
}

TEST_CASE("Tree hashing performance") {
    size_t n = 100000;
    size_t maxLength = 200;
    size_t maxDepth = 100;

    auto rd = Operon::Random();
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic);

    std::vector<Tree> trees(n);
    auto btc = BalancedTreeCreator { sizeDistribution, maxDepth, maxLength };
    std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return btc(rd, grammar, inputs); });
    Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;

    model.start();
    std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [](auto& t) {t.Sort(Operon::HashMode::Strict); });
    model.finish();
    auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed());
    auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });

    fmt::print("\nElapsed: {} s, performance: {:.4e} nodes/second.\n", elapsed.count() / 1000.0, totalNodes * 1000.0 / elapsed.count());
}

TEST_CASE("Hash collisions") {
    size_t n = 1000000;
    size_t maxLength = 200;
    size_t maxDepth = 100;

    auto rd = Operon::Random(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic);

    std::vector<size_t> indices(n);
    std::vector<Operon::Hash> seeds(n);
    std::vector<Tree> trees(n);

    auto btc = BalancedTreeCreator { sizeDistribution, maxDepth, maxLength };

    std::iota(indices.begin(), indices.end(), 0);
    std::generate(std::execution::unseq, seeds.begin(), seeds.end(), [&](){ return rd(); });
    std::transform(std::execution::par_unseq, indices.begin(), indices.end(), trees.begin(), [&](auto i) {
        Operon::Random rand(seeds[i]);
        auto tree = btc(rand, grammar, inputs);
        tree.Sort(Operon::HashMode::Strict);
        return tree;
            });

    std::unordered_set<uint64_t> set64; 
    std::unordered_set<uint32_t> set32; 

    auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<>{}, [](auto& tree) { return tree.Length(); });

    for(auto& tree : trees) {
        for(auto& node : tree.Nodes()) {
            auto h = node.CalculatedHashValue;
            set64.insert(h);
            set32.insert(static_cast<uint32_t>(h & 0xFFFFFFFFLL));
        }
        tree.Nodes().clear();
    }
    double s64 = set64.size();
    double s32 = set32.size();
    fmt::print("total nodes: {}, {:.3f}% unique, unique 64-bit hashes: {}, unique 32-bit hashes: {}, collision rate: {:.3f}%\n", totalNodes, s64/totalNodes * 100, s64, s32, (1 - s32/s64) * 100);
}

TEST_CASE("Tree distance performance") 
{
    size_t n = 1000;
    size_t maxLength = 100;
    size_t maxDepth = 100;

    auto rd = Operon::Random(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic | NodeType::Exp | NodeType::Log);

    std::vector<Tree> trees(n);
    auto btc = BalancedTreeCreator { sizeDistribution, maxDepth, maxLength };
    std::generate(std::execution::unseq, trees.begin(), trees.end(), [&]() { return btc(rd, grammar, inputs); });
    Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;

    SECTION("Strict diversity") {
        std::vector<Operon::Distance::HashVector> hashes(trees.size());
        std::transform(trees.begin(), trees.end(), hashes.begin(), [](Tree tree) { return MakeHashes(tree, Operon::HashMode::Strict); });
        size_t reps = 50;
        
        // measure speed of vectorized intersection
        auto totalOps = n * (n-1) / 2;
        double tMean, tStddev, opsPerSecond;
        double diversity;

        // measured speed of vectorized intersection
        MeanVarianceCalculator elapsedCalc;
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    double s = hashes[i].size() + hashes[j].size();
                    size_t c = Operon::Distance::CountIntersectSIMD(hashes[i], hashes[j]);
                    calc.Add(1 - c / s);
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed()).count();
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = 1000 * totalOps / tMean; // from ms to second
        fmt::print("strict diversity (vector): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);

        // measured speed of scalar intersection
        elapsedCalc.Reset();
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    double s = hashes[i].size() + hashes[j].size();
                    size_t c = Operon::Distance::CountIntersect(hashes[i], hashes[j]);
                    calc.Add(1 - c / s);
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed()).count();
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = 1000 * totalOps / tMean;
        fmt::print("strict diversity (scalar): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);
    }

    SECTION("Relaxed diversity") {
        std::vector<Operon::Distance::HashVector> hashes(trees.size());
        std::transform(trees.begin(), trees.end(), hashes.begin(), [](Tree tree) { return MakeHashes(tree, Operon::HashMode::Relaxed); });
        size_t reps = 50;
        
        // measure speed of vectorized intersection
        auto totalOps = n * (n-1) / 2;
        double tMean, tStddev, opsPerSecond;
        double diversity;

        // measured speed of vectorized intersection
        MeanVarianceCalculator elapsedCalc;
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    double s = hashes[i].size() + hashes[j].size();
                    size_t c = Operon::Distance::CountIntersectSIMD(hashes[i], hashes[j]);
                    calc.Add(1 - c / s);
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed()).count();
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = 1000 * totalOps / tMean; // from ms to second
        fmt::print("relaxed diversity (vector): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);

        // measured speed of scalar intersection
        elapsedCalc.Reset();
        for(size_t k = 0; k < reps; ++k) {
            MeanVarianceCalculator calc;
            model.start();
            for (size_t i = 0; i < hashes.size() - 1; ++i) {
                for (size_t j = i+1; j < hashes.size(); ++j) {
                    double s = hashes[i].size() + hashes[j].size();
                    size_t c = Operon::Distance::CountIntersect(hashes[i], hashes[j]);
                    calc.Add(1 - c / s);
                }
            }
            model.finish();
            diversity = calc.Mean();
            auto ms = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed()).count();
            elapsedCalc.Add(ms);
        };
        tMean        = elapsedCalc.Mean();
        tStddev      = elapsedCalc.StandardDeviation();
        opsPerSecond = 1000 * totalOps / tMean;
        fmt::print("relaxed diversity (scalar): {:.6f}, elapsed ms: {:.3f} ± {:.3f}, speed: {:.3e} operations/s\n", diversity, tMean, tStddev, opsPerSecond);
    }
}

TEST_CASE("Selection performance")
{
    size_t nTrees = 10'000;
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

    using Ind = Individual<1>;

    std::vector<Ind> individuals(nTrees);
    Grammar grammar;
    for(size_t i = 0; i < nTrees; ++i) {
        individuals[i].Genotype = creator(random, grammar, inputs);
        individuals[i][0] = std::uniform_real_distribution(0.0, 1.0)(random);
    }

    auto benchSelector = [&](SelectorBase<Individual<1>, 0>& selector) -> size_t {
        size_t sum = 0u;
        for (size_t i = 0; i < nTrees; ++i) {
            sum += selector(random); 
        }
        return sum;
    };

    SECTION("Tournament Selector")
    {
        TournamentSelector<Ind, 0> tournamentSelector(2);
        tournamentSelector.Prepare(individuals);
        BENCHMARK("Tournament (prepare)") { tournamentSelector.Prepare(individuals);                                  };
        // unfortunately due to how Catch works we have to unroll this 
        BENCHMARK("Tournament size 2")    { tournamentSelector.TournamentSize(2); benchSelector(tournamentSelector);  };
        BENCHMARK("Tournament size 3")    { tournamentSelector.TournamentSize(3); benchSelector(tournamentSelector);  };
        BENCHMARK("Tournament size 4")    { tournamentSelector.TournamentSize(4); benchSelector(tournamentSelector);  };
        BENCHMARK("Tournament size 5")    { tournamentSelector.TournamentSize(5); benchSelector(tournamentSelector);  };
        BENCHMARK("Tournament size 6")    { tournamentSelector.TournamentSize(6); benchSelector(tournamentSelector);  };
        BENCHMARK("Tournament size 7")    { tournamentSelector.TournamentSize(7); benchSelector(tournamentSelector);  };
        BENCHMARK("Tournament size 8")    { tournamentSelector.TournamentSize(8); benchSelector(tournamentSelector);  };
        BENCHMARK("Tournament size 9")    { tournamentSelector.TournamentSize(9); benchSelector(tournamentSelector);  };
        BENCHMARK("Tournament size 10")   { tournamentSelector.TournamentSize(10); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 11")   { tournamentSelector.TournamentSize(11); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 12")   { tournamentSelector.TournamentSize(12); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 13")   { tournamentSelector.TournamentSize(13); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 14")   { tournamentSelector.TournamentSize(14); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 15")   { tournamentSelector.TournamentSize(15); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 16")   { tournamentSelector.TournamentSize(16); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 17")   { tournamentSelector.TournamentSize(17); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 18")   { tournamentSelector.TournamentSize(18); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 19")   { tournamentSelector.TournamentSize(19); benchSelector(tournamentSelector); };
        BENCHMARK("Tournament size 20")   { tournamentSelector.TournamentSize(20); benchSelector(tournamentSelector); };
    }

}
}

