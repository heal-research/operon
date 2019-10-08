#include <catch2/catch.hpp>
#include <execution>

#include "core/eval.hpp"
#include "core/grammar.hpp"
#include "core/jsf.hpp"
#include "core/metrics.hpp"
#include "core/stats.hpp"
#include "operators/selection.hpp"
#include "operators/initialization.hpp"

#include <tbb/task_scheduler_init.h>

namespace Operon::Test {

TEST_CASE("Sextic GPops", "[performance]")
{
//    auto threads = tbb::task_scheduler_init::default_num_threads();
//    tbb::task_scheduler_init init(threads);
    auto rd = Random::JsfRand<64>();
    auto ds = Dataset("../data/Sextic.csv", true);
    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    size_t n = 10'000;
    std::vector<size_t> numRows { 1000 };
    std::vector<size_t> avgLen { 50 };
    std::vector<operon::scalar_t> results;

    size_t maxDepth = 10000;
    size_t maxLength = 10000;

    Grammar grammar;

    for (auto len : avgLen) {
        std::uniform_int_distribution<size_t> sizeDistribution(len, len);
        auto creator = GrowTreeCreator { sizeDistribution, maxDepth, len };
        std::vector<Tree> trees(n);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); });
        for(auto nRows : numRows) {

            Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;
            Range range { 0, nRows };

            size_t reps = 0;
            model.start();
            BENCHMARK("Parallel")
            {
                ++reps;
                std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [&](const auto& tree) { return Evaluate<float>(tree, ds, range).size(); });
            };
            model.finish();
            auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed() / reps);
            auto gpops = totalNodes * range.Size() * 1000.0 / elapsed.count();
	    fmt::print("Float,{},{},{:.4e}\n", len, nRows, gpops);

            reps = 0;
            model.start();
            BENCHMARK("Parallel")
            {
                ++reps;
                std::for_each(std::execution::par_unseq, trees.begin(), trees.end(), [&](const auto& tree) { return Evaluate<operon::scalar_t>(tree, ds, range).size(); });
            };
            model.finish();
            totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, std::plus<> {}, [](auto& tree) { return tree.Length(); });
            elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(model.elapsed() / reps);
            gpops = totalNodes * range.Size() * 1000.0 / elapsed.count();
	    fmt::print("Double,{},{},{:.4e}\n", len, nRows, gpops);
        }
    }
}

TEST_CASE("Evaluation performance", "[performance]")
{
    size_t n = 10'000;
    size_t maxLength = 100;
    size_t maxDepth = 1000;

    auto rd = Random::JsfRand<64>();
    auto ds = Dataset("../data/Sextic.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    Range range = { 0, ds.Rows() };

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto creator = GrowTreeCreator { sizeDistribution, maxDepth, maxLength };

    std::vector<Tree> trees(n);
    std::vector<operon::scalar_t> fit(n);

    auto evaluate = [&](auto& tree) -> size_t {
        auto estimated = Evaluate<operon::scalar_t>(tree, ds, range);
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
    size_t maxDepth = 12;

    auto rd = Random::JsfRand<64>();
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto growCreator = GrowTreeCreator { sizeDistribution, maxDepth, maxLength };

    std::vector<Tree> trees(n);

    Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;

    auto print_performance = [&](auto d) {
        auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(d);
        fmt::print("\nElapsed: {} s, performance: {:.4e} trees/s, \n", elapsed.count() / 1000.0, trees.size() * 1000.0 / elapsed.count());
    };
    Grammar grammar;
    grammar.SetConfig(Grammar::Arithmetic);
    size_t k = 0;
    SECTION("Grow tree creator")
    {
        k = 0;
        model.start();
        BENCHMARK("Sequential")
        {
            ++k;
            std::generate(std::execution::seq, trees.begin(), trees.end(), [&]() { return growCreator(rd, grammar, inputs); });
        };
        model.finish();
        print_performance(model.elapsed() / k);

        k = 0;
        model.start();
        BENCHMARK("Parallel")
        {
            ++k;
            std::generate(std::execution::par_unseq, trees.begin(), trees.end(), [&]() { return growCreator(rd, grammar, inputs); });
        };
        model.finish();
        print_performance(model.elapsed() / k);
    }
}

TEST_CASE("Selection performance")
{
    size_t nTrees = 10'000;
    size_t maxLength = 100;
    size_t maxDepth = 12;

    auto random = Random::JsfRand<64>(1234);
    auto ds = Dataset("../data/Poly-10.csv", true);

    auto target = "Y";
    auto variables = ds.Variables();
    std::vector<Variable> inputs;
    std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](const auto& v) { return v.Name != target; });

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto creator = GrowTreeCreator { sizeDistribution, maxDepth, maxLength };

    using Ind = Individual<1>;

    std::vector<Ind> individuals(nTrees);
    Grammar grammar;
    for(size_t i = 0; i < nTrees; ++i) {
        individuals[i].Genotype = creator(random, grammar, inputs);
        individuals[i][0] = std::uniform_real_distribution(0.0, 1.0)(random);
    }

    auto benchSelector = [&](SelectorBase<Individual<1>, 0, true>& selector) -> size_t {
        size_t sum = 0u;
        for (size_t i = 0; i < nTrees; ++i) {
            sum += selector(random); 
        }
        return sum;
    };

    SECTION("Tournament Selector")
    {
        TournamentSelector<Ind, 0, true> tournamentSelector(2);
        tournamentSelector.Prepare(individuals);
        BENCHMARK("Tournament (prepare)") { tournamentSelector.Prepare(individuals);                                    };
        // unfortunately due to how Catch works we have to unroll this 
        BENCHMARK("Tournament size 2")    { tournamentSelector.TournamentSize(2); benchSelector(tournamentSelector);    };
        BENCHMARK("Tournament size 3")    { tournamentSelector.TournamentSize(3); benchSelector(tournamentSelector);    };
        BENCHMARK("Tournament size 4")    { tournamentSelector.TournamentSize(4); benchSelector(tournamentSelector);    };
        BENCHMARK("Tournament size 5")    { tournamentSelector.TournamentSize(5); benchSelector(tournamentSelector);    };
        BENCHMARK("Tournament size 6")    { tournamentSelector.TournamentSize(6); benchSelector(tournamentSelector);    };
        BENCHMARK("Tournament size 7")    { tournamentSelector.TournamentSize(7); benchSelector(tournamentSelector);    };
        BENCHMARK("Tournament size 8")    { tournamentSelector.TournamentSize(8); benchSelector(tournamentSelector);    };
        BENCHMARK("Tournament size 9")    { tournamentSelector.TournamentSize(9); benchSelector(tournamentSelector);    };
        BENCHMARK("Tournament size 10")   { tournamentSelector.TournamentSize(10); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 11")   { tournamentSelector.TournamentSize(11); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 12")   { tournamentSelector.TournamentSize(12); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 13")   { tournamentSelector.TournamentSize(13); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 14")   { tournamentSelector.TournamentSize(14); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 15")   { tournamentSelector.TournamentSize(15); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 16")   { tournamentSelector.TournamentSize(16); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 17")   { tournamentSelector.TournamentSize(17); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 18")   { tournamentSelector.TournamentSize(18); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 19")   { tournamentSelector.TournamentSize(19); benchSelector(tournamentSelector);   };
        BENCHMARK("Tournament size 20")   { tournamentSelector.TournamentSize(20); benchSelector(tournamentSelector);   };
    }

//    SECTION("Tournament Selector (optimized)")
//    {
//        RankTournamentSelector<Ind, 0, true> rankedSelector(2);
//        rankedSelector.Prepare(individuals);
//        BENCHMARK("Tournament (prepare)") { rankedSelector.Prepare(individuals);                                   };
//        // unfortunately due to how Catch works we have to unroll this 
//        BENCHMARK("Tournament size 2")    { rankedSelector.TournamentSize(2); benchSelector(rankedSelector);  };
//        BENCHMARK("Tournament size 3")    { rankedSelector.TournamentSize(3); benchSelector(rankedSelector);  };
//        BENCHMARK("Tournament size 4")    { rankedSelector.TournamentSize(4); benchSelector(rankedSelector);  };
//        BENCHMARK("Tournament size 5")    { rankedSelector.TournamentSize(5); benchSelector(rankedSelector);  };
//        BENCHMARK("Tournament size 6")    { rankedSelector.TournamentSize(6); benchSelector(rankedSelector);  };
//        BENCHMARK("Tournament size 7")    { rankedSelector.TournamentSize(7); benchSelector(rankedSelector);  };
//        BENCHMARK("Tournament size 8")    { rankedSelector.TournamentSize(8); benchSelector(rankedSelector);  };
//        BENCHMARK("Tournament size 9")    { rankedSelector.TournamentSize(9); benchSelector(rankedSelector);  };
//        BENCHMARK("Tournament size 10")   { rankedSelector.TournamentSize(10); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 11")   { rankedSelector.TournamentSize(11); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 12")   { rankedSelector.TournamentSize(12); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 13")   { rankedSelector.TournamentSize(13); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 14")   { rankedSelector.TournamentSize(14); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 15")   { rankedSelector.TournamentSize(15); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 16")   { rankedSelector.TournamentSize(16); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 17")   { rankedSelector.TournamentSize(17); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 18")   { rankedSelector.TournamentSize(18); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 19")   { rankedSelector.TournamentSize(19); benchSelector(rankedSelector); };
//        BENCHMARK("Tournament size 20")   { rankedSelector.TournamentSize(20); benchSelector(rankedSelector); };
//    }

//    SECTION("Proportional Selector")
//    {
//        ProportionalSelector<Individual<1>, 0, true> proportionalSelector;
//        proportionalSelector.Prepare(individuals);
//
//        BENCHMARK("Proportional proportionalSelector (prepare)") { proportionalSelector.Prepare(individuals); };
//        BENCHMARK("Proportional selection") { benchSelector(proportionalSelector); };
//    }
}
}
