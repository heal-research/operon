#include <catch2/catch.hpp>
#include <execution>

#include "core/eval.hpp"
#include "core/jsf.hpp"
#include "core/grammar.hpp"
#include "core/stats.hpp"
#include "operators/initialization.hpp"

namespace Operon::Test
{
    TEST_CASE("Evaluation performance", "[performance]")
    {
        size_t n = 5000;
        size_t maxLength = 100;
        size_t maxDepth = 12;

        auto rd = Random::JsfRand<64>();
        auto ds = Dataset("../data/poly-10.csv", true);

        auto target = "Y";
        auto targetValues = ds.GetValues(target);
        auto variables = ds.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });

        Range range = { 0, ds.Rows() };

        auto creator = GrowTreeCreator(maxDepth, maxLength);

        std::vector<Tree> trees(n);
        std::vector<double> fit(n);

        auto grammar = Grammar();
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 

        auto evaluate = [&](auto& tree)
        {
            auto estimated = Evaluate<double>(tree, ds, range); 
            auto r2 = RSquared(estimated.begin(), estimated.end(), targetValues.begin() + range.Start);
            return r2;
        };

        Catch::Benchmark::Detail::ChronometerModel<std::chrono::steady_clock> model;

        auto totalNodes = std::transform_reduce(std::execution::par_unseq, trees.begin(), trees.end(), 0UL, [](size_t lhs, size_t rhs) { return lhs + rhs; }, [](auto& tree) { return tree.Length();} );

        auto print_performance = [&](auto d)
        {
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(d);
            fmt::print("\nTotal nodes: {}, elapsed: {} s, performance: {} nodes/s\n", totalNodes, elapsed.count() / 1000.0, totalNodes * ds.Rows() * 1000.0 / elapsed.count());
        };

        SECTION("Arithmetic")
        {
            size_t k = 0;
            grammar.SetConfig(Grammar::Arithmetic);
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 
            // [+, -, *, /]
            model.start();
            BENCHMARK("Sequential")
            {
                ++k;
                std::transform(std::execution::unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
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
        }

        SECTION("Arithmetic + Exp + Log")
        {
            // [+, -, *, /, exp, log]
            size_t k = 0;
            grammar.SetConfig(Grammar::Arithmetic | NodeType::Exp | NodeType::Log);
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 
            model.start();
            BENCHMARK("Sequential")
            {
                ++k;
                std::transform(std::execution::unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
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
        }

        SECTION("Arithmetic + Sin + Cos")
        {
            // [+, -, *, /, exp, log]
            size_t k = 0;
            grammar.SetConfig(Grammar::Arithmetic | NodeType::Sin | NodeType::Cos);
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 
            model.start();
            BENCHMARK("Sequential")
            {
                ++k;
                std::transform(std::execution::unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
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
        }

        SECTION("Arithmetic + Sqrt + Cbrt + Square")
        {
            // [+, -, *, /, exp, log]
            size_t k = 0;
            grammar.SetConfig(Grammar::Arithmetic | NodeType::Sqrt | NodeType::Cbrt | NodeType::Square);
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 
            model.start();
            BENCHMARK("Sequential")
            {
                ++k;
                std::transform(std::execution::unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
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
        }
        
        SECTION("Arithmetic + Exp + Log + Sin + Cos + Tan + Sqrt + Cbrt + Square")
        {
            // [+, -, *, /, exp, log]
            size_t k = 0;
            grammar.SetConfig(Grammar::Full);
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 
            model.start();
            BENCHMARK("Sequential")
            {
                ++k;
                std::transform(std::execution::unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
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
        }
    }
}
