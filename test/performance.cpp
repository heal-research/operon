#include <catch2/catch.hpp>
#include <execution>

#include "core/eval.hpp"
#include "core/jsf.hpp"
#include "core/grammar.hpp"
#include "core/stats.hpp"
#include "operators/initialization.hpp"

namespace Operon::Test
{
    TEST_CASE("Evaluation performance arithmetic grammar", "[performance]")
    {
        size_t n = 10000;
        size_t maxLength = 50;
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
        grammar.SetConfig(Grammar::Arithmetic);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 

        auto evaluate = [&](auto& tree)
        {
            auto estimated = Evaluate<double>(tree, ds, range); 
            auto r2 = RSquared(estimated.begin(), estimated.end(), targetValues.begin() + range.Start);
            return r2;
        };


        BENCHMARK("Arithmetic grammar")
        {
            std::transform(std::execution::par_unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
        };

        grammar.SetConfig(Grammar::TypeCoherent);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 

        BENCHMARK("Typecoherent grammar")
        {
            std::transform(std::execution::par_unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
        };

        grammar.SetConfig(Grammar::Full);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, grammar, inputs); }); 

        BENCHMARK("Full grammar")
        {
            std::transform(std::execution::par_unseq, trees.begin(), trees.end(), fit.begin(), evaluate);
        };
    }
}
