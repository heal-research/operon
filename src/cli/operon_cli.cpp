#include <fmt/core.h>
#include <cxxopts.hpp>

#include "initialization.hpp"
#include "mutation.hpp"
#include "crossover.hpp"
#include "selection.hpp"
#include "sgp.hpp"

using namespace Operon;

int main(int argc, char* argv[])
{
    Operon::Random::JsfRand<64> random;

    const auto config        = GeneticAlgorithmConfig
    {
        100,         // Generations
        (size_t)1e6, // Evaluations
        0,          // Const opt iterations
        500000,        // Population size
        1.0,         // Crossover probability,
        0.25,        // Mutation probability,
    };
    const size_t maxDepth    = 7;
    const size_t maxLength   = 50;

    auto creator             = GrowTreeCreator(maxDepth, maxLength);
    auto crossover           = SubtreeCrossover(0.9, maxDepth, maxLength);
    auto mutator             = OnePointMutation();

    const auto dataset       = Dataset("../data/poly-10.csv", true);
    auto target              = "Y";
    auto inputs              = dataset.VariableNames();
    inputs.erase(std::remove_if(inputs.begin(), inputs.end(), [&](const std::string& s) { return s == target; }), inputs.end());

    const auto trainingRange = Range { 0, 250 };
    const auto testRange     = Range { 250, 500 };
    const auto problem       = Problem(dataset, inputs, target, trainingRange, testRange);

    const bool maximization  = true;
    const size_t idx         = 0;
    const size_t tSize       = 50;

    TournamentSelector<Individual<1>, idx, maximization> selector(tSize);
    GeneticAlgorithm(random, problem, config, creator, selector, crossover, mutator);

    return 0;
}

