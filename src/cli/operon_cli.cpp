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
    Operon::Random::JsfRand<64> jsf;

    const auto config = GeneticAlgorithmConfig
    {
        100,         // Generations
        (size_t)1e6, // Evaluations
        10,          // Const opt iterations
        100000,        // Population size
        1.0,         // Crossover probability,
        0.25,        // Mutation probability,
        true         // Maximization
    };
    
    using Ind              = Individual<1>; // 1 single objective
    const size_t tSize     = 5;
    const size_t maxDepth  = 7;
    const size_t maxLength = 25;

    auto selector          = TournamentSelector<Ind>(tSize, config.Maximization);
    auto creator           = GrowTreeCreator(maxDepth, maxLength);
    auto crossover         = SubtreeCrossover(0.9, maxDepth, maxLength);
    auto mutator           = OnePointMutation();

    auto dataset           = Dataset("poly10.csv", true);
    auto inputs            = dataset.VariableNames();
    auto target            = "Y";
    inputs.erase(std::remove_if(inputs.begin(), inputs.end(), [&](const std::string& s) { return s == target; }), inputs.end());

    auto trainingRange     = Range { 0, 250 };
    auto testRange         = Range { 250, 500 };
    const auto problem     = Problem(dataset, inputs, target, trainingRange, testRange);
    GeneticAlgorithm<GrowTreeCreator, TournamentSelector<Ind>, SubtreeCrossover, OnePointMutation>(jsf, problem, config, creator, selector, crossover, mutator);

    return 0;
}

