#include <doctest/doctest.h>

#include "algorithms/gp.hpp"
#include "operators/evaluator.hpp"
#include "operators/generator.hpp"
#include "operators/selection.hpp"
#include "operators/reinserter/replaceworst.hpp"

#include "nanobench.h"

#include <tbb/task_scheduler_init.h>
#include <tbb/global_control.h>

using namespace Operon;

TEST_CASE("Evolution speed") {
    GeneticAlgorithmConfig config;
    config.Generations          = 100;
    config.PopulationSize       = 1000;
    config.PoolSize             = 1000;
    config.Evaluations          = 1000000;
    config.Iterations           = 0;
    config.CrossoverProbability = 1.0;
    config.MutationProbability  = 0.25;
    config.Seed                 = 42;

    Dataset ds("../data/Poly-10.csv", /* csv has header */ true);
    const std::string target = "Y";

    Range trainingRange { 0, ds.Rows() / 2 };
    Range testRange     { ds.Rows() / 2, ds.Rows() };
    Problem problem(ds, ds.Variables(), target, trainingRange, testRange);
    problem.GetGrammar().SetConfig(Grammar::Arithmetic);


    using Ind        = Individual; // an individual holding one fitness value
    using Evaluator  = RSquaredEvaluator;
    using Selector   = TournamentSelector;
    using Reinserter = ReplaceWorstReinserter<>;
    using Crossover  = SubtreeCrossover;
    using Mutation   = MultiMutation;
    //using Generator  = BasicOffspringGenerator<Evaluator, Crossover, Mutation, Selector, Selector>;
    using Generator  = OffspringSelectionGenerator;

    // set up the solution creator 
    size_t maxTreeDepth  = 10;
    size_t maxTreeLength = 50;
    std::uniform_int_distribution<size_t> treeSizeDistribution(1, maxTreeLength);
    BalancedTreeCreator creator { problem.GetGrammar(), problem.InputVariables() };

    Initializer initializer { creator, treeSizeDistribution };

    // set up crossover and mutation
    double internalNodeBias = 0.9;
    SubtreeCrossover crossover { internalNodeBias, maxTreeDepth, maxTreeLength };
    MultiMutation mutation;
    OnePointMutation onePoint;
    ChangeVariableMutation changeVar { problem.InputVariables() };
    ChangeFunctionMutation changeFunc { problem.GetGrammar() };
    mutation.Add(onePoint, 1.0);
    mutation.Add(changeVar, 1.0);
    mutation.Add(changeFunc, 1.0);

    // set up remaining operators

    auto comparison = [](gsl::span<const Individual> pop, gsl::index i, gsl::index j) { return pop[i][0] < pop[j][0]; };
    Selector selector(comparison);
    Reinserter reinserter;

    // set up a genetic programming algorithm
    Random random(config.Seed);

    ankerl::nanobench::Bench b;
    b.performanceCounters(true);

    tbb::global_control c(tbb::global_control::max_allowed_parallelism, 1);

    for (size_t i = 100; i <= 10000; i += 100) {
        config.PopulationSize = i;
        config.PoolSize = i;
        config.Evaluations = config.Generations * i;
        config.Seed = random();

        Evaluator evaluator(problem);
        evaluator.LocalOptimizationIterations(config.Iterations);
        evaluator.Budget(config.Evaluations);
        Generator generator(evaluator, crossover, mutation, selector, selector);

        GeneticProgrammingAlgorithm gp(problem, config, initializer, generator, reinserter); 

        b.complexityN(i).run("GP", [&]() { gp.Run(random, nullptr); });
    }

    std::cout << "GP complexity: " << b.complexityBigO() << std::endl;
}

