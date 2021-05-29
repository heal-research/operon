// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <doctest/doctest.h>

#include "algorithms/gp.hpp"
#include "operators/evaluator.hpp"
#include "operators/generator.hpp"
#include "operators/selection.hpp"
#include "operators/reinserter/replaceworst.hpp"

#include "nanobench.h"
#include "taskflow/taskflow.hpp"

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
    auto problem = Operon::Problem(ds).Inputs(ds.Variables()).Target(target).TrainingRange(trainingRange).TestRange(testRange);

    problem.GetPrimitiveSet().SetConfig(PrimitiveSet::Arithmetic);


    using Evaluator  = RSquaredEvaluator;
    using Selector   = TournamentSelector;
    using Reinserter = ReplaceWorstReinserter;
    //using Generator  = BasicOffspringGenerator<Evaluator, Crossover, Mutation, Selector, Selector>;
    using Generator  = OffspringSelectionGenerator;

    // set up the solution creator 
    size_t maxTreeDepth  = 10;
    size_t maxTreeLength = 50;
    std::uniform_int_distribution<size_t> treeSizeDistribution(1, maxTreeLength);
    BalancedTreeCreator creator { problem.GetPrimitiveSet(), problem.InputVariables() };

    Initializer initializer { creator, treeSizeDistribution };

    // set up crossover and mutation
    double internalNodeBias = 0.9;
    SubtreeCrossover crossover { internalNodeBias, maxTreeDepth, maxTreeLength };
    MultiMutation mutation;
    OnePointMutation onePoint;
    ChangeVariableMutation changeVar { problem.InputVariables() };
    ChangeFunctionMutation changeFunc { problem.GetPrimitiveSet() };
    mutation.Add(onePoint, 1.0);
    mutation.Add(changeVar, 1.0);
    mutation.Add(changeFunc, 1.0);

    // set up remaining operators

    auto comparison = [](Individual const& lhs, Individual const& rhs) { return lhs[0] < rhs[0]; };
    Selector selector(comparison);
    Reinserter reinserter(comparison);

    // set up a genetic programming algorithm
    RandomGenerator random(config.Seed);

    ankerl::nanobench::Bench b;
    b.performanceCounters(true);

    tf::Executor executor(1);

    for (size_t i = 100; i <= 10000; i += 100) {
        config.PopulationSize = i;
        config.PoolSize = i;
        config.Evaluations = config.Generations * i;
        config.Seed = random();

        Interpreter interpreter;
        Evaluator evaluator(problem, interpreter);
        evaluator.SetLocalOptimizationIterations(config.Iterations);
        evaluator.SetBudget(config.Evaluations);
        Generator generator(evaluator, crossover, mutation, selector, selector);

        GeneticProgrammingAlgorithm gp(problem, config, initializer, generator, reinserter); 

        b.complexityN(i).run("GP", [&]() { gp.Run(executor, random, nullptr); });
    }

    std::cout << "GP complexity: " << b.complexityBigO() << std::endl;
}

