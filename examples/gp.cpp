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

#include "algorithms/gp.hpp"
#include "operators/evaluator.hpp"
#include "operators/generator.hpp"
#include "operators/selection.hpp"
#include "operators/reinserter/replaceworst.hpp"

using namespace Operon;

int main(int, char**) {
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
    Problem problem(ds);
    problem.Inputs(ds.Variables()).Target(target).TrainingRange(trainingRange).TestRange(testRange);
    problem.GetPrimitiveSet().SetConfig(PrimitiveSet::Arithmetic);

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
    RSquaredEvaluator evaluator(problem);
    evaluator.SetLocalOptimizationIterations(config.Iterations);
    evaluator.SetBudget(config.Evaluations);

    auto comp = [](Individual const& lhs, Individual const& rhs) { 
        return lhs[0] < rhs[0]; 
    };
    TournamentSelector selector(comp);
    selector.SetTournamentSize(5); 

    ReplaceWorstReinserter<> reinserter(comp);
    BasicOffspringGenerator generator(evaluator, crossover, mutation, selector, selector);

    // set up a genetic programming algorithm
    GeneticProgrammingAlgorithm gp(problem, config, initializer, generator, reinserter); 
    RandomGenerator random(config.Seed);

    int generation = 0;
    auto report = [&] { fmt::print("{}\n", ++generation); };
    gp.Run(random, report);

    return 0;
}
