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

int main(int argc, char** argv) {
    Operon::GeneticAlgorithmConfig config;
    config.Generations          = 100;
    config.PopulationSize       = 1000;
    config.PoolSize             = 1000;
    config.Evaluations          = 1000000;
    config.Iterations           = 0;
    config.CrossoverProbability = 1.0;
    config.MutationProbability  = 0.25;
    config.Seed                 = 42;

    Operon::Dataset ds("../data/Poly-10.csv", /* csv has header */ true);
    const std::string target = "Y";

    Operon::Range trainingRange { 0, ds.Rows() / 2 };
    Operon::Range testRange     { ds.Rows() / 2, ds.Rows() };
    Operon::Problem problem(ds, ds.Variables(), target, trainingRange, testRange);
    problem.GetGrammar().SetConfig(Operon::Grammar::Arithmetic);

    using Ind        = Operon::Individual<1>; // an individual holding one fitness value
    using Evaluator  = Operon::RSquaredEvaluator<Ind>;
    using Selector   = Operon::TournamentSelector<Ind, 0>;
    using Reinserter = Operon::ReplaceWorstReinserter<Ind, 0>;
    using Crossover  = Operon::SubtreeCrossover;
    using Mutation   = Operon::MultiMutation;
    using Generator  = Operon::BasicOffspringGenerator<Evaluator, Crossover, Mutation, Selector, Selector>;

    // set up the solution creator 
    size_t maxTreeDepth  = 10;
    size_t maxTreeLength = 50;
    std::uniform_int_distribution<size_t> treeSizeDistribution(1, maxTreeLength);
    Operon::BalancedTreeCreator creator { treeSizeDistribution, maxTreeDepth, maxTreeLength };

    // set up crossover and mutation
    Operon::SubtreeCrossover crossover { /* internal node bias */ 0.9, maxTreeDepth, maxTreeLength };
    Operon::MultiMutation mutation;
    Operon::OnePointMutation onePoint;
    Operon::ChangeVariableMutation changeVar { problem.InputVariables() };
    Operon::ChangeFunctionMutation changeFunc { problem.GetGrammar() };
    mutation.Add(onePoint, 1.0);
    mutation.Add(changeVar, 1.0);
    mutation.Add(changeFunc, 1.0);

    // set up remaining operators
    Evaluator evaluator(problem);
    evaluator.LocalOptimizationIterations(config.Iterations);
    evaluator.Budget(config.Evaluations);

    Selector selector(/* tournament size */ 5);
    Reinserter reinserter;
    Generator generator(evaluator, crossover, mutation, selector, selector);

    // set up a genetic programming algorithm
    Operon::GeneticProgrammingAlgorithm gp(problem, config, creator, generator, reinserter); 
    Operon::Random random(config.Seed);

    int generation = 0;
    auto report = [&] { fmt::print("{}\n", ++generation); };
    gp.Run(random, report);
}
