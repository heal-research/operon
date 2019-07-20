#ifndef SGP_HPP
#define SGP_HPP

#include <fmt/core.h>
#include <execution>

#include "operator.hpp"
#include "problem.hpp"
#include "eval.hpp"
#include "stats.hpp"
#include "format.hpp"

namespace Operon 
{
    struct GeneticAlgorithmConfig
    {
        size_t Generations;
        size_t Evaluations;
        size_t Iterations;
        size_t PopulationSize;
        double CrossoverProbability;
        double MutationProbability;
        bool   Maximization;
    };

    template<typename Creator, typename Selector, typename Crossover, typename Mutator>
    void GeneticAlgorithm(RandomDevice& random, const Problem& problem, const GeneticAlgorithmConfig config, Creator& creator, Selector& selector, Crossover& crossover, Mutator& mutator)
    {
        auto& grammar      = problem.GetGrammar();
        auto& dataset      = problem.GetDataset();
        auto target        = problem.TargetVariable();

        auto trainingRange = problem.TrainingRange();
        auto targetValues  = dataset.GetValues(target);

        using Ind          = typename Selector::TSelectable;
        auto idx           = Selector::Index;
        auto variables     = dataset.Variables();
        variables.erase(std::remove_if(variables.begin(), variables.end(), [&](const auto& v) { return v.Name == target; }), variables.end());

        std::vector<Ind>    parents(config.PopulationSize);
        std::vector<Ind>    offspring(config.PopulationSize);

        std::generate(parents.begin(), parents.end(), [&]() { return Ind { creator(random, grammar, variables), 0.0 }; });
        std::uniform_real_distribution<double> uniformReal(0, 1); // for crossover and mutation

        for (size_t gen = 0; gen < config.Generations; ++gen)
        {
            auto evaluate = [&](auto& p) 
            {
                if (config.Iterations > 0)
                {
                    OptimizeAutodiff(p.Genotype, dataset, targetValues, trainingRange, config.Iterations);
                }
                auto estimated = Evaluate<double>(p.Genotype, dataset, trainingRange);
                p.Fitness[idx] = RSquared(estimated.begin(), estimated.end(), targetValues.begin() + trainingRange.Start);
            };

            // perform evaluation
            std::for_each(std::execution::par_unseq, parents.begin(), parents.end(), evaluate);

            // prepare offspring 
            offspring.clear();
            offspring.resize(config.PopulationSize);

            // preserve one elite
            auto comp = [&](const auto& lhs, const auto& rhs) { return lhs.Fitness[idx] < rhs.Fitness[idx]; };
            auto best = config.Maximization
                ? std::max_element(parents.begin(), parents.end(), comp)
                : std::min_element(parents.begin(), parents.end(), comp);
            offspring[0] = *best;

            fmt::print("Generation {}: {} {}\n", gen+1, best->Fitness[idx], InfixFormatter::Format(best->Genotype, dataset));

            selector.Reset(parents); // apply selector on current parents

            // produce some offspring
            auto iterate = [&](auto& ind) 
            {
                auto first = selector(random);
                Tree child;

                // crossover 
                if (uniformReal(random) < config.CrossoverProbability)
                {
                    auto second = selector(random);
                    child = crossover(random, parents[first].Genotype, parents[second].Genotype);
                }
                // mutation
                if (uniformReal(random) < config.MutationProbability)
                {
                    child = child.Length() > 0 ? mutator(random, child) : mutator(random, parents[first].Genotype);
                }
                ind.Genotype = std::move(child);
            };
            std::for_each(std::execution::par_unseq, offspring.begin() + 1, offspring.end(), iterate);
            // the offspring become the parents
            parents.swap(offspring);
        }
    }
}
#endif

