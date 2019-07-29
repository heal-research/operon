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
    };

    // this should be designed such that it has:
    // - ExecutionPolicy (parallel, sequential)
    // - InitializationPolicy
    // - ParentSelectionPolicy
    // - OffspringSelectionPolicy
    // - RecombinationPolicy (it can interact with the selection policy; how to handle both crossover and mutation?)
    // - some policy/distinction between single- and multi-objective?
    // - we should not pass operators as parameters (should be instantiated/handled by the respective policy)
    template<typename Ind, size_t Idx, bool Max>
    void GeneticAlgorithm(RandomDevice& random, const Problem& problem, const GeneticAlgorithmConfig config, CreatorBase& creator, SelectorBase<Ind, Idx, Max>& selector, CrossoverBase& crossover, MutatorBase& mutator)
    {
        auto& grammar      = problem.GetGrammar();
        auto& dataset      = problem.GetDataset();
        auto target        = problem.TargetVariable();

        auto trainingRange = problem.TrainingRange();
        auto targetValues  = dataset.GetValues(target);

        auto variables     = dataset.Variables();
        variables.erase(std::remove_if(variables.begin(), variables.end(), [&](const auto& v) { return v.Name == target; }), variables.end());

        std::vector<Ind>    parents(config.PopulationSize);
        std::vector<Ind>    offspring(config.PopulationSize);

        std::generate(std::execution::par_unseq, parents.begin(), parents.end(), [&]() { return Ind { creator(random, grammar, variables), 0.0 }; });
        std::uniform_real_distribution<double> uniformReal(0, 1); // for crossover and mutation

        auto evaluate = [&](auto& ind) 
        {
            if (config.Iterations > 0)
            {
                OptimizeAutodiff(ind.Genotype, dataset, targetValues, trainingRange, config.Iterations);
            }
            auto estimated   = Evaluate<double>(ind.Genotype, dataset, trainingRange);
            ind.Fitness[Idx] = RSquared(estimated.begin(), estimated.end(), targetValues.begin() + trainingRange.Start);
        };

        for (size_t gen = 0; gen < config.Generations; ++gen)
        {
            // perform evaluation
            std::for_each(std::execution::par_unseq, parents.begin(), parents.end(), evaluate);

            // prepare offspring 
            offspring.clear();
            offspring.resize(config.PopulationSize);

            // preserve one elite
            auto [minElem, maxElem] = std::minmax_element(parents.begin(), parents.end(), [](const Ind& lhs, const Ind& rhs) { return lhs.Fitness[Idx] < rhs.Fitness[Idx]; });
            auto best = Max ? maxElem : minElem;
            offspring[0] = *best;
            auto sum = std::transform_reduce(std::execution::par_unseq, parents.begin(), parents.end(), 0UL, [&](size_t lhs, size_t rhs) { return lhs + rhs; }, [&](Ind& p) { return p.Genotype.Length();} );

            fmt::print("Generation {}: {} {} {}\n", gen+1, (double)sum / config.PopulationSize, best->Fitness[Idx], InfixFormatter::Format(best->Genotype, dataset));

            if (1 - best->Fitness[Idx] < 1e-6)
            {
                break;
            }

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

