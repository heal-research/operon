#ifndef SGP_HPP
#define SGP_HPP

#include <fmt/core.h>
#include <execution>
#include <thread>
#include <chrono>

#include "core/operator.hpp"
#include "core/problem.hpp"
#include "core/eval.hpp"
#include "core/stats.hpp"
#include "core/format.hpp"

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

        auto variables = dataset.Variables();
        std::vector<Variable> inputs;
        std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto& v) { return v.Name != target; });

        std::vector<Ind>    parents(config.PopulationSize);
        std::vector<Ind>    offspring(config.PopulationSize);

        std::vector<gsl::index> indices(config.PopulationSize);
        std::iota(indices.begin(), indices.end(), 0L);
        std::vector<RandomDevice::result_type> seeds(config.PopulationSize);
        std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

        thread_local RandomDevice rndlocal = random;

        auto create = [&](gsl::index i)
        {
            // create one random generator per thread
            //auto rndlocal = RandomDevice(seeds[i]);
            rndlocal.Seed(seeds[i]);
            parents[i].Genotype = creator(rndlocal, grammar, inputs);
        };

        //std::generate(std::execution::par_unseq, parents.begin(), parents.end(), create);
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), create);
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
            // get some new seeds
            std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

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
            auto iterate = [&](gsl::index i) 
            {
                rndlocal.Seed(seeds[i]);
                auto first = selector(rndlocal);
                Tree child;

                // crossover 
                if (uniformReal(rndlocal) < config.CrossoverProbability)
                {
                    auto second = selector(rndlocal);
                    child = crossover(rndlocal, parents[first].Genotype, parents[second].Genotype);
                }
                // mutation
                if (uniformReal(rndlocal) < config.MutationProbability)
                {
                    child = child.Length() > 0 ? mutator(rndlocal, child) : mutator(rndlocal, parents[first].Genotype);
                }
                offspring[i].Genotype = child.Nodes().empty() ? parents[first].Genotype : std::move(child);
            };
            std::for_each(std::execution::par_unseq, indices.cbegin() + 1, indices.cend(), iterate);

            // the offspring become the parents
            parents.swap(offspring);
        }
    }
}
#endif

