#ifndef OSGP_HPP
#define OSGP_HPP

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
    struct OffspringSelectionGeneticAlgorithmConfig
    {
        size_t Generations;
        size_t Evaluations;
        size_t Iterations;
        size_t PopulationSize;
        double CrossoverProbability;
        double MutationProbability;
        size_t MaxSelectionPressure;
    };

    // this should be designed such that it has:
    // - ExecutionPolicy (par, seq)
    // - InitializationPolicy
    // - ParentSelectionPolicy
    // - OffspringSelectionPolicy
    // - RecombinationPolicy (it can interact with the selection policy; how to handle both crossover and mutation?)
    // - some policy/distinction between single- and multi-objective?
    // - we should not pass operators as parameters (should be instantiated/handled by the respective policy)
    template<typename Ind, size_t Idx, bool Max>
    void OffspringSelectionGeneticAlgorithm(operon::rand_t& random, const Problem& problem, const OffspringSelectionGeneticAlgorithmConfig config, CreatorBase& creator, SelectorBase<Ind, Idx, Max>& selector, CrossoverBase& crossover, MutatorBase& mutator)
    {
        fmt::print("max sel pressure: {}\n", config.MaxSelectionPressure);
        auto& grammar      = problem.GetGrammar();
        auto& dataset      = problem.GetDataset();
        auto target        = problem.TargetVariable();

        auto trainingRange = problem.TrainingRange();
        auto testRange     = problem.TestRange();
        auto targetValues  = dataset.GetValues(target);
        auto targetTrain   = targetValues.subspan(trainingRange.Start, trainingRange.Size());
        auto targetTest    = targetValues.subspan(testRange.Start, testRange.Size());

        // we run with two populations which get swapped with each other
        std::vector<Ind> parents(config.PopulationSize);   // parent population
        std::vector<Ind> offspring(config.PopulationSize); // offspring population

        // easier to work with indices 
        std::vector<gsl::index> indices(config.PopulationSize);
        std::iota(indices.begin(), indices.end(), 0L);
        // random seeds for each thread
        std::vector<operon::rand_t::result_type> seeds(config.PopulationSize);
        std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });
        // keep track of local search effort
        std::vector<gsl::index> localSearchEvaluations(config.PopulationSize);

        const auto& inputs = problem.InputVariables();

        thread_local operon::rand_t rndlocal = random;

        const auto worst = Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max();
        auto create = [&](gsl::index i)
        {
            // create one random generator per thread
            rndlocal.Seed(seeds[i]);
            parents[i].Genotype     = creator(rndlocal, grammar, inputs);
            parents[i].Fitness[Idx] = worst;
        };

        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), create);
        std::uniform_real_distribution<double> uniformReal(0, 1); // for crossover and mutation

        auto evaluate = [&](Ind& ind, gsl::index i) 
        {
            if (config.Iterations > 0)
            {
                auto summary = OptimizeAutodiff(ind.Genotype, dataset, targetTrain, trainingRange, config.Iterations);
                localSearchEvaluations[i] += summary.num_successful_steps + summary.num_unsuccessful_steps;
            }
            auto estimated   = Evaluate<double>(ind.Genotype, dataset, trainingRange);
            auto fitness     = 1 - NormalizedMeanSquaredError(estimated.begin(), estimated.end(), targetTrain.begin());
            ind.Fitness[Idx] = ceres::IsFinite(fitness) ? fitness : worst;
        };

        // perform evaluation
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](gsl::index i) { evaluate(parents[i], i); });
    
        std::atomic_ulong evaluated;
        std::atomic_bool terminate = false;
        double selectionPressure = 0;

        for (size_t gen = 0, evaluations = parents.size(); gen < config.Generations; ++gen)
        {
            // get some new seeds
            std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

            // preserve one elite
            auto [minElem, maxElem] = std::minmax_element(parents.begin(), parents.end(), [](const Ind& lhs, const Ind& rhs) { return lhs.Fitness[Idx] < rhs.Fitness[Idx]; });
            auto best = Max ? maxElem : minElem;

            auto sum = std::transform_reduce(std::execution::par_unseq, parents.begin(), parents.end(), 0UL, [&](size_t lhs, size_t rhs) { return lhs + rhs; }, [&](Ind& p) { return p.Genotype.Length();} );

            auto& bestTree = best->Genotype;
            bestTree.Reduce(); // makes it a little nicer to visualize

            auto estimatedTest = Evaluate<double>(best->Genotype, dataset, testRange);
            auto nmseTest  = NormalizedMeanSquaredError(estimatedTest.begin(), estimatedTest.end(), targetTest.begin());
            fmt::print("Generation {}: {} {} {} {:.6f} {:.6f} {}\n", gen+1, (double)sum / config.PopulationSize, selectionPressure, evaluations, best->Fitness[Idx], 1 - nmseTest, InfixFormatter::Format(bestTree, dataset, 6));
            if (terminate)
            {
                return;
            }

            offspring[0] = *best;
            selector.Reset(parents); // apply selector on current parents

            evaluated = 0;

            auto fitness = [&](gsl::index i) { return parents[i].Fitness[Idx]; };

            std::fill(localSearchEvaluations.begin(), localSearchEvaluations.end(), 0L);

            // produce some offspring
            auto iterate = [&](gsl::index i) 
            {
                if (terminate) 
                {
                    return;
                }

                rndlocal.Seed(seeds[i]);

                do {
                    auto first = selector(rndlocal);
                    std::optional<Tree> child;

                    double f = fitness(first);

                    // crossover 
                    if (uniformReal(rndlocal) < config.CrossoverProbability)
                    {
                        auto second = selector(rndlocal);
                        child = crossover(rndlocal, parents[first].Genotype, parents[second].Genotype);
                        f = Max ? std::max(fitness(first), fitness(second)) 
                                : std::min(fitness(first), fitness(second));
                    }
                    // make sure we have an offspring individual 
                    if (!child.has_value())
                    {
                        auto tmp = parents[first].Genotype; // make a copy
                        child = tmp; 
                    }

                    // mutation
                    if (uniformReal(rndlocal) < config.MutationProbability)
                    {
                        mutator(rndlocal, child.value()); // mutate in place
                    }

                    auto ind = Ind { std::move(child.value()), worst };

                    evaluate(ind, i);
                    ++evaluated;
                    if ((Max && ind.Fitness[Idx] > f) || (!Max && ind.Fitness[Idx] < f))
                    {
                        offspring[i] = std::move(ind);
                        return;
                    }
                }
                while (!terminate && evaluated < config.PopulationSize * config.MaxSelectionPressure && evaluated + evaluations < config.Evaluations);
                // if this point is reached we should terminate the algorithm
                terminate = true;
            };
            std::for_each(std::execution::par_unseq, indices.cbegin() + 1, indices.cend(), iterate);
            selectionPressure = static_cast<double>(evaluated) / config.PopulationSize;
            if (selectionPressure > config.MaxSelectionPressure)
            {
                terminate = true;
            }
            evaluations += evaluated;
            evaluations += std::reduce(std::execution::par_unseq, localSearchEvaluations.begin(), localSearchEvaluations.end(), 0L, std::plus<gsl::index>{});

            // the offspring become the parents
            parents.swap(offspring);
        }
    }
}
#endif

