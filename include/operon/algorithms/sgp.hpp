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
    // - ExecutionPolicy (par, par_unseq)
    // - InitializationPolicy
    // - ParentSelectionPolicy
    // - OffspringSelectionPolicy
    // - RecombinationPolicy (it can interact with the selection policy; how to handle both crossover and mutation?)
    // - some policy/distinction between single- and multi-objective?
    // - we should not pass operators as parameters (should be instantiated/handled by the respective policy)
    template<typename TCreator, typename TSelector, typename TCrossover, typename TMutator>
    void GeneticAlgorithm(operon::rand_t& random, const Problem& problem, const GeneticAlgorithmConfig config, TCreator& creator, TSelector& selector, TCrossover& crossover, TMutator& mutator)
    {
        auto& grammar      = problem.GetGrammar();
        auto& dataset      = problem.GetDataset();
        auto target        = problem.TargetVariable();

        auto trainingRange = problem.TrainingRange();
        auto testRange     = problem.TestRange();
        auto targetValues  = dataset.GetValues(target);
        auto targetTrain   = targetValues.subspan(trainingRange.Start, trainingRange.Size());
        auto targetTest    = targetValues.subspan(testRange.Start, testRange.Size());

        const auto& inputs = problem.InputVariables();

        using Ind                = typename TSelector::SelectableType;
        constexpr gsl::index Idx = TSelector::SelectableIndex;
        constexpr bool Max       = TSelector::Maximization;

        std::vector<Ind> parents(config.PopulationSize);
        std::vector<Ind> offspring(config.PopulationSize);

        std::vector<gsl::index> indices(config.PopulationSize);
        std::iota(indices.begin(), indices.end(), 0L);

        std::vector<operon::rand_t::result_type> seeds(config.PopulationSize);
        std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

        thread_local operon::rand_t rndlocal = random;
        auto worst = Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max();

        auto create = [&](gsl::index i)
        {
            // create one random generator per thread
            rndlocal.Seed(seeds[i]);
            parents[i].Genotype     = creator(rndlocal, grammar, inputs);
            parents[i].Fitness[Idx] = worst;
        };

        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), create);
        std::uniform_real_distribution<double> uniformReal(0, 1); // for crossover and mutation

        std::atomic_ulong evaluated      = 0UL;
        std::atomic_ulong evaluatedLocal = 0UL;
        std::atomic_bool terminate       = false;

        auto evaluate = [&](std::vector<Ind>& individuals, gsl::index idx) 
        {
            if (terminate)
            {
                return;
            }
            auto& ind = individuals[idx];
            if (config.Iterations > 0)
            {
                auto summary = OptimizeAutodiff(ind.Genotype, dataset, targetTrain, trainingRange, config.Iterations);
                evaluatedLocal += summary.num_successful_steps + summary.num_unsuccessful_steps;
            }
            auto estimated   = Evaluate<double>(ind.Genotype, dataset, trainingRange);
            ++evaluated;
            auto fitness     = 1 - NormalizedMeanSquaredError(estimated.begin(), estimated.end(), targetTrain.begin());
            ind.Fitness[Idx] = ceres::IsFinite(fitness) ? fitness : worst;

            if (evaluated + evaluatedLocal > config.Evaluations)
            {
                terminate = true;   
            }
        };

        for (size_t gen = 0UL; gen < config.Generations && !terminate; ++gen)
        {
            // get some new seeds
            std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

            // perform evaluation
            std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](gsl::index i) { evaluate(parents, i); });

            // preserve one elite
            auto [minElem, maxElem] = std::minmax_element(parents.begin(), parents.end(), [](const Ind& lhs, const Ind& rhs) { return lhs.Fitness[Idx] < rhs.Fitness[Idx]; });
            auto best = Max ? maxElem : minElem;

            auto sum = std::transform_reduce(std::execution::par_unseq, parents.begin(), parents.end(), 0UL, [&](size_t lhs, size_t rhs) { return lhs + rhs; }, [&](Ind& p) { return p.Genotype.Length();} );

            auto& bestTree = best->Genotype;
            bestTree.Reduce(); // makes it a little nicer to visualize

            auto estimatedTest = Evaluate<double>(best->Genotype, dataset, testRange);
            auto nmseTest  = NormalizedMeanSquaredError(estimatedTest.begin(), estimatedTest.end(), targetTest.begin());
            fmt::print("{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\n", gen, (double)sum / config.PopulationSize, evaluated, evaluatedLocal, best->Fitness[Idx], 1 - nmseTest);

            offspring[0] = *best;
            selector.Prepare(parents); // apply selector on current parents

            // produce some offspring
            auto iterate = [&](gsl::index i) 
            {
                rndlocal.Seed(seeds[i]);
                auto first = selector(rndlocal);
                std::optional<Tree> child;

                // crossover 
                if (uniformReal(rndlocal) < config.CrossoverProbability)
                {
                    auto second = selector(rndlocal);
                    child = crossover(rndlocal, parents[first].Genotype, parents[second].Genotype);
                }
                if (!child.has_value())
                {
                    auto tmp = parents[first].Genotype;
                    child = tmp;
                }
                // mutation
                if (uniformReal(rndlocal) < config.MutationProbability)
                {
                    mutator(rndlocal, child.value());
                }
                offspring[i].Genotype = std::move(child.value());
            };
            std::for_each(std::execution::par_unseq, indices.cbegin() + 1, indices.cend(), iterate);

            // the offspring become the parents
            parents.swap(offspring);
        }
    }
}
#endif

