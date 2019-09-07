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
    template<typename TCreator, typename TRecombinator>
    void OffspringSelectionGeneticAlgorithm(operon::rand_t& random, const Problem& problem, const OffspringSelectionGeneticAlgorithmConfig config, TCreator& creator, TRecombinator& recombinator)
    {
        auto& grammar      = problem.GetGrammar();
        auto& dataset      = problem.GetDataset();
        auto target        = problem.TargetVariable();

        auto testRange     = problem.TestRange();
        auto targetValues  = dataset.GetValues(target);
        auto targetTest    = targetValues.subspan(testRange.Start, testRange.Size());

        using Ind = typename TRecombinator::SelectorType::SelectableType;
        constexpr bool Idx = TRecombinator::SelectorType::SelectableIndex;
        constexpr bool Max = TRecombinator::SelectorType::Maximization;

        // we run with two populations which get swapped with each other
        std::vector<Ind> parents(config.PopulationSize);   // parent population
        std::vector<Ind> offspring(config.PopulationSize); // offspring population

        // easier to work with indices 
        std::vector<gsl::index> indices(config.PopulationSize);
        std::iota(indices.begin(), indices.end(), 0L);
        // random seeds for each thread
        std::vector<operon::rand_t::result_type> seeds(config.PopulationSize);
        std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

        // flag to signal algorithm termination
        std::atomic_bool terminate = false;

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
        auto& evaluator = recombinator.Evaluator();

        auto evaluate = [&](Ind& ind) 
        {
            auto fitness = evaluator(random, ind, config.Iterations);
            ind.Fitness[Idx] = ceres::IsFinite(fitness) ? fitness : worst;
        };

        // perform evaluation
        std::for_each(std::execution::par_unseq, parents.begin(), parents.end(), evaluate);

        double selectionPressure = 0;
    
        for (size_t gen = 0; gen < config.Generations; ++gen)
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
            fmt::print("{}\t{}\t{}\t{}\t{}\t{:.6f}\t{:.6f}\n", gen+1, (double)sum / config.PopulationSize, selectionPressure, evaluator.TotalEvaluations(), evaluator.LocalEvaluations(), 1 - best->Fitness[Idx], 1 - nmseTest);

            if (terminate)
            {
                return;
            }

            offspring[0] = *best;
            recombinator.Prepare(parents);

            auto lastEvaluations = evaluator.TotalEvaluations();
            
            // produce some offspring
            auto iterate = [&](gsl::index i) 
            {
                if (terminate) 
                {
                    return;
                }

                rndlocal.Seed(seeds[i]);

                do {
                    auto recombinant  = recombinator(rndlocal, config.CrossoverProbability, config.MutationProbability);
                    auto evaluations  = evaluator.TotalEvaluations();
                    selectionPressure = static_cast<double>(evaluations - lastEvaluations) / config.PopulationSize;

                    if (evaluations > config.Evaluations || selectionPressure > config.MaxSelectionPressure)
                    {
                        terminate = true;
                    }

                    if (recombinant.has_value())
                    {
                        offspring[i]  = std::move(recombinant.value());
                        return;
                    }
                }
                while (!terminate);
            };
            std::for_each(std::execution::par_unseq, indices.cbegin() + 1, indices.cend(), iterate);

            // the offspring become the parents
            parents.swap(offspring);
        }
    }
}
#endif

