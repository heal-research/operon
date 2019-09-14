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
#include "analyzers/diversity.hpp"
#include "algorithms/config.hpp"

namespace Operon 
{
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
        auto t0 = std::chrono::high_resolution_clock::now();

        auto& grammar      = problem.GetGrammar();
        auto& dataset      = problem.GetDataset();
        auto target        = problem.TargetVariable();

        auto testRange     = problem.TestRange();
        auto targetValues  = dataset.GetValues(target);
        auto targetTest    = targetValues.subspan(testRange.Start, testRange.Size());

        using Ind = typename TRecombinator::SelectorType::SelectableType;
        constexpr bool Idx = TRecombinator::SelectorType::SelectableIndex;
        constexpr bool Max = TRecombinator::SelectorType::Maximization;

        PopulationDiversityAnalyzer<Ind> diversityAnalyzer;

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
            auto fitness = evaluator(rndlocal, ind);
            ind.Fitness[Idx] = ceres::IsFinite(fitness) ? fitness : worst;
        };

        // perform evaluation
        std::for_each(std::execution::par_unseq, parents.begin(), parents.end(), evaluate);

        for (size_t gen = 0; gen < config.Generations; ++gen)
        {
            // get some new seeds
            std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

#ifdef _MSC_VER 
            auto avgLength = std::reduce(parents.begin(), parents.end(), 0UL, [](size_t partial, const auto& p) { return partial + p.Genotype.Length(); }) / static_cast<double>(config.PopulationSize);
            auto avgQuality = std::reduce(parents.begin(), parents.end(), 0., [=](size_t partial, const auto& p) { return partial + p.Fitness[Idx]; }) / static_cast<double>(config.PopulationSize);
#else
            auto avgLength = std::transform_reduce(std::execution::par_unseq, parents.begin(), parents.end(), 0UL, std::plus<size_t>{}, [](Ind& p) { return p.Genotype.Length();} ) / static_cast<double>(config.PopulationSize);
            auto avgQuality = std::transform_reduce(std::execution::par_unseq, parents.begin(), parents.end(), 0.0, std::plus<double>{}, [=](Ind& p) { return p.Fitness[Idx];} ) / static_cast<double>(config.PopulationSize);
#endif
            avgQuality = std::clamp(avgQuality, 0.0, 1.0);

            // preserve one elite
            auto [minElem, maxElem] = std::minmax_element(parents.begin(), parents.end(), [&](const Ind& lhs, const Ind& rhs) { return lhs.Fitness[Idx] < rhs.Fitness[Idx]; });
            auto best = Max ? maxElem : minElem;
            double errorTrain = std::clamp(best->Fitness[Idx], 0.0, 1.0);
            auto estimatedTest = Evaluate<double>(best->Genotype, dataset, testRange);
            double errorTest  = std::clamp(RSquared(estimatedTest.begin(), estimatedTest.end(), targetTest.begin()), 0.0, 1.0);
            auto t1 = std::chrono::high_resolution_clock::now();

            //diversityAnalyzer.Prepare(parents);
            //auto hybridDiversity = diversityAnalyzer.HybridDiversity();
            //auto structDiversity = diversityAnalyzer.StructuralDiversity();
            auto hybridDiversity = 0.0;
            auto structDiversity = 0.0;

            if ((Max && std::abs(1 - best->Fitness[Idx]) < 1e-6) || (!Max && std::abs(best->Fitness[Idx]) < 1e-6))
            {
                terminate = true;
            }

            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;
            fmt::print("{:#3.3f}\t{}\t{:.1f}\t{:.3f}\t{:.3f}\t{:.4f}\t{:.1f}\t{}\t{}\t{}\t{:.4f}\t{:.4f}\n", elapsed, gen+1, avgLength, hybridDiversity, structDiversity, avgQuality, recombinator.SelectionPressure(), evaluator.FitnessEvaluations(), evaluator.LocalEvaluations(), evaluator.TotalEvaluations(), errorTrain, errorTest);

            if (terminate)
            {
                return;
            }

            offspring[0] = *best;
            recombinator.Prepare(parents);
            
            // produce some offspring
            auto iterate = [&](gsl::index i) 
            {
                rndlocal.Seed(seeds[i]);

                while (!(terminate = recombinator.Terminate())) {
                    auto recombinant  = recombinator(rndlocal, config.CrossoverProbability, config.MutationProbability);

                    if (recombinant.has_value())
                    {
                        offspring[i]  = std::move(recombinant.value());
                        return;
                    }
                }
            };
            std::for_each(std::execution::par_unseq, indices.cbegin() + 1, indices.cend(), iterate);
            // we check for empty offspring (in case of early termination due to selection pressure or stuff)
            // and fill them with the parents
            for(auto i : indices)
            {
                if (offspring[i].Genotype.Nodes().empty())
                {
                    offspring[i] = parents[i];
                }
            }
            // the offspring become the parents
            parents.swap(offspring);
        }
    }
}
#endif

