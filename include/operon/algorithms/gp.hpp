/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
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

#ifndef GP_HPP
#define GP_HPP

#include "algorithms/config.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/metrics.hpp"
#include "operators/crossover.hpp"
#include "operators/creator.hpp"
#include "operators/initializer.hpp"
#include "operators/mutation.hpp"
#include "operators/generator.hpp"

namespace Operon {

template <typename TInitializer, typename ExecutionPolicy = std::execution::parallel_unsequenced_policy>
class GeneticProgrammingAlgorithm {
private:
    std::reference_wrapper<const Problem> problem_;
    std::reference_wrapper<const GeneticAlgorithmConfig> config_;

    std::reference_wrapper<const TInitializer> initializer_;
    std::reference_wrapper<const OffspringGeneratorBase> generator_;
    std::reference_wrapper<const ReinserterBase> reinserter_;

    std::vector<Individual> parents;
    std::vector<Individual> offspring;

    size_t generation;

public:
    explicit GeneticProgrammingAlgorithm(const Problem& problem, const GeneticAlgorithmConfig& config, const TInitializer& initializer, const OffspringGeneratorBase& generator, const ReinserterBase& reinserter)
        : problem_(problem)
        , config_(config)
        , initializer_(initializer)
        , generator_(generator)
        , reinserter_(reinserter)
        , parents(config.PopulationSize)
        , offspring(config.PoolSize)
        , generation(0UL)
    {
    }

    const gsl::span<const Individual> Parents() const { return gsl::span<const Individual>(parents); }
    const gsl::span<Individual> Parents() { return gsl::span<Individual>(parents); }
    const gsl::span<const Individual> Offspring() const { return gsl::span<const Individual>(offspring); }

    const Problem& GetProblem() const { return problem_.get(); }
    const GeneticAlgorithmConfig& GetConfig() const { return config_.get(); }

    const TInitializer& GetInitializer() const { return initializer_.get(); }
    const OffspringGeneratorBase& GetGenerator() const { return generator_.get(); }
    const ReinserterBase& GetReinserter() const { return reinserter_.get(); }

    size_t Generation() const { return generation; }

    void Reset()
    {
        generation = 0;
    }

    void Run(Operon::RandomGenerator& random, std::function<void()> report = nullptr)
    {
        auto& config       = GetConfig();
        auto& initializer  = GetInitializer();
        auto& generator    = GetGenerator();
        auto& reinserter   = GetReinserter();
        // easier to work with indices
        std::vector<gsl::index> indices(std::max(config.PopulationSize, config.PoolSize));
        std::iota(indices.begin(), indices.end(), 0L);
        // random seeds for each thread
        std::vector<Operon::RandomGenerator::result_type> seeds(config.PopulationSize);
        std::generate(seeds.begin(), seeds.end(), [&]() { return random(); });

        std::vector<size_t> treeLengths(config.PopulationSize);
        std::uniform_int_distribution<size_t> treeLengthDistribution(1, 50);

        auto idx = 0;

        auto create = [&](gsl::index i) {
            // create one random generator per thread
            Operon::RandomGenerator rndlocal{seeds[i]};
            parents[i].Genotype = initializer(rndlocal);
            parents[i][idx] = Operon::Numeric::Max<Operon::Scalar>();
        };
        const auto& evaluator = generator.Evaluator();
        auto evaluate = [&](Individual& ind) {
            auto f = evaluator(random, ind);
            if (!std::isfinite(f)) { f = Operon::Numeric::Max<Operon::Scalar>(); }
            ind[idx] = f;
        };

        // generate the initial population and perform evaluation
        ExecutionPolicy executionPolicy;
        std::for_each(executionPolicy, indices.begin(), indices.begin() + config.PopulationSize, create);
        std::for_each(executionPolicy, parents.begin(), parents.end(), evaluate);

        // flag to signal algorithm termination
        std::atomic_bool terminate = false;
        // produce some offspring
        auto iterate = [&](gsl::index i) {
            Operon::RandomGenerator rndlocal{seeds[i]};

            while (!(terminate = generator.Terminate())) {
                if (auto result = generator(rndlocal, config.CrossoverProbability, config.MutationProbability); result.has_value()) {
                    offspring[i] = std::move(result.value());
                    return;
                }
            }
        };

        // report statistics for the initial population 
        if (report) { std::invoke(report); }

        for (generation = 1; generation <= config.Generations; ++generation) {
            // get some new seeds
            std::generate(seeds.begin(), seeds.end(), [&]() { return random(); });
            // preserve one elite
            auto best = std::min_element(parents.begin(), parents.end(), [&](const auto& lhs, const auto& rhs) { return lhs[idx] < rhs[idx]; });
            offspring[0] = *best;

            generator.Prepare(parents);
            // we always allow one elite (maybe this should be more configurable?)
            std::for_each(executionPolicy, indices.cbegin() + 1, indices.cbegin() + config.PoolSize, iterate);
            // merge pool back into pop
            reinserter(random, parents, offspring);

            // report progress and stats
            if (report) { std::invoke(report); }

            // stop if termination requested
            //if (terminate || best->Fitness[idx] < 1e-6) { return; }
            if (terminate) return;
        }
    }
};
} // namespace operon

#endif

