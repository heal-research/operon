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

#ifndef GP_HPP
#define GP_HPP

#include "algorithms/config.hpp"
#include "core/eval.hpp"
#include "core/format.hpp"
#include "core/metrics.hpp"
#include "operators/crossover.hpp"
#include "operators/creator.hpp"
#include "operators/mutation.hpp"
#include "operators/generator.hpp"

namespace Operon {
template <typename TCreator, typename TGenerator, typename TReinserter, typename ExecutionPolicy = std::execution::parallel_unsequenced_policy>
class GeneticProgrammingAlgorithm {
    using T = typename TGenerator::FemaleSelectorType::SelectableType;
    static constexpr bool Idx = TGenerator::FemaleSelectorType::SelectableIndex;

private:
    std::reference_wrapper<const Problem> problem_;
    std::reference_wrapper<const GeneticAlgorithmConfig> config_;

    std::reference_wrapper<const TCreator> creator_;
    std::reference_wrapper<const TGenerator> generator_;
    std::reference_wrapper<const TReinserter> reinserter_;

    std::vector<T> parents;
    std::vector<T> offspring;

    size_t generation;

public:
    explicit GeneticProgrammingAlgorithm(const Problem& problem, const GeneticAlgorithmConfig& config, const TCreator& creator, const TGenerator& generator, const TReinserter& reinserter)
        : problem_(problem)
        , config_(config)
        , creator_(creator)
        , generator_(generator)
        , reinserter_(reinserter)
        , parents(config.PopulationSize)
        , offspring(config.PoolSize)
        , generation(0UL)
    {
    }

    const gsl::span<const T> Parents() const { return gsl::span<const T>(parents); }
    const gsl::span<const T> Offspring() const { return gsl::span<const T>(offspring); }

    const Problem& GetProblem() const { return problem_.get(); }
    const GeneticAlgorithmConfig& GetConfig() const { return config_.get(); }

    const TCreator& GetCreator() const { return creator_.get(); }
    const TGenerator& GetGenerator() const { return generator_.get(); }
    const TReinserter& GetReinserter() const { return reinserter_.get(); }

    size_t Generation() const { return generation; }

    void Reset()
    {
        generation = 0;
    }

    void Run(Operon::Random& random, std::function<void()> report = nullptr)
    {
        auto& config       = GetConfig();
        auto& creator      = GetCreator();
        auto& generator    = GetGenerator();
        auto& reinserter   = GetReinserter();
        auto& problem      = GetProblem();
        auto& grammar      = problem.GetGrammar();
        auto targetValues  = problem.TargetValues();
        // easier to work with indices
        std::vector<gsl::index> indices(std::max(config.PopulationSize, config.PoolSize));
        std::iota(indices.begin(), indices.end(), 0L);
        // random seeds for each thread
        std::vector<Operon::Random::result_type> seeds(config.PopulationSize);
        std::generate(seeds.begin(), seeds.end(), [&]() { return random(); });


        const auto& inputs = problem.InputVariables();

        auto create = [&](gsl::index i) {
            // create one random generator per thread
            Operon::Random rndlocal{seeds[i]};
            parents[i].Genotype = creator(rndlocal, grammar, inputs);
            parents[i][Idx] = Operon::Numeric::Max<Operon::Scalar>();
        };
        const auto& evaluator = generator.Evaluator();
        auto evaluate = [&](T& ind) {
            auto f = evaluator(random, ind);
            if (!std::isfinite(f)) { f = Operon::Numeric::Max<Operon::Scalar>(); }
            ind[Idx] = f;
        };

        ExecutionPolicy executionPolicy;

        std::for_each(executionPolicy, indices.begin(), indices.begin() + config.PopulationSize, create);
        std::for_each(executionPolicy, parents.begin(), parents.end(), evaluate);

        // flag to signal algorithm termination
        std::atomic_bool terminate = false;
        // produce some offspring
        auto iterate = [&](gsl::index i) {
            Operon::Random rndlocal{seeds[i]};

            while (!(terminate = generator.Terminate())) {
                if (auto result = generator(rndlocal, config.CrossoverProbability, config.MutationProbability); result.has_value()) {
                    offspring[i] = std::move(result.value());
                    return;
                }
            }
        };

        for (generation = 0; generation < config.Generations; ++generation) {
            // get some new seeds
            std::generate(seeds.begin(), seeds.end(), [&]() { return random(); });

            // preserve one elite
            auto [minElem, maxElem] = std::minmax_element(parents.begin(), parents.end(), [&](const auto& lhs, const auto& rhs) { return lhs[Idx] < rhs[Idx]; });

            auto best = minElem;
            offspring[0] = *best;

            // this assumes fitness is always > 0
            terminate = best->Fitness[Idx] < 1e-6;

            if (report) { std::invoke(report); }

            if (terminate) { return; }

            generator.Prepare(parents);
            // we always allow one elite (maybe this should be more configurable?)
            std::for_each(executionPolicy, indices.cbegin() + 1, indices.cbegin() + config.PoolSize, iterate);
            // merge pool back into pop
            reinserter(random, parents, offspring);
        }
    }
};
} // namespace operon

#endif

