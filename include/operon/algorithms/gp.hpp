// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef GP_HPP
#define GP_HPP

#include "algorithms/config.hpp"
#include "core/format.hpp"
#include "core/metrics.hpp"
#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/generator.hpp"
#include "operators/initializer.hpp"
#include "operators/mutation.hpp"

#include <chrono>
#include <execution>
#include <tbb/global_control.h>
#include <thread>

namespace Operon {

template <typename TInitializer>
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

    std::vector<Individual> const& Parents() const { return parents; }
    std::vector<Individual>& Parents() { return parents; }
    std::vector<Individual> const& Offspring() const { return offspring; }
    std::vector<Individual>& Offspring() { return offspring; }

    const Problem& GetProblem() const { return problem_.get(); }
    const GeneticAlgorithmConfig& GetConfig() const { return config_.get(); }

    const TInitializer& GetInitializer() const { return initializer_.get(); }
    const OffspringGeneratorBase& GetGenerator() const { return generator_.get(); }
    const ReinserterBase& GetReinserter() const { return reinserter_.get(); }

    size_t Generation() const { return generation; }

    void Reset()
    {
        generation = 0;
        generator_.get().Evaluator().Reset();
    }

    void Run(Operon::RandomGenerator& random, std::function<void()> report = nullptr, size_t threads = 0)
    {
        auto& config = GetConfig();
        auto& initializer = GetInitializer();
        auto& generator = GetGenerator();
        auto& reinserter = GetReinserter();
        // easier to work with indices
        std::vector<size_t> indices(std::max(config.PopulationSize, config.PoolSize));
        std::iota(indices.begin(), indices.end(), 0L);
        // one rng per thread
        size_t s = std::max(config.PopulationSize, config.PoolSize);
        std::vector<Operon::RandomGenerator> rngs;
        for (size_t i = 0; i < s; ++i) {
            rngs.emplace_back(random());
        }

        auto idx = 0;
        auto create = [&](size_t i) {
            // create one random generator per thread
            parents[i].Genotype = initializer(rngs[i]);
            parents[i][idx] = Operon::Numeric::Max<Operon::Scalar>();
        };
        const auto& evaluator = generator.Evaluator();

        auto evaluate = [&](Individual& ind) {
            auto f = evaluator(random, ind);
            if (!std::isfinite(f)) {
                f = Operon::Numeric::Max<Operon::Scalar>();
            }
            ind[idx] = f;
        };

        // start the chronometer
        auto t0 = std::chrono::steady_clock::now();

        // generate the initial population and perform evaluation
        tbb::global_control c(tbb::global_control::max_allowed_parallelism, threads ? threads : std::thread::hardware_concurrency());

        std::for_each(std::execution::par_unseq, indices.begin(), indices.begin() + config.PopulationSize, create);
        std::for_each(std::execution::par_unseq, parents.begin(), parents.end(), evaluate);

        // flag to signal algorithm termination
        std::atomic_bool terminate = false;
        // produce some offspring
        auto iterate = [&](size_t i) {
            while (!(terminate = generator.Terminate())) {
                auto t1 = std::chrono::steady_clock::now();
                double elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()) / 1e3;
                terminate = elapsed > config.TimeLimit;

                if (auto result = generator(rngs[i], config.CrossoverProbability, config.MutationProbability); result.has_value()) {
                    offspring[i] = std::move(result.value());
                    return;
                }
            }
        };

        // report statistics for the initial population
        if (report)
            std::invoke(report);

        for (generation = 1; generation <= config.Generations; ++generation) {
            // preserve one elite
            auto best = std::min_element(parents.begin(), parents.end(), [&](const auto& lhs, const auto& rhs) { return lhs[idx] < rhs[idx]; });
            offspring[0] = *best;

            generator.Prepare(parents);
            // we always allow one elite (maybe this should be more configurable?)
            std::for_each(std::execution::par_unseq, indices.cbegin() + 1, indices.cbegin() + config.PoolSize, iterate);
            // merge pool back into pop
            reinserter(random, parents, offspring);

            // report progress and stats
            if (report) {
                std::invoke(report);
            }

            // stop if termination requested
            //if (terminate || best->Fitness[idx] < 1e-6) { return; }
            if (terminate)
                return;
        }
    }
};
} // namespace operon

#endif
