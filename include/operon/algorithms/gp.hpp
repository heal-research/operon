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
#include <thread>

#include "taskflow/taskflow.hpp"

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

    void Run(Operon::RandomGenerator& random, std::function<void()> report = nullptr, size_t threads = 0) {
        if (!threads) {
            threads = std::thread::hardware_concurrency();
        }
        tf::Executor executor(threads);
        Run(executor, random, report);
    }

    void Run(tf::Executor& executor, Operon::RandomGenerator& random, std::function<void()> report = nullptr)
    {
        auto& config = GetConfig();
        auto& initializer = GetInitializer();
        auto& generator = GetGenerator();
        auto& reinserter = GetReinserter();
        auto& problem = GetProblem();

        // random seeds for each thread
        size_t s = std::max(config.PopulationSize, config.PoolSize);
        std::vector<Operon::RandomGenerator> rngs;
        for (size_t i = 0; i < s; ++i) {
            rngs.emplace_back(random());
        }

        auto idx = 0;
        const auto& evaluator = generator.Evaluator();

        // we want to allocate all the memory that will be necessary for evaluation (e.g. for storing model responses)
        // in one go and use it throughout the generations in order to minimize the memory pressure
        auto trainSize = problem.TrainingRange().Size();

        ENSURE(executor.num_workers() > 0);

        // start the chronometer
        auto t0 = std::chrono::steady_clock::now();

        tf::Taskflow taskflow;
        // initialize population
        taskflow.for_each_index(0ul, parents.size(), 1ul, [&](size_t i) {
            // allocate some memory for this worker
            parents[i].Genotype = initializer(rngs[i]);
            parents[i][idx] = evaluator(rngs[i], parents[i]);
        });
        executor.run(taskflow).wait(); // wait on this future to finish
        taskflow.clear(); // clear tasks and associated data
        ++generation;

        std::atomic_bool terminate{ false }; // flag to signal algorithm termination

        auto preserveElite = taskflow.emplace([&]() {
            offspring[0] = *std::min_element(parents.begin(), parents.end(), [&](const auto& lhs, const auto& rhs) { return lhs[idx] < rhs[idx]; });
        });

        auto prepareGenerator = taskflow.emplace([&]() { generator.Prepare(parents); });

        auto generateOffspring = taskflow.for_each_index(1ul, offspring.size(), 1ul, [&](size_t i) {
            auto id = executor.this_worker_id();
            while (!(terminate = generator.Terminate())) {
                if (auto result = generator(rngs[i], config.CrossoverProbability, config.MutationProbability); result.has_value()) {
                    offspring[i] = std::move(result.value());
                    return;
                }
            }
        });

        auto reinsert = taskflow.emplace([&]() { reinserter(random, parents, offspring); });
        auto reportProgress = taskflow.emplace([&]() { if (report) std::invoke(report); });

        preserveElite.precede(prepareGenerator);
        prepareGenerator.precede(generateOffspring);
        generateOffspring.precede(reinsert);
        reinsert.precede(reportProgress);

        executor.run_until(taskflow, [&]() { return terminate || generation++ == config.Generations; });
        executor.wait_for_all();
    }
};
} // namespace operon

#endif

