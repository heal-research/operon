// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon/algorithms/gp.hpp"
#include <algorithm>                         // for max, min_element
#include <atomic>                            // for atomic_bool
#include <memory>                            // for allocator, allocator_tra...
#include <optional>                          // for optional
#include <taskflow/taskflow.hpp>             // for taskflow, subflow
#include <vector>                            // for vector, vector::size_type
#include "operon/core/contracts.hpp"         // for ENSURE
#include "operon/core/operator.hpp"          // for OperatorBase
#include "operon/core/problem.hpp"           // for Problem
#include "operon/core/range.hpp"             // for Range
#include "operon/core/tree.hpp"              // for Tree
#include "operon/operators/initializer.hpp"  // for CoefficientInitializerBase
#include "operon/operators/reinserter.hpp"   // for ReinserterBase

namespace Operon {
    auto GeneticProgrammingAlgorithm::Run(tf::Executor& executor, Operon::RandomGenerator& random, std::function<void()> report) -> void
    {
        const auto& config = GetConfig();
        const auto& treeInit = GetTreeInitializer();
        const auto& coeffInit = GetCoefficientInitializer();
        const auto& generator = GetGenerator();
        const auto& reinserter = GetReinserter();
        const auto& problem = GetProblem();

        // random seeds for each thread
        size_t s = std::max(config.PopulationSize, config.PoolSize);
        std::vector<Operon::RandomGenerator> rngs;
        for (size_t i = 0; i < s; ++i) {
            rngs.emplace_back(random());
        }

        auto idx = 0;
        auto const& evaluator = generator.Evaluator();

        // we want to allocate all the memory that will be necessary for evaluation (e.g. for storing model responses)
        // in one go and use it throughout the generations in order to minimize the memory pressure
        auto trainSize = problem.TrainingRange().Size();

        ENSURE(executor.num_workers() > 0);
        std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());

        tf::Taskflow taskflow;

        std::atomic_bool terminate{ false }; // flag to signal algorithm termination

        // while loop control flow
        auto [init, cond, body, back, done] = taskflow.emplace(
            [&](tf::Subflow& subflow) {
                auto init = subflow.for_each_index(0UL, parents_.size(), 1UL, [&](size_t i) {
                    auto id = executor.this_worker_id();
                    // make sure the worker has a large enough buffer
                    if (slots[id].size() < trainSize) { slots[id].resize(trainSize); }
                    parents_[i].Genotype = treeInit(rngs[i]);
                    coeffInit(rngs[i], parents_[i].Genotype);
                    parents_[i].Fitness = evaluator(rngs[i], parents_[i], slots[id]);
                }).name("initialize population");
                auto reportProgress = subflow.emplace([&](){ if (report) { std::invoke(report); } }).name("report progress");
                init.precede(reportProgress);
            }, // init
            [&]() { return terminate || generation_ == config.Generations; }, // loop condition
            [&](tf::Subflow& subflow) {
                auto keepElite = subflow.emplace([&]() {
                    offspring_[0] = *std::min_element(parents_.begin(), parents_.end(), [&](const auto& lhs, const auto& rhs) { return lhs[idx] < rhs[idx]; });
                }).name("keep elite");
                auto prepareGenerator = subflow.emplace([&]() { generator.Prepare(parents_); }).name("prepare generator");
                auto generateOffspring = subflow.for_each_index(1UL, offspring_.size(), 1UL, [&](size_t i) {
                    auto buf = Operon::Span<Operon::Scalar>(slots[executor.this_worker_id()]);
                    while (!(terminate = generator.Terminate())) {
                        if (auto resULt = generator(rngs[i], config.CrossoverProbability, config.MutationProbability, buf); resULt.has_value()) {
                            offspring_[i] = std::move(resULt.value());
                            return;
                        }
                    }
                }).name("generate offspring");
                auto reinsert = subflow.emplace([&]() { reinserter(random, parents_, offspring_); }).name("reinsert");
                auto incrementGeneration = subflow.emplace([&]() { ++generation_; }).name("increment generation");
                auto reportProgress = subflow.emplace([&](){ if (report) { std::invoke(report); } }).name("report progress");

                // set-up subflow graph
                //reportProgress.precede(keepElite);
                keepElite.precede(prepareGenerator);
                prepareGenerator.precede(generateOffspring);
                generateOffspring.precede(reinsert);
                reinsert.precede(incrementGeneration);
                incrementGeneration.precede(reportProgress);
            }, // loop body (evolutionary main loop)
            [&]() { return 0; }, // jump back to the next iteration
            [&]() { /* all done */ }  // work done, report last gen and stop
        ); // evolutionary loop

        init.name("init");
        cond.name("termination");
        body.name("main loop");
        back.name("back");
        done.name("done");
        taskflow.name("GP");

        init.precede(cond);
        cond.precede(body, done);
        body.precede(back);
        back.precede(cond);

        executor.run(taskflow);
        executor.wait_for_all();
    }

} // namespace Operon
