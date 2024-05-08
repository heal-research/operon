// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>                         // for max, min_element
#include <atomic>                            // for atomic_bool
#include <chrono>                            // for steady_clock
#include <memory>                            // for allocator, allocator_tra...
#include <optional>                          // for optional
#include <taskflow/taskflow.hpp>             // for taskflow, subflow
#include <taskflow/algorithm/for_each.hpp>   // for taskflow.for_each_index
#include <vector>                            // for vector, vector::size_type

#include "operon/algorithms/gp.hpp"
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

    auto t0 = std::chrono::steady_clock::now();
    auto elapsed = [t0]() {
        auto t1 = std::chrono::steady_clock::now();
        constexpr double ms{1e3};
        return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()) / ms;
    };

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

    auto stop = [&]() {
        return generator.Terminate() || Generation() == config.Generations || elapsed() > static_cast<double>(config.TimeLimit);
    };

    auto parents = Parents();
    auto offspring = Offspring();

    // while loop control flow
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&](tf::Subflow& subflow) {
            auto init = subflow.for_each_index(size_t{0}, parents.size(), size_t{1}, [&](size_t i) {
                parents[i].Genotype = treeInit(rngs[i]);
                coeffInit(rngs[i], parents[i].Genotype);
            }).name("initialize population");
            auto prepareEval = subflow.emplace([&]() { evaluator.Prepare(parents); }).name("prepare evaluator");
            auto eval = subflow.for_each_index(size_t{0}, parents.size(), size_t{1}, [&](size_t i) {
                auto id = executor.this_worker_id();
                // make sure the worker has a large enough buffer
                if (slots[id].size() < trainSize) {
                    slots[id].resize(trainSize);
                }
                parents[i].Fitness = evaluator(rngs[i], parents[i], slots[id]);
            }).name("evaluate population");
            auto reportProgress = subflow.emplace([&](){ if (report) { std::invoke(report); } }).name("report progress");
            init.precede(prepareEval);
            prepareEval.precede(eval);
            eval.precede(reportProgress);
        }, // init
        stop, // loop condition
        [&](tf::Subflow& subflow) {
            auto keepElite = subflow.emplace([&]() {
                offspring[0] = *std::min_element(parents.begin(), parents.end(), [&](const auto& lhs, const auto& rhs) { return lhs[idx] < rhs[idx]; });
            }).name("keep elite");
            auto prepareGenerator = subflow.emplace([&]() { generator.Prepare(parents); }).name("prepare generator");
            auto generateOffspring = subflow.for_each_index(size_t{1}, offspring.size(), size_t{1}, [&](size_t i) {
                auto buf = Operon::Span<Operon::Scalar>(slots[executor.this_worker_id()]);
                while (!stop()) {
                    if (auto result = generator(rngs[i], config.CrossoverProbability, config.MutationProbability, config.LocalSearchProbability, buf); result.has_value()) {
                        offspring[i] = std::move(result.value());
                        return;
                    }
                }
            }).name("generate offspring");
            auto reinsert = subflow.emplace([&]() { reinserter(random, Parents(), offspring); }).name("reinsert");
            auto incrementGeneration = subflow.emplace([&]() { ++Generation(); }).name("increment generation");
            auto reportProgress = subflow.emplace([&](){ if (report) { std::invoke(report); } }).name("report progress");

            // set-up subflow graph
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

auto GeneticProgrammingAlgorithm::Run(Operon::RandomGenerator& random, std::function<void()> report, size_t threads) -> void {
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
    }
    tf::Executor executor(threads);
    Run(executor, random, std::move(report));
}
} // namespace Operon
