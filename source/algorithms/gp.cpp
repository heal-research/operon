// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm> // for max
#include <chrono> // for steady_clock
#include <cstddef> // for size_t
#include <functional> // for std::function
#include <optional> // for std::optional
#include <random> // for bernoulli_distribution
#include <thread> // for std::thread
#include <utility> // for std::move
#include <vector> // for std::vector
// NOLINTBEGIN(misc-include-cleaner)
#include <taskflow/algorithm/for_each.hpp> // for taskflow.for_each_index
#include <taskflow/core/flow_builder.hpp> // for subflow
#include <taskflow/core/taskflow.hpp> // for taskflow
// NOLINTEND(misc-include-cleaner)

#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/phase_timer.hpp"
#include "operon/core/contracts.hpp" // for ENSURE
#include "operon/core/types.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/reinserter.hpp"

namespace Operon {
auto GeneticProgrammingAlgorithm::Run(tf::Executor& executor, Operon::RandomGenerator& random, Operon::ReportCallback report, bool warmStart) -> void
{
    auto const savedGeneration = Generation();
    Reset();
    if (warmStart) { Generation() = savedGeneration; }

    const auto config = GetConfig();
    const auto& treeInit = GetTreeInitializer();
    const auto& coeffInit = GetCoefficientInitializer();
    const auto& generator = GetGenerator();
    const auto& reinserter = GetReinserter();
    const auto& problem = GetProblem();

    auto t0 = std::chrono::steady_clock::now();
    auto computeElapsed = [t0]() -> double {
        auto t1 = std::chrono::steady_clock::now();
        constexpr double ms { 1e3 };
        return static_cast<double>(std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count()) / ms;
    };

    // random seeds for each thread — reuse existing states on warm resume, seed fresh otherwise
    size_t const s = std::max(config.PopulationSize, config.PoolSize);
    auto& rngs = WorkerRngs();
    if (rngs.size() != s) {
        rngs.clear();
        rngs.reserve(s);
        for (size_t i = 0; i < s; ++i) { rngs.emplace_back(random()); }
    }

    auto const& evaluator = generator->Evaluator();

    // we want to allocate all the memory that will be necessary for evaluation (e.g. for storing model responses)
    // in one go and use it throughout the generations in order to minimize the memory pressure
    auto trainSize = problem->TrainingRange().Size();

    ENSURE(executor.num_workers() > 0);
    std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());

    auto stop = [&]() -> bool {
        Elapsed() = computeElapsed();
        return StopRequested() || generator->Terminate() || Generation() == config.Generations || Elapsed() > static_cast<double>(config.TimeLimit);
    };

    auto parents = Parents();
    auto offspring = Offspring();
    std::vector<Operon::RandomGenerator> savedRngs; // used only on warm resume
    std::vector<std::optional<std::vector<Operon::Scalar>>> originalCoeffs(parents.size());

    auto optimizeInitialIndividual = [&](size_t i) -> std::optional<std::vector<Operon::Scalar>> {
        auto const* optimizer = generator->Optimizer();
        if (optimizer == nullptr || !std::bernoulli_distribution{config.LocalSearchProbability}(rngs[i])) {
            return std::nullopt;
        }

        auto coeff = parents[i].Genotype.GetCoefficients();
        auto t0 = std::chrono::steady_clock::now();
        auto [optimizedTree, outcome] = (*optimizer)(rngs[i], std::move(parents[i].Genotype));
        auto t1 = std::chrono::steady_clock::now();
        auto const& diag = Diagnostics(outcome);
        evaluator->ResidualEvaluations += diag.FunctionEvaluations;
        evaluator->JacobianEvaluations += diag.JacobianEvaluations;
        evaluator->CostFunctionTime += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
        parents[i].Genotype = std::move(optimizedTree);

        if (std::bernoulli_distribution{config.LamarckianProbability}(rngs[i])) {
            return std::nullopt;
        }
        return coeff;
    };

    auto timer = executor.make_observer<PhaseTimer>();

    // while loop control flow
    tf::Taskflow taskflow;
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareEval = subflow.emplace([&]() -> void { evaluator->Prepare(parents); }).name("prepare evaluator");
            auto reportProgress = subflow.emplace([&, timer]() -> void {
                                             Timings() = timer->Timings();
                                             if (report && std::invoke(report)) { RequestStop(); }
                                         }).name("report progress");

            auto eval = subflow.for_each_index(size_t { 0 }, parents.size(), size_t { 1 }, [&](size_t i) -> void {
                                   auto id = executor.this_worker_id();
                                   if (slots[id].size() < trainSize) { slots[id].resize(trainSize); }
                                   parents[i].Fitness = (*evaluator)(rngs[i], parents[i], slots[id]);
                               })
                            .name("evaluate population");
            eval.precede(reportProgress);

            if (IsFitted() && warmStart) {
                // Re-evaluate to catch evaluator/objective config mismatches, but snapshot and restore
                // the worker RNG states so that subsequent generations remain deterministic.
                auto saveRngs    = subflow.emplace([&]() { savedRngs = rngs; }).name("save rng states");
                auto restoreRngs = subflow.emplace([&]() { rngs = std::move(savedRngs); }).name("restore rng states");
                prepareEval.precede(saveRngs);
                saveRngs.precede(eval);
                eval.precede(restoreRngs);
                restoreRngs.precede(reportProgress);
            } else {
                auto init = subflow.for_each_index(size_t { 0 }, parents.size(), size_t { 1 }, [&](size_t i) -> void {
                                       parents[i].Genotype = (*treeInit)(rngs[i]);
                                       (*coeffInit)(rngs[i], parents[i].Genotype);
                                   })
                                .name("initialize population");
                auto localSearch = subflow.for_each_index(size_t { 0 }, parents.size(), size_t { 1 }, [&](size_t i) -> void {
                                       originalCoeffs[i] = optimizeInitialIndividual(i);
                                   })
                                .name("local search on initial population");
                auto restoreCoeffs = subflow.for_each_index(size_t { 0 }, parents.size(), size_t { 1 }, [&](size_t i) -> void {
                                        if (originalCoeffs[i]) { parents[i].Genotype.SetCoefficients(*originalCoeffs[i]); }
                                    })
                                .name("restore non-lamarckian coefficients");
                init.precede(localSearch);
                localSearch.precede(prepareEval);
                prepareEval.precede(eval);
                eval.precede(restoreCoeffs);
                restoreCoeffs.precede(reportProgress);
            }
        }, // init
        stop, // loop condition
        [&, timer](tf::Subflow& subflow) -> void {
            // Elitism (if any) is now handled uniformly by ReinserterBase
            // (reinserter.hpp) - it protects the top EliteCount() parents
            // before reinsert() runs, so no offspring slot needs to be
            // reserved for a hand-picked elite here anymore.
            auto prepareGenerator = subflow.emplace([&]() -> void {
                                        generator->Prepare(parents);
                                        // Stamp the cache clock with the generation offspring will
                                        // belong to *before* they're evaluated - Generation() itself
                                        // isn't incremented until after reinsert() below, so without
                                        // this the cache would stamp entries with the prior
                                        // generation's number (see incrementGeneration).
                                        if (auto* cache = config.Cache) { cache->SetGeneration(Generation() + 1); }
                                    }).name("prepare generator");
            auto generateOffspring = subflow.for_each_index(size_t { 0 }, offspring.size(), size_t { 1 }, [&](size_t i) -> void {
                                                slots[executor.this_worker_id()].resize(trainSize);
                                                auto buf = Operon::Span<Operon::Scalar>(slots[executor.this_worker_id()]);
                                                while (!stop()) {
                                                    if (auto result = (*generator)(rngs[i], config.CrossoverProbability, config.MutationProbability, config.LocalSearchProbability, config.LamarckianProbability, buf); result.has_value()) {
                                                        offspring[i] = std::move(result.value());
                                                        return;
                                                    }
                                                }
                                            })
                                         .name("generate offspring");
            auto reinsert = subflow.emplace([&]() -> void { (*reinserter)(random, Parents(), offspring); }).name("reinsert");
            auto incrementGeneration = subflow.emplace([&]() -> void { ++Generation(); }).name("increment generation");
            auto reportProgress = subflow.emplace([&, timer]() -> void {
                                             Timings() = timer->Timings();
                                             if (report && std::invoke(report)) { RequestStop(); }
                                         }).name("report progress");

            // set-up subflow graph
            prepareGenerator.precede(generateOffspring);
            generateOffspring.precede(reinsert);
            reinsert.precede(incrementGeneration);
            incrementGeneration.precede(reportProgress);
        }, // loop body (evolutionary main loop)
        [&]() -> int { return 0; }, // jump back to the next iteration
        [&]() -> void { IsFitted() = true; /* all done */ } // work done, report last gen and stop
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

    executor.run(taskflow).wait();
    Timings() = timer->Timings();
    executor.remove_observer(std::move(timer));
}

auto GeneticProgrammingAlgorithm::Run(Operon::RandomGenerator& random, Operon::ReportCallback report, size_t threads, bool warmStart) -> void
{
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
    }
    tf::Executor executor(threads);
    Run(executor, random, std::move(report), warmStart);
}
} // namespace Operon
