// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm> // for max
#include <atomic> // for atomic_size_t
#include <chrono> // for steady_clock
#include <cstddef> // for size_t
#include <functional> // for std::function
#include <optional> // for std::optional
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
#include "operon/operators/evaluator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/reinserter.hpp"

namespace Operon {
auto GeneticProgrammingAlgorithm::Run(
    tf::Executor& executor, Operon::RandomGenerator& random, Operon::ReportCallback report, bool warmStart) -> void
{
    auto const savedGeneration = Generation();
    Reset();
    if (warmStart) {
        Generation() = savedGeneration;
    }

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

    // Reuse worker RNG states on warm resume.
    size_t const s = std::max(config.PopulationSize, config.PoolSize);
    auto& rngs = WorkerRngs();
    if (rngs.size() != s) {
        rngs.clear();
        rngs.reserve(s);
        for (size_t i = 0; i < s; ++i) {
            rngs.emplace_back(random());
        }
    }

    auto const& evaluator = generator->Evaluator();

    // Reuse evaluation buffers across generations.
    auto trainSize = problem->TrainingRange().Size();

    ENSURE(executor.num_workers() > 0);
    std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());

    // Workers poll stop() concurrently; only ordered tasks publish Elapsed().
    // A worker that observes a stop condition latches it here, so a condition that is transient (e.g. selection
    // pressure, which Prepare() resets) still ends the run after the generation it cut short. This is internal
    // state: StopRequested() only ever reflects RequestStop() and a true report callback.
    std::atomic_bool terminationObserved { false };
    auto shouldStop = [&](double elapsed) -> bool {
        return StopRequested() || terminationObserved.load(std::memory_order_relaxed) || generator->Terminate()
            || Generation() == config.Generations || elapsed > static_cast<double>(config.TimeLimit);
    };
    auto stop = [&]() -> bool {
        auto const stopping = shouldStop(computeElapsed());
        if (stopping) {
            terminationObserved.store(true, std::memory_order_relaxed);
        }
        return stopping;
    };
    auto loopCondition = [&]() -> bool {
        Elapsed() = computeElapsed();
        return shouldStop(Elapsed());
    };

    auto parents = Parents();
    auto offspring = Offspring();
    std::vector<Operon::RandomGenerator> savedRngs; // used only on warm resume
    std::vector<std::optional<std::vector<Operon::Scalar>>> originalCoeffs(parents.size());
    std::atomic_size_t generatedOffspring {}; // pool slots filled in the current generation
    std::atomic_size_t attemptedOffspring {}; // generator invocations (accepted + rejected) in the current generation

    // Subflow tasks outlive the init callable, so keep this in Run's scope.
    bool const warmResume = IsFitted() && warmStart;

    auto timer = executor.make_observer<PhaseTimer>();

    // while loop control flow
    tf::Taskflow taskflow;
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareEval
                = subflow.emplace([&]() -> void { evaluator->Prepare(parents); }).name("prepare evaluator");
            auto reportProgress = subflow
                                      .emplace([&, timer]() -> void {
                                          Timings() = timer->Timings();
                                          if (report && std::invoke(report)) {
                                              RequestStop();
                                          }
                                      })
                                      .name("report progress");

            // Local search runs separately before evaluation on cold start.
            auto eval
                = subflow
                      .for_each_index(size_t { 0 }, parents.size(), size_t { 1 },
                          [&](size_t i) -> void {
                              auto id = executor.this_worker_id();
                              if (slots[id].size() < trainSize) {
                                  slots[id].resize(trainSize);
                              }
                              ScoreIndividual(rngs[i], parents[i], *evaluator, generator->Optimizer(), /*pLocal=*/0.0,
                                  config.LamarckianProbability, Operon::Span<Operon::Scalar>(slots[id]));
                          })
                      .name("evaluate population");
            if (warmResume) {
                // Re-evaluate on resume without advancing worker RNG states.
                auto saveRngs = subflow.emplace([&]() -> void { savedRngs = rngs; }).name("save rng states");
                auto restoreRngs = subflow.emplace([&]() -> void { rngs = std::move(savedRngs); }).name("restore rng states");
                prepareEval.precede(saveRngs);
                saveRngs.precede(eval);
                eval.precede(restoreRngs);
                restoreRngs.precede(reportProgress);
            } else {
                auto init = subflow
                                .for_each_index(size_t { 0 }, parents.size(), size_t { 1 },
                                    [&](size_t i) -> void {
                                        parents[i].Genotype = (*treeInit)(rngs[i]);
                                        (*coeffInit)(rngs[i], parents[i].Genotype);
                                    })
                                .name("initialize population");
                // Prepare sees the post-optimization population.
                auto localSearch = subflow
                                       .for_each_index(size_t { 0 }, parents.size(), size_t { 1 },
                                           [&](size_t i) -> void {
                                               originalCoeffs[i] = LocalSearch(rngs[i], parents[i], *evaluator,
                                                   generator->Optimizer(), config.LocalSearchProbability,
                                                   config.LamarckianProbability);
                                           })
                                       .name("local search on initial population");
                auto restoreCoeffs = subflow
                                         .for_each_index(size_t { 0 }, parents.size(), size_t { 1 },
                                             [&](size_t i) -> void {
                                                 if (originalCoeffs[i]) {
                                                     auto const id = executor.this_worker_id();
                                                     if (slots[id].size() < trainSize) {
                                                         slots[id].resize(trainSize);
                                                     }
                                                     RestoreNonLamarckianCoefficients(parents[i], *evaluator,
                                                         *originalCoeffs[i], Operon::Span<Operon::Scalar>(slots[id]));
                                                 }
                                             })
                                         .name("restore non-lamarckian coefficients");
                init.precede(localSearch);
                localSearch.precede(prepareEval);
                prepareEval.precede(eval);
                eval.precede(restoreCoeffs);
                restoreCoeffs.precede(reportProgress);
            }
        }, // init
        loopCondition, // loop condition
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareGenerator = subflow
                                        .emplace([&]() -> void {
                                            generatedOffspring = 0;
                                            attemptedOffspring = 0;
                                            SetIncompleteGeneration(std::nullopt);
                                            generator->Prepare(parents);
                                            // Offspring belong to the next generation.
                                            if (auto* cache = config.Cache) {
                                                cache->SetGeneration(Generation() + 1);
                                            }
                                        })
                                        .name("prepare generator");
            auto generateOffspring
                = subflow
                      .for_each_index(size_t { 0 }, offspring.size(), size_t { 1 },
                          [&](size_t i) -> void {
                              slots[executor.this_worker_id()].resize(trainSize);
                              auto buf = Operon::Span<Operon::Scalar>(slots[executor.this_worker_id()]);
                              size_t attempts = 0;
                              while (!stop()) {
                                  ++attempts;
                                  if (auto result
                                      = (*generator)(rngs[i], config.CrossoverProbability, config.MutationProbability,
                                          config.LocalSearchProbability, config.LamarckianProbability, buf);
                                      result.has_value()) {
                                      offspring[i] = std::move(result.value());
                                      generatedOffspring.fetch_add(1, std::memory_order_relaxed);
                                      break;
                                  }
                              }
                              attemptedOffspring.fetch_add(attempts, std::memory_order_relaxed);
                          })
                      .name("generate offspring");
            // Discard incomplete pools; unfilled slots may contain stale individuals.
            auto reinsert
                = subflow
                      .emplace([&]() -> void {
                          auto const generated = generatedOffspring.load(std::memory_order_relaxed);
                          if (generated != offspring.size()) {
                              SetIncompleteGeneration(OffspringProgress {
                                  .Attempted = attemptedOffspring.load(std::memory_order_relaxed),
                                  .Completed = generated,
                                  .PoolSize = offspring.size() });
                              // No RequestStop(): the latched termination ends the run, and StopRequested() stays
                              // false unless the user asked.
                              return;
                          }
                          (*reinserter)(random, Parents(), offspring);
                      })
                      .name("reinsert");
            auto incrementGeneration
                = subflow
                      .emplace([&]() -> void {
                          if (generatedOffspring.load(std::memory_order_relaxed) == offspring.size()) {
                              ++Generation();
                          }
                      })
                      .name("increment generation");
            auto reportProgress = subflow
                                      .emplace([&, timer]() -> void {
                                          // Include discarded work in telemetry, but skip its report.
                                          Elapsed() = computeElapsed();
                                          Timings() = timer->Timings();
                                          if (generatedOffspring.load(std::memory_order_relaxed) != offspring.size()) {
                                              return;
                                          }
                                          if (report && std::invoke(report)) {
                                              RequestStop();
                                          }
                                      })
                                      .name("report progress");

            // set-up subflow graph
            prepareGenerator.precede(generateOffspring);
            generateOffspring.precede(reinsert);
            reinsert.precede(incrementGeneration);
            incrementGeneration.precede(reportProgress);
        }, // loop body (evolutionary main loop)
        [&]() -> int { return 0; }, // jump back to the next iteration
        [&]() -> void {
            IsFitted() = true; /* all done */
        } // work done, report last gen and stop
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

    auto cleanupTimer = [&]() -> void {
        Timings() = timer->Timings();
        executor.remove_observer(std::move(timer));
    };

    try {
        executor.run(taskflow).get(); // .wait() would silently drop an exception thrown by any task
    } catch (...) {
        cleanupTimer();
        throw;
    }
    cleanupTimer();
}

auto GeneticProgrammingAlgorithm::Run(
    Operon::RandomGenerator& random, Operon::ReportCallback report, size_t threads, bool warmStart) -> void
{
    if (threads == 0) {
        threads = std::thread::hardware_concurrency();
    }
    tf::Executor executor(threads);
    Run(executor, random, std::move(report), warmStart);
}
} // namespace Operon
