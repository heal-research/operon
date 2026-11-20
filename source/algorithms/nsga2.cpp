// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm> // for stable_sort, copy_n, max
#include <atomic> // for atomic_size_t
#include <chrono> // for steady_clock
#include <cmath> // for isfinite
#include <iterator> // for move_iterator, back_inse...
#include <limits> // for numeric_limits
#include <memory> // for allocator, allocator_tra...
#include <optional> // for optional
#include <random> // for bernoulli_distribution
#include <taskflow/algorithm/for_each.hpp> // for taskflow.for_each_index
#include <vector> // for vector, vector::size_type

#include "budget_ledger.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/algorithms/phase_timer.hpp"
#include "operon/core/contracts.hpp" // for ENSURE
#include "operon/core/problem.hpp" // for Problem
#include "operon/core/range.hpp" // for Range
#include "operon/core/tree.hpp" // for Tree
#include "operon/hash/zobrist.hpp" // for Zobrist::SlotScope
#include "operon/operators/evaluator.hpp" // for ScoreIndividual
#include "operon/operators/initializer.hpp" // for CoefficientInitializerBase
#include "operon/operators/non_dominated_sorter.hpp" // for RankSorter
#include "operon/operators/reinserter.hpp" // for ReinserterBase

namespace Operon {

auto NSGA2::UpdateDistance(Operon::Span<Individual> pop) -> void
{
    // assign distance. each front is sorted for each objective
    size_t const m = pop.front().Fitness.size();
    auto inf = std::numeric_limits<Operon::Scalar>::infinity();
    for (size_t i = 0; i < fronts_.size(); ++i) {
        auto& front = fronts_[i];
        for (size_t obj = 0; obj < m; ++obj) {
            SingleObjectiveComparison comp(obj);
            std::stable_sort(front.begin(), front.end(), [&](auto a, auto b) -> auto { return comp(pop[a], pop[b]); });
            auto min = pop[front.front()][obj];
            auto max = pop[front.back()][obj];
            for (size_t j = 0; j < front.size(); ++j) {
                auto idx = front[j];

                pop[idx].Rank = i;
                if (obj == 0) {
                    pop[idx].Distance = 0;
                }

                auto mPrev = j > 0 ? pop[front[j - 1]][obj] : inf;
                auto mNext = j < front.size() - 1 ? pop[front[j + 1]][obj] : inf;
                auto distance = (mNext - mPrev) / (max - min);
                if (j == 0 || j == front.size() - 1) {
                    distance = inf;
                } else if (!std::isfinite(distance)) {
                    distance = 0;
                }
                pop[idx].Distance += distance;
            }
        }
    }
}

auto NSGA2::Sort(Operon::Span<Individual> pop) -> void
{
    auto config = GetConfig();
    auto eps = static_cast<Operon::Scalar>(config.Epsilon);
    auto eq
        = [eps](auto const& lhs, auto const& rhs) -> auto { return Operon::Equal {}(lhs.Fitness, rhs.Fitness, eps); };

    // Preserve the parent and offspring subspans by sorting indices.
    Operon::Vector<size_t> order(pop.size());
    std::iota(order.begin(), order.end(), size_t { 0 });
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) -> auto {
        return std::ranges::lexicographical_compare(pop[a].Fitness, pop[b].Fitness);
    });
    // mark the duplicates for stable_partition
    for (auto i = order.begin(); i < order.end();) {
        pop[*i].Rank = 0;
        auto j = i + 1;
        for (; j < order.end() && eq(pop[*i], pop[*j]); ++j) {
            pop[*j].Rank = 1;
        }
        i = j;
    }
    auto r = std::stable_partition(order.begin(), order.end(), [&](size_t i) -> auto { return !pop[i].Rank; });
    // The sorter reads only fitness; avoid copying genotypes.
    Operon::Vector<Individual> uniq;
    uniq.reserve(static_cast<size_t>(std::distance(order.begin(), r)));
    for (auto it = order.begin(); it < r; ++it) {
        Individual ind;
        ind.Fitness = pop[*it].Fitness;
        uniq.push_back(std::move(ind));
    }
    // do the sorting
    fronts_ = (*sorter_)(Operon::Span<Operon::Individual const>(uniq.data(), uniq.size()), eps);
    // translate front indices (into uniq) back to true indices into pop
    for (auto& f : fronts_) {
        for (auto& idx : f) {
            idx = order[idx];
        }
    }
    // sort the fronts for consistency between sorting algos
    for (auto& f : fronts_) {
        std::stable_sort(f.begin(), f.end());
    }
    // banish the duplicates into the last front (already true pop indices)
    if (r < order.end()) {
        Operon::Vector<size_t> last(static_cast<size_t>(std::distance(r, order.end())));
        std::copy(r, order.end(), last.begin());
        std::stable_sort(last.begin(), last.end());
        fronts_.push_back(last);
    }
    // calculate crowding distance
    UpdateDistance(pop);
    // update best front
    best_.clear();
    std::transform(fronts_.front().begin(), fronts_.front().end(), std::back_inserter(best_),
        [&](auto i) -> auto { return pop[i]; });
}

auto NSGA2::Run(tf::Executor& executor, Operon::RandomGenerator& random, Operon::ReportCallback report, bool warmStart)
    -> void // NOLINT(readability-function-cognitive-complexity)
{
    // Configuration error (std::invalid_argument): rejected before anything about the algorithm changes.
    detail::RequireViableConfiguration(GetConfig());

    auto const savedGeneration = Generation();
    // Resumability is captured before anything is reset: only an object whose last Run() completed (or whose
    // state a checkpoint restore replaced and marked fitted) holds a population a warm start may continue.
    auto const wasFitted = IsFitted();
    if (warmStart && !wasFitted) {
        // A Run() that ended in an exception leaves parents, ranks and RNG states half-way through a generation, and a
        // never-run object has no population at all: a warm start would silently build a new population at the
        // generation the object had reached.
        throw std::logic_error("warm start of an algorithm without a fitted population (the object was never run, or "
                               "a previous Run() threw an exception, or a restored checkpoint did not mark it "
                               "fitted): an algorithm object is not resumable after a task exception, create a new "
                               "one");
    }
    Reset();
    if (warmStart) {
        Generation() = savedGeneration;
    }
    // Cleared for the duration of the run, set again only by the `done` task: an exception anywhere in between leaves
    // the object not fitted, however the previous run ended.
    IsFitted() = false;

    const auto& config = GetConfig();
    const auto& treeInit = GetTreeInitializer();
    const auto& coeffInit = GetCoefficientInitializer();
    const auto& generator = GetGenerator();
    const auto& reinserter = GetReinserter();
    const auto& problem = GetProblem();

    auto t0 = std::chrono::steady_clock::now();
    auto computeElapsed = [t0]() -> double {
        auto t1 = std::chrono::steady_clock::now();
        constexpr double us { 1e6 };
        return static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / us;
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

    auto const* evaluator = generator->Evaluator();
    // The run's optimizer accounting (reset by Reset() above): every local search below is recorded here.
    auto& accounting = Accounting();

    // Reuse evaluation buffers across generations.
    auto trainSize = problem->TrainingRange().Size();

    ENSURE(executor.num_workers() > 0);
    std::vector<std::vector<Operon::Scalar>> slots(executor.num_workers());

    // A worker can observe a transient generator condition such as selection pressure before Prepare() clears it.
    // Latch that observation so the loop cannot start another generation after the incomplete pool is discarded.
    std::atomic_bool terminationObserved { false };
    auto stop = [&]() -> bool {
        auto const stopping = terminationObserved.load(std::memory_order_relaxed) || PollStop(computeElapsed());
        if (stopping) {
            terminationObserved.store(true, std::memory_order_relaxed);
        }
        return stopping;
    };
    auto loopCondition = [&]() -> bool {
        Elapsed() = computeElapsed();
        return terminationObserved.load(std::memory_order_relaxed) || PollStop(Elapsed());
    };

    auto& individuals = Individuals();
    auto parents = Parents();
    auto offspring = Offspring();
    std::vector<Operon::RandomGenerator> savedRngs; // used only on warm resume
    // The fitness a warm resume was given (a checkpoint's parents). It is not reproducible by a plain evaluation of the
    // genotype whenever the individual's fitness came from a transposition-cache hit (the entry holds the fitness of
    // another coefficient set of the same structure) or from local search that did not write its coefficients back, so
    // the re-evaluation below must not replace it: selection of the next generation reads it. The re-evaluation still
    // runs (Prepare() of the evaluator, and an evaluator with another objective count wins), but it does NOT detect
    // an evaluator that computes something else with the same number of objectives: the caller has to establish
    // that the evaluator matches the checkpoint (the command-line programs compare a run fingerprint first).
    std::vector<Operon::Vector<Operon::Scalar>> savedFitness; // used only on warm resume
    std::vector<Operon::RandomGenerator> generationRngs; // slot RNG states at the start of the generation attempt
    std::vector<std::optional<std::vector<Operon::Scalar>>> originalCoeffs(parents.size());

    // Declared here (Run()'s own scope), not inside the "init" task's
    // callable below: a subflow's tasks run after that callable returns, so
    // a variable local to it and captured by reference would be dangling by
    // the time a worker thread actually reads it.
    bool const warmResume = warmStart; // wasFitted is guaranteed here

    // Rank and Distance of the parents are state of the run: the tournament selection of the next generation reads
    // them, and after a completed generation they come from the last sort of parents plus offspring, not from a sort
    // of the retained parents. A warm resume re-sorts the parents (to rebuild the fronts), which would replace them;
    // they are put back afterwards so that the resumed run continues like the uninterrupted one. They are put back
    // only together with the checkpointed fitness: if the objective count changed, the freshly sorted values (of the
    // new objectives) are kept, because the old ones belong to another objective space.
    bool fitnessKept = true;
    std::vector<std::pair<size_t, Operon::Scalar>> savedRankDistance;
    if (warmResume) {
        savedRankDistance.reserve(parents.size());
        for (auto const& ind : parents) {
            savedRankDistance.emplace_back(ind.Rank, ind.Distance);
        }
    }

    // while loop control flow
    // The cache whose generation is being generated and has not been committed yet (set when the generation is
    // opened, cleared when it is committed or rolled back). Written only by the single-task steps of the loop body
    // and read after the run, when an exception unwound it.
    Operon::Zobrist* openGenerationCache = nullptr;

    auto timer = executor.make_observer<PhaseTimer>();

    tf::Taskflow taskflow;
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareEval
                = subflow.emplace([&]() -> void { evaluator->Prepare(parents); }).name("prepare evaluator");
            auto nonDominatedSort = subflow
                                        .emplace([&]() -> void {
                                            Sort(parents);
                                            for (size_t i = 0; fitnessKept && i < savedRankDistance.size(); ++i) {
                                                parents[i].Rank = savedRankDistance[i].first;
                                                parents[i].Distance = savedRankDistance[i].second;
                                            }
                                        })
                                        .name(std::string { SortTaskName });
            auto reportProgress = subflow
                                      .emplace([&, timer]() -> void {
                                          ledger.CommitInitialization();
                                          SetBudgetReport(ledger.Report());
                                          Timings() = timer->Timings();
                                          if (report && std::invoke(report)) {
                                              RequestStop();
                                          }
                                      })
                                      .name("report progress");
            nonDominatedSort.precede(reportProgress);

            // Sort after evaluation; local search runs separately on cold start.
            auto eval
                = subflow
                      .for_each_index(size_t { 0 }, parents.size(), size_t { 1 },
                          [&](size_t i) -> void {
                              auto const id = executor.this_worker_id();
                              slots[id].resize(trainSize);
                              ScoreIndividual(rngs[i], parents[i], *evaluator, generator->Optimizer(), /*pLocal=*/0.0,
                                  config.LamarckianProbability, Operon::Span<Operon::Scalar>(slots[id]), accounting);
                          })
                      .name("evaluate population");
            if (warmResume) {
                // Re-evaluate to catch evaluator/objective config mismatches, but snapshot and restore
                // the worker RNG states so that subsequent generations remain deterministic.
                auto saveRngs = subflow
                                    .emplace([&]() {
                                        savedRngs = rngs;
                                        savedFitness.clear();
                                        for (auto const& ind : parents) {
                                            savedFitness.push_back(ind.Fitness);
                                        }
                                    })
                                    .name("save rng states");
                auto restoreRngs = subflow
                                       .emplace([&]() {
                                           rngs = std::move(savedRngs);
                                           // keep the checkpointed fitness of an individual unless the objective
                                           // count changed (an evaluator that does not match the checkpoint): then the
                                           // re-evaluation wins
                                           for (size_t i = 0; i < savedFitness.size(); ++i) {
                                               if (savedFitness[i].size() == parents[i].Fitness.size()) {
                                                   parents[i].Fitness = std::move(savedFitness[i]);
                                               } else {
                                                   fitnessKept = false;
                                               }
                                           }
                                       })
                                       .name("restore rng states");
                prepareEval.precede(saveRngs);
                saveRngs.precede(eval);
                eval.precede(restoreRngs);
                restoreRngs.precede(nonDominatedSort);
            } else {
                auto init = subflow
                                .for_each_index(size_t { 0 }, parents.size(), size_t { 1 },
                                    [&](size_t i) -> void {
                                        parents[i].Genotype = (*treeInit)(rngs[i]);
                                        (*coeffInit)(rngs[i], parents[i].Genotype);
                                    })
                                .name("initialize population");
                // Local search runs before prepareEval (not after, alongside
                // eval) so that an evaluator snapshotting the population in
                // Prepare() (e.g. DiversityEvaluator) sees post-optimization
                // genotypes rather than the raw initial ones.
                auto localSearch
                    = subflow
                          .for_each_index(size_t { 0 }, parents.size(), size_t { 1 },
                              [&](size_t i) -> void {
                                  auto search = LocalSearch(rngs[i], parents[i], generator->Optimizer(),
                                      config.LocalSearchProbability, config.LamarckianProbability);
                                  // Charged before the initial population is scored or committed.
                                  accounting.Record(search);
                                  originalCoeffs[i] = std::move(search.OriginalCoefficients);
                              })
                          .name("local search on initial population");
                auto restoreCoeffs = subflow
                                         .for_each_index(size_t { 0 }, parents.size(), size_t { 1 },
                                             [&](size_t i) -> void {
                                                 if (originalCoeffs[i]) {
                                                     auto const id = executor.this_worker_id();
                                                     slots[id].resize(trainSize);
                                                     RestoreNonLamarckianCoefficients(parents[i], *evaluator,
                                                         *originalCoeffs[i], Operon::Span<Operon::Scalar>(slots[id]));
                                                 }
                                             })
                                         .name("restore non-lamarckian coefficients");
                init.precede(localSearch);
                localSearch.precede(prepareEval);
                prepareEval.precede(eval);
                eval.precede(restoreCoeffs);
                restoreCoeffs.precede(nonDominatedSort);
            }
        }, // init
        loopCondition, // loop condition
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareGenerator = subflow
                                        .emplace([&]() -> void {
                                            BeginOffspringProgress();
                                            SetIncompleteGeneration(std::nullopt);
                                            // The slot RNG streams are restored from here if the generation is
                                            // discarded, so a warm resume does not depend on how far it got.
                                            generationRngs = rngs;
                                            ledger.BeginGeneration(generator->MaxAttempts(parents.size()));
                                            generator->Prepare(parents);
                                            // Stamp the cache clock with the generation offspring will
                                            // belong to *before* they're evaluated - Generation() itself
                                            // isn't incremented until after reinsert() below, so without
                                            // this the cache would stamp entries with the prior
                                            // generation's number (see incrementGeneration). The
                                            // generator's cache is the one entries are stamped in and
                                            // expired from (SetCache may have replaced config.Cache).
                                            if (auto* cache = generator->Cache()) {
                                                cache->SetGeneration(Generation() + 1);
                                                openGenerationCache = cache;
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
                              bool filled = false;
                              while (!stop()) {
                                  ++attempts;
                                  auto result
                                      = (*generator)(rngs[i], config.CrossoverProbability, config.MutationProbability,
                                          config.LocalSearchProbability, config.LamarckianProbability, buf, accounting);
                                  ledger.AddAttempt(lastSlot, result.has_value());
                                  if (result) {
                                      offspring[i] = std::move(*result);
                                      filled = true;
                                      ENSURE(offspring[i].Genotype.Length() > 0);
                                      break;
                                  }
                              }
                              CountOffspring(attempts, filled);
                          })
                      .name("generate offspring");
            auto nonDominatedSort = subflow
                                        .emplace([&]() -> void {
                                            auto const progress = Progress();
                                            if (progress.Completed != offspring.size()) {
                                                SetIncompleteGeneration(progress);
                                                // The poll that ended the generation recorded its reason; this only
                                                // makes sure the run loop sees it (external requests are sticky, the
                                                // limits only grow).
                                                RecordStop(StopConditions(computeElapsed()));
                                                return;
                                            }
                                            Sort(individuals);
                                        })
                                        .name(std::string { SortTaskName });
            // Delegates to the reinserter's actual merge strategy (same call
            // shape as GP, gp.cpp) instead of just truncating a globally
            // sorted array via ReinserterBase::Sort.
            auto reinsert = subflow
                                .emplace([&]() -> void {
                                    if (Progress().Completed == offspring.size()) {
                                        (*reinserter)(random, parents, offspring);
                                    }
                                })
                                .name("reinsert");
            auto incrementGeneration = subflow
                                           .emplace([&]() -> void {
                                               if (Progress().Completed == offspring.size()) {
                                                   ++Generation();
                                               }
                                           })
                                           .name("increment generation");
            auto reportProgress = subflow
                                      .emplace([&, timer]() -> void {
                                          // Include discarded work in telemetry, but skip its report.
                                          Elapsed() = computeElapsed();
                                          Timings() = timer->Timings();
                                          if (Progress().Completed != offspring.size()) {
                                              return;
                                          }
                                          if (report && std::invoke(report)) {
                                              RequestStop();
                                          }
                                      })
                                      .name("report progress");

            // set-up subflow graph
            prepareGenerator.precede(generateOffspring);
            generateOffspring.precede(nonDominatedSort);
            nonDominatedSort.precede(reinsert);
            reinsert.precede(incrementGeneration);
            incrementGeneration.precede(reportProgress);
        }, // loop body (evolutionary main loop)
        [&]() -> int { return 0; }, // jump back to the next iteration
        [&]() -> void {
            SetBudgetReport(ledger.Report());
            IsFitted() = true; /* all done */
        } // work done, report last gen and stop
    ); // evolutionary loop

    init.name("init");
    cond.name("termination");
    body.name("main loop");
    back.name("back");
    done.name("done");
    taskflow.name("NSGA2");

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
        // A task threw. The run has finished (get() returned), so no worker touches the cache any more. A generation
        // that was opened but not committed never happened: remove its entries (Committed mode), as for a generation
        // cut short. Nothing else is rolled back: parents, ranks and RNG states may be half-way through a
        // generation, so the object is not resumable (a later warm Run() throws std::logic_error). What the caller
        // can rely on: the cache holds no entry of the failed generation, Generation() is the last committed one,
        // and a new algorithm object can reuse the cache.
        if (openGenerationCache != nullptr) {
            openGenerationCache->AbortGeneration();
        }
        cleanupTimer();
        throw;
    }
    cleanupTimer();
}

auto NSGA2::Run(Operon::RandomGenerator& random, Operon::ReportCallback report, size_t threads, bool warmStart) -> void
{
    if (threads == 0U) {
        threads = std::thread::hardware_concurrency();
    }
    tf::Executor executor(threads);
    Run(executor, random, std::move(report), warmStart);
}
} // namespace Operon
