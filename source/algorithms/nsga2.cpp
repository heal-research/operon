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

    // Sort an index permutation instead of physically reordering `pop`.
    // `pop` here is the caller's combined parents+offspring storage, and
    // Run() hands ReinserterBase::operator() fixed Parents()/Offspring()
    // subspans over that same storage after this call returns - if we
    // physically reordered pop, those subspans would silently become
    // arbitrary halves of a fitness-sorted array rather than "the actual
    // previous parents" / "the actual freshly generated offspring", which
    // would undermine reinsertion strategies that rely on that distinction
    // (e.g. ReplaceWorstReinserter). Rank/Distance below are still written
    // directly into pop[order[i]], so they land on the correct individual
    // regardless of pop's storage order.
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
    // The sorter needs Span<Individual const>, but only ever reads .Fitness
    // (never .Genotype/.Rank/.Distance) - build lightweight fitness-only
    // stand-ins instead of deep-copying every unique individual's tree.
    Operon::Vector<Individual> uniq;
    uniq.reserve(static_cast<size_t>(std::distance(order.begin(), r)));
    for (auto it = order.begin(); it < r; ++it) {
        Individual ind; // default ctor only fills a 1-element placeholder Fitness, unlike Individual(nObj)
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

    // random seeds for each thread — reuse existing states on warm resume, seed fresh otherwise
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

    // we want to allocate all the memory that will be necessary for evaluation (e.g. for storing model responses)
    // in one go and use it throughout the generations in order to minimize the memory pressure
    auto trainSize = problem->TrainingRange().Size();

    ENSURE(executor.num_workers() > 0);
    std::vector<std::vector<Operon::Scalar>> slots(executor.num_workers());

    // The budget is a limit on committed work (see BudgetLedger). Initialization always commits.
    detail::BudgetLedger ledger { evaluator, &accounting, config.Evaluations };
    ledger.Begin();
    // Set by the task that ends the run from inside a generation: a discard (for a resource limit, a stop request or
    // the time limit) or a committed generation that crossed the attempt cap. A resource-limit discard ends
    // the run without a (public) stop request.
    RunEnd ledgerEnd = RunEnd::None;
    bool generationCommitted = false; // outcome of the current generation attempt

    // stop() is polled concurrently by offspring-generation workers and must
    // not write shared state. It holds the two asynchronous stop conditions: a stop request and the time limit.
    // A generation they cut short is discarded; they are not part of the determinism guarantee. The evaluation
    // budget, the attempt cap and the generation limit are not polled here: the first two are decided per
    // generation by the ledger, the generation limit at the loop condition. Elapsed() is published only from
    // single-task contexts (the loop condition and the report task), which the taskflow graph orders against every
    // reader.
    auto stop = [&]() -> bool { return StopRequested() || computeElapsed() > static_cast<double>(config.TimeLimit); };
    // Why the run ends, in the order documented at RunEnd; None to go on.
    auto decideEnd = [&](double elapsed) -> RunEnd {
        if (ledgerEnd != RunEnd::None) {
            return ledgerEnd;
        }
        if (Generation() >= config.Generations) {
            return RunEnd::GenerationLimit;
        }
        if (ledger.Exhausted()) {
            return RunEnd::BudgetSpent;
        }
        if (StopRequested()) {
            return RunEnd::StopRequested;
        }
        return elapsed > static_cast<double>(config.TimeLimit) ? RunEnd::TimeLimit : RunEnd::None;
    };
    auto loopCondition = [&]() -> bool {
        Elapsed() = computeElapsed();
        auto const end = decideEnd(Elapsed());
        if (end != RunEnd::None) {
            ledger.SetEnd(end);
        }
        return end != RunEnd::None;
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
    std::atomic_size_t generatedOffspring {};
    std::atomic_size_t attemptedOffspring {};

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

            // nonDominatedSort runs after eval in both paths to rebuild fronts_ from current fitness.
            // Local search (if any) always runs before prepareEval below, either
            // in the cold-start localSearch task or not at all (warm resume,
            // pLocal=0), so this loop itself never applies local search -
            // passing pLocal=0 makes it a plain evaluate.
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
                                            generatedOffspring = 0;
                                            attemptedOffspring = 0;
                                            generationCommitted = false;
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
                              bool const lastSlot = i + 1 == offspring.size();
                              // Entries this slot inserts are visible to its own later attempts only (see
                              // CacheVisibility::Committed); the scope belongs to the generator's cache.
                              Operon::Zobrist::SlotScope const cacheSlot { generator->Cache() };
                              // Workers stop starting attempts once the generation has already spent more than
                              // the budget has left (or made more attempts than the cap): the counters only
                              // grow, so this fires only for a generation that could not commit anyway.
                              while (!stop() && !ledger.Exceeded()) {
                                  ++attempts;
                                  auto result
                                      = (*generator)(rngs[i], config.CrossoverProbability, config.MutationProbability,
                                          config.LocalSearchProbability, config.LamarckianProbability, buf, accounting);
                                  ledger.AddAttempt(lastSlot, result.has_value());
                                  if (result) {
                                      offspring[i] = std::move(*result);
                                      generatedOffspring.fetch_add(1, std::memory_order_relaxed);
                                      ENSURE(offspring[i].Genotype.Length() > 0);
                                      break;
                                  }
                              }
                              attemptedOffspring.fetch_add(attempts, std::memory_order_relaxed);
                          })
                      .name("generate offspring");
            auto nonDominatedSort
                = subflow
                      .emplace([&]() -> void {
                          // The commit decision (see the GP driver): everything that changes the committed state
                          // (ranks, fronts, best, reinsertion) sits behind it.
                          auto const generated = generatedOffspring.load(std::memory_order_relaxed);
                          auto const full = generated == offspring.size();
                          auto const exceeded = ledger.Exceeded();
                          if (!full || exceeded) {
                              auto const progress = ledger.Progress(attemptedOffspring.load(std::memory_order_relaxed),
                                  generated, offspring.size(), StopRequested());
                              SetIncompleteGeneration(progress);
                              ledger.DiscardGeneration();
                              // The offspring workers have finished. Their cache entries belong to a generation
                              // that does not exist: drop them so a retry cannot see them (Committed only).
                              if (auto* cache = generator->Cache()) {
                                  cache->AbortGeneration();
                              }
                              openGenerationCache = nullptr;
                              rngs = generationRngs;
                              if (exceeded) {
                                  ledgerEnd = RunEnd::ResourceLimit;
                              } else {
                                  // cut short by a stop request or the time limit
                                  ledgerEnd = progress.Reason == DiscardReason::StopRequested ? RunEnd::StopRequested
                                                                                              : RunEnd::TimeLimit;
                                  RequestStop();
                              }
                              return;
                          }
                          ledger.CommitGeneration();
                          generationCommitted = true;
                          // A committed generation that made more attempts than the cap (its last attempt
                          // crossed it) ends the run after it is reported, like the selection-pressure stop did.
                          if (ledger.AttemptCapReached()) {
                              ledgerEnd = RunEnd::AttemptCap;
                          }
                          Sort(individuals);
                      })
                      .name(std::string { SortTaskName });
            // Delegates to the reinserter's actual merge strategy (same call
            // shape as GP, gp.cpp) instead of just truncating a globally
            // sorted array via ReinserterBase::Sort.
            auto reinsert = subflow
                                .emplace([&]() -> void {
                                    if (generationCommitted) {
                                        (*reinserter)(random, parents, offspring);
                                    }
                                })
                                .name("reinsert");
            auto incrementGeneration = subflow
                                           .emplace([&]() -> void {
                                               if (generationCommitted) {
                                                   ++Generation();
                                                   openGenerationCache
                                                       = nullptr; // committed: a later exception must not undo it
                                                   SetBudgetReport(ledger.Report());
                                               }
                                           })
                                           .name("increment generation");
            auto reportProgress = subflow
                                      .emplace([&, timer]() -> void {
                                          // Time and phase telemetry describe the work done even when the
                                          // generation is discarded; only the report callback is skipped.
                                          Elapsed() = computeElapsed();
                                          Timings() = timer->Timings();
                                          if (!generationCommitted) {
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
