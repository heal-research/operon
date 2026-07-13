// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm> // for stable_sort, copy_n, max
#include <chrono> // for steady_clock
#include <cmath> // for isfinite
#include <iterator> // for move_iterator, back_inse...
#include <limits> // for numeric_limits
#include <memory> // for allocator, allocator_tra...
#include <optional> // for optional
#include <taskflow/algorithm/for_each.hpp> // for taskflow.for_each_index
#include <vector> // for vector, vector::size_type

#include "operon/algorithms/nsga2.hpp"
#include "operon/algorithms/phase_timer.hpp"
#include "operon/core/contracts.hpp" // for ENSURE
#include "operon/core/problem.hpp" // for Problem
#include "operon/core/range.hpp" // for Range
#include "operon/core/tree.hpp" // for Tree
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
    auto eq = [eps](auto const& lhs, auto const& rhs) -> auto { return Operon::Equal {}(lhs.Fitness, rhs.Fitness, eps); };

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
    std::iota(order.begin(), order.end(), size_t{0});
    std::stable_sort(order.begin(), order.end(), [&](size_t a, size_t b) -> auto { return std::ranges::lexicographical_compare(pop[a].Fitness, pop[b].Fitness); });
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
        for (auto& idx : f) { idx = order[idx]; }
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
    std::transform(fronts_.front().begin(), fronts_.front().end(), std::back_inserter(best_), [&](auto i) -> auto { return pop[i]; });
}

auto NSGA2::Run(tf::Executor& executor, Operon::RandomGenerator& random, Operon::ReportCallback report, bool warmStart) -> void // NOLINT(readability-function-cognitive-complexity)
{
    auto const savedGeneration = Generation();
    Reset();
    if (warmStart) { Generation() = savedGeneration; }

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
        for (size_t i = 0; i < s; ++i) { rngs.emplace_back(random()); }
    }

    auto const* evaluator = generator->Evaluator();

    // we want to allocate all the memory that will be necessary for evaluation (e.g. for storing model responses)
    // in one go and use it throughout the generations in order to minimize the memory pressure
    auto trainSize = problem->TrainingRange().Size();

    ENSURE(executor.num_workers() > 0);
    std::vector<std::vector<Operon::Scalar>> slots(executor.num_workers());

    auto stop = [&]() -> bool {
        Elapsed() = computeElapsed();
        return StopRequested() || generator->Terminate() || Generation() == config.Generations || Elapsed() > static_cast<double>(config.TimeLimit);
    };

    auto& individuals = Individuals();
    auto parents = Parents();
    auto offspring = Offspring();
    std::vector<Operon::RandomGenerator> savedRngs; // used only on warm resume

    // while loop control flow
    auto timer = executor.make_observer<PhaseTimer>();

    tf::Taskflow taskflow;
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareEval = subflow.emplace([&]() -> void { evaluator->Prepare(parents); }).name("prepare evaluator");
            auto nonDominatedSort = subflow.emplace([&]() -> void { Sort(parents); }).name(std::string{SortTaskName});
            auto reportProgress = subflow.emplace([&, timer]() -> void {
                                             Timings() = timer->Timings();
                                             if (report && std::invoke(report)) { RequestStop(); }
                                         })
                                      .name("report progress");
            nonDominatedSort.precede(reportProgress);

            // nonDominatedSort runs after eval in both paths to rebuild fronts_ from current fitness.
            auto eval = subflow.for_each_index(size_t { 0 }, parents.size(), size_t { 1 }, [&](size_t i) -> void {
                                   auto const id = executor.this_worker_id();
                                   slots[id].resize(trainSize);
                                   parents[i].Fitness = (*evaluator)(rngs[i], parents[i], slots[id]);
                               })
                            .name("evaluate population");
            eval.precede(nonDominatedSort);

            if (IsFitted() && warmStart) {
                // Re-evaluate to catch evaluator/objective config mismatches, but snapshot and restore
                // the worker RNG states so that subsequent generations remain deterministic.
                auto saveRngs    = subflow.emplace([&]() { savedRngs = rngs; }).name("save rng states");
                auto restoreRngs = subflow.emplace([&]() { rngs = std::move(savedRngs); }).name("restore rng states");
                prepareEval.precede(saveRngs);
                saveRngs.precede(eval);
                eval.precede(restoreRngs);
                restoreRngs.precede(nonDominatedSort);
            } else {
                auto init = subflow.for_each_index(size_t { 0 }, parents.size(), size_t { 1 }, [&](size_t i) -> void {
                                       parents[i].Genotype = (*treeInit)(rngs[i]);
                                       (*coeffInit)(rngs[i], parents[i].Genotype);
                                   })
                                .name("initialize population");
                init.precede(prepareEval);
                prepareEval.precede(eval);
            }
        }, // init
        stop, // loop condition
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareGenerator = subflow.emplace([&]() -> void { generator->Prepare(parents); }).name("prepare generator");
            auto generateOffspring = subflow.for_each_index(size_t { 0 }, offspring.size(), size_t { 1 }, [&](size_t i) -> void {
                                                slots[executor.this_worker_id()].resize(trainSize);
                                                auto buf = Operon::Span<Operon::Scalar>(slots[executor.this_worker_id()]);
                                                while (!stop()) {
                                                    auto result = (*generator)(rngs[i], config.CrossoverProbability, config.MutationProbability, config.LocalSearchProbability, config.LamarckianProbability, buf);
                                                    if (result) {
                                                        offspring[i] = std::move(*result);
                                                        ENSURE(offspring[i].Genotype.Length() > 0);
                                                        return;
                                                    }
                                                }
                                            })
                                         .name("generate offspring");
            auto nonDominatedSort = subflow.emplace([&]() -> void { Sort(individuals); }).name(std::string{SortTaskName});
            // Delegates to the reinserter's actual merge strategy (same call
            // shape as GP, gp.cpp) instead of just truncating a globally
            // sorted array via ReinserterBase::Sort. This makes the
            // --reinserter choice (keep-best vs. replace-worst) a real,
            // meaningful knob for NSGA2 rather than a silently inert one -
            // note KeepBestReinserter's merge is a positional pairwise
            // tournament, not a global top-K selection, so this is not
            // guaranteed to reproduce the old truncation-based selection
            // byte-for-byte even at default settings.
            auto reinsert = subflow.emplace([&]() -> void { (*reinserter)(random, parents, offspring); }).name("reinsert");
            auto incrementGeneration = subflow.emplace([&]() -> void { ++Generation(); }).name("increment generation");
            auto reportProgress = subflow.emplace([&, timer]() -> void {
                                             Timings() = timer->Timings();
                                             if (report && std::invoke(report)) { RequestStop(); }
                                         }).name("report progress");

            // set-up subflow graph
            prepareGenerator.precede(generateOffspring);
            generateOffspring.precede(nonDominatedSort);
            nonDominatedSort.precede(reinsert);
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
    taskflow.name("NSGA2");

    init.precede(cond);
    cond.precede(body, done);
    body.precede(back);
    back.precede(cond);

    executor.run(taskflow).wait();
    Timings() = timer->Timings();
    executor.remove_observer(std::move(timer));
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
