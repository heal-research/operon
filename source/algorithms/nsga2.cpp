// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>                                 // for stable_sort, copy_n, max
#include <atomic>                                    // for atomic_bool
#include <chrono>                                    // for steady_clock
#include <cmath>                                     // for isfinite
#include <iterator>                                  // for move_iterator, back_inse...
#include <limits>                                    // for numeric_limits
#include <memory>                                    // for allocator, allocator_tra...
#include <optional>                                  // for optional
#include <taskflow/taskflow.hpp>                     // for taskflow, subflow
#include <vector>                                    // for vector, vector::size_type
#include <fmt/ranges.h>

#include "operon/algorithms/nsga2.hpp"
#include "operon/core/contracts.hpp"                 // for ENSURE
#include "operon/core/operator.hpp"                  // for OperatorBase
#include "operon/core/problem.hpp"                   // for Problem
#include "operon/core/range.hpp"                     // for Range
#include "operon/core/tree.hpp"                      // for Tree
#include "operon/operators/initializer.hpp"          // for CoefficientInitializerBase
#include "operon/operators/non_dominated_sorter.hpp" // for RankSorter
#include "operon/operators/reinserter.hpp"           // for ReinserterBase

namespace Operon {

auto NSGA2::UpdateDistance(Operon::Span<Individual> pop) -> void
{
    // assign distance. each front is sorted for each objective
    size_t m = pop.front().Fitness.size();
    auto inf = std::numeric_limits<Operon::Scalar>::max();
    for (size_t i = 0; i < fronts_.size(); ++i) {
        auto& front = fronts_[i];
        for (size_t obj = 0; obj < m; ++obj) {
            SingleObjectiveComparison comp(obj);
            std::stable_sort(front.begin(), front.end(), [&](auto a, auto b) { return comp(pop[a], pop[b]); });
            auto min = pop.front()[obj];
            auto max = pop.back()[obj];
            for (size_t j = 0; j < front.size(); ++j) {
                auto idx = front[j];

                pop[idx].Rank = i;
                if (obj == 0) {
                    pop[idx].Distance = 0;
                }

                auto mPrev = j > 0 ? pop[j - 1][obj] : inf;
                auto mNext = j < front.size() - 1 ? pop[j + 1][obj] : inf;
                auto distance = (mNext - mPrev) / (max - min);
                if (!std::isfinite(distance)) {
                    distance = 0;
                }
                pop[idx].Distance += distance;
            }
        }
    }
}

auto NSGA2::Sort(Operon::Span<Individual> pop) -> void
{
    auto eps = static_cast<Operon::Scalar>(GetConfig().Epsilon);
    auto less = [eps](auto const& lhs, auto const& rhs) { return Operon::Less{}(lhs.Fitness, rhs.Fitness, eps); };
    auto eq = [eps](auto const& lhs, auto const& rhs) { return Operon::Equal{}(lhs.Fitness, rhs.Fitness, eps); };
    // sort the population lexicographically
    std::stable_sort(pop.begin(), pop.end(), less);
    // mark the duplicates for stable_partition
    for(auto i = pop.begin(); i < pop.end(); ) {
        i->Rank = 0;
        auto j = i + 1;
        for (; j < pop.end() && eq(*i, *j); ++j) {
            j->Rank = 1;
        }
        i = j;
    }
    auto r = std::stable_partition(pop.begin(), pop.end(), [](auto const& ind) { return !ind.Rank; });
    Operon::Span<Operon::Individual const> uniq(pop.begin(), r);
    // do the sorting
    fronts_ = sorter_(uniq, eps);
    // sort the fronts for consistency between sorting algos
    for (auto& f : fronts_) {
        std::stable_sort(f.begin(), f.end());
    }
    // banish the duplicates into the last front
    if (r < pop.end()) {
        std::vector<size_t> last(pop.size() - uniq.size());
        std::iota(last.begin(), last.end(), uniq.size());
        fronts_.push_back(last);
    }
    // calculate crowding distance
    UpdateDistance(pop);
    // update best front
    best_.clear();
    std::transform(fronts_.front().begin(), fronts_.front().end(), std::back_inserter(best_), [&](auto i) { return pop[i]; });
    archive_.Insert(best_);
}

auto NSGA2::Run(tf::Executor& executor, Operon::RandomGenerator& random, std::function<void()> report) -> void
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

    auto const& evaluator = generator.Evaluator();

    // we want to allocate all the memory that will be necessary for evaluation (e.g. for storing model responses)
    // in one go and use it throughout the generations in order to minimize the memory pressure
    auto trainSize = problem.TrainingRange().Size();

    ENSURE(executor.num_workers() > 0);
    std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());

    tf::Taskflow taskflow;

    auto stop = [&]() {
        return generator.Terminate() || generation_ == config.Generations || elapsed() > static_cast<double>(config.TimeLimit);
    };

    // while loop control flow
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&](tf::Subflow& subflow) {
            auto init = subflow.for_each_index(size_t{0}, parents_.size(), size_t{1}, [&](size_t i) {
                parents_[i].Genotype = treeInit(rngs[i]);
                coeffInit(rngs[i], parents_[i].Genotype);
            }).name("initialize population");
            auto prepareEval = subflow.emplace([&]() { evaluator.Prepare(parents_); }).name("prepare evaluator");
            auto eval = subflow.for_each_index(size_t{0}, parents_.size(), size_t{1}, [&](size_t i) {
                auto id = executor.this_worker_id();
                // make sure the worker has a large enough buffer
                if (slots[id].size() < trainSize) {
                    slots[id].resize(trainSize);
                }
                parents_[i].Fitness = evaluator(rngs[i], parents_[i], slots[id]);
            }).name("evaluate population");
            auto nonDominatedSort = subflow.emplace([&]() { Sort(parents_); }).name("non-dominated sort");
            auto reportProgress = subflow.emplace([&]() { if (report) { std::invoke(report); } }).name("report progress");
            init.precede(prepareEval);
            prepareEval.precede(eval);
            eval.precede(nonDominatedSort);
            nonDominatedSort.precede(reportProgress);
        }, // init
        stop, // loop condition
        [&](tf::Subflow& subflow) {
            auto prepareGenerator = subflow.emplace([&]() { generator.Prepare(parents_); }).name("prepare generator");
            auto generateOffspring = subflow.for_each_index(size_t{0}, offspring_.size(), size_t{1}, [&](size_t i) {
                auto buf = Operon::Span<Operon::Scalar>(slots[executor.this_worker_id()]);
                while (!stop()) {
                    if (auto result = generator(rngs[i], config.CrossoverProbability, config.MutationProbability, buf); result.has_value()) {
                        offspring_[i] = std::move(result.value());
                        ENSURE(offspring_[i].Genotype.Length() > 0);
                        return;
                    }
                }
            }).name("generate offspring");
            auto nonDominatedSort = subflow.emplace([&]() { Sort(individuals_); }).name("non-dominated sort");
            auto reinsert = subflow.emplace([&]() { reinserter.Sort(individuals_); }).name("reinsert");
            auto incrementGeneration = subflow.emplace([&]() { ++generation_; }).name("increment generation");
            auto reportProgress = subflow.emplace([&]() { if (report) { std::invoke(report); } }).name("report progress");

            // set-up subflow graph
            prepareGenerator.precede(generateOffspring);
            generateOffspring.precede(nonDominatedSort);
            nonDominatedSort.precede(reinsert);
            reinsert.precede(incrementGeneration);
            incrementGeneration.precede(reportProgress);
        }, // loop body (evolutionary main loop)
        [&]() { return 0; }, // jump back to the next iteration
        [&]() { /* done nothing to do */ } // work done, report last gen and stop
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

    executor.run(taskflow);
    executor.wait_for_all();
}

auto NSGA2::Run(Operon::RandomGenerator& random, std::function<void()> report, size_t threads) -> void
{
    if (threads == 0U) {
        threads = std::thread::hardware_concurrency();
    }
    tf::Executor executor(threads);
    Run(executor, random, std::move(report));
}
} // namespace Operon
