// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <chrono>
#include <iterator>
#include <numeric>
#include <ranges>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>
#include <vector>

#include "operon/algorithms/mo_base.hpp"
#include "operon/algorithms/phase_timer.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/operator.hpp"
#include "operon/core/problem.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"

namespace Operon {

auto MultiObjectiveGABase::Sort(Operon::Span<Individual> pop) -> void
{
    auto config = GetConfig();
    auto eps = static_cast<Operon::Scalar>(config.Epsilon);
    auto eq = [eps](auto const& lhs, auto const& rhs) -> auto { return Operon::Equal{}(lhs.Fitness, rhs.Fitness, eps); };
    std::stable_sort(pop.begin(), pop.end(), [](auto const& a, auto const& b) -> auto { return std::ranges::lexicographical_compare(a.Fitness, b.Fitness); });
    for (auto i = pop.begin(); i < pop.end();) {
        i->Rank = 0;
        auto j = i + 1;
        for (; j < pop.end() && eq(*i, *j); ++j) {
            j->Rank = 1;
        }
        i = j;
    }
    auto r = std::stable_partition(pop.begin(), pop.end(), [](auto const& ind) -> auto { return !ind.Rank; });
    Operon::Span<Operon::Individual const> const uniq(pop.begin(), r);
    fronts_ = (*sorter_)(uniq, eps);
    for (auto& f : fronts_) {
        std::stable_sort(f.begin(), f.end());
    }
    if (r < pop.end()) {
        Operon::Vector<size_t> last(pop.size() - uniq.size());
        std::iota(last.begin(), last.end(), uniq.size());
        fronts_.push_back(last);
    }
    UpdateDistance(pop);
    best_.clear();
    std::transform(fronts_.front().begin(), fronts_.front().end(), std::back_inserter(best_), [&](auto i) -> auto { return pop[i]; });
}

auto MultiObjectiveGABase::Run(tf::Executor& executor, Operon::RandomGenerator& random, std::function<void()> report, bool warmStart) -> void // NOLINT(readability-function-cognitive-complexity)
{
    Reset();

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

    size_t const s = std::max(config.PopulationSize, config.PoolSize);
    std::vector<Operon::RandomGenerator> rngs;
    rngs.reserve(s);
    for (size_t i = 0; i < s; ++i) {
        rngs.emplace_back(random());
    }

    auto const* evaluator = generator->Evaluator();
    auto trainSize = problem->TrainingRange().Size();

    ENSURE(executor.num_workers() > 0);
    std::vector<std::vector<Operon::Scalar>> slots(executor.num_workers());

    auto stop = [&]() -> bool {
        Elapsed() = computeElapsed();
        return generator->Terminate() || Generation() == config.Generations || Elapsed() > static_cast<double>(config.TimeLimit);
    };

    auto& individuals = Individuals();
    auto parents = Parents();
    auto offspring = Offspring();

    auto timer = executor.make_observer<PhaseTimer>();

    tf::Taskflow taskflow;
    auto [init, cond, body, back, done] = taskflow.emplace(
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareEval = subflow.emplace([&]() -> void { evaluator->Prepare(parents); }).name("prepare evaluator");
            auto eval = subflow.for_each_index(size_t{0}, parents.size(), size_t{1}, [&](size_t i) -> void {
                auto const id = executor.this_worker_id();
                slots[id].resize(trainSize);
                parents[i].Fitness = (*evaluator)(rngs[i], parents[i], slots[id]);
            }).name("evaluate population");
            auto nonDominatedSort = subflow.emplace([&]() -> void { Sort(parents); }).name(std::string{SortTaskName});
            auto reportProgress = subflow.emplace([&, timer]() -> void {
                Timings() = timer->Timings();
                if (report) { std::invoke(report); }
            }).name("report progress");
            prepareEval.precede(eval);
            eval.precede(nonDominatedSort);
            nonDominatedSort.precede(reportProgress);

            if (!(IsFitted() && warmStart)) {
                auto initPop = subflow.for_each_index(size_t{0}, parents.size(), size_t{1}, [&](size_t i) -> void {
                    parents[i].Genotype = (*treeInit)(rngs[i]);
                    (*coeffInit)(rngs[i], parents[i].Genotype);
                }).name("initialize population");
                initPop.precede(prepareEval);
            }
        },
        stop,
        [&, timer](tf::Subflow& subflow) -> void {
            auto prepareGenerator = subflow.emplace([&]() -> void { generator->Prepare(parents); }).name("prepare generator");
            auto generateOffspring = subflow.for_each_index(size_t{0}, offspring.size(), size_t{1}, [&](size_t i) -> void {
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
            }).name("generate offspring");
            auto nonDominatedSort = subflow.emplace([&]() -> void { Sort(individuals); }).name(std::string{SortTaskName});
            auto reinsert = subflow.emplace([&]() -> void { reinserter->Sort(individuals); }).name("reinsert");
            auto incrementGeneration = subflow.emplace([&]() -> void { ++Generation(); }).name("increment generation");
            auto reportProgress = subflow.emplace([&, timer]() -> void {
                Timings() = timer->Timings();
                if (report) { std::invoke(report); }
            }).name("report progress");

            prepareGenerator.precede(generateOffspring);
            generateOffspring.precede(nonDominatedSort);
            nonDominatedSort.precede(reinsert);
            reinsert.precede(incrementGeneration);
            incrementGeneration.precede(reportProgress);
        },
        [&]() -> int { return 0; },
        [&]() -> void { IsFitted() = true; }
    );

    init.name("init");
    cond.name("termination");
    body.name("main loop");
    back.name("back");
    done.name("done");
    taskflow.name("MultiObjectiveGP");

    init.precede(cond);
    cond.precede(body, done);
    body.precede(back);
    back.precede(cond);

    executor.run(taskflow).wait();
    Timings() = timer->Timings();
    executor.remove_observer(std::move(timer));
}

auto MultiObjectiveGABase::Run(Operon::RandomGenerator& random, std::function<void()> report, size_t threads, bool warmStart) -> void
{
    if (threads == 0U) {
        threads = std::thread::hardware_concurrency();
    }
    tf::Executor executor(threads);
    Run(executor, random, std::move(report), warmStart);
}

} // namespace Operon
