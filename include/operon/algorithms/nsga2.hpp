// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_NSGA2_HPP
#define OPERON_NSGA2_HPP

#include "algorithms/config.hpp"
#include "core/format.hpp"
#include "core/metrics.hpp"
#include "core/types.hpp"
#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/generator.hpp"
#include "operators/initializer.hpp"
#include "operators/mutation.hpp"

#include <chrono>
#include <thread>

#include "taskflow/taskflow.hpp"

namespace Operon {

namespace detail {

    template<size_t N>
    Operon::Vector<int> ComputeRanks(Operon::Span<Individual> pop) {
        Operon::Vector<int> ranks(pop.size(), 0);
        for (size_t i = 0; i < pop.size() - 1; ++i) {
            for (size_t j = i + 1; j < pop.size(); ++j) {
                auto d = pop[i].Compare<N>(pop[j]);
                if (d == DominanceResult::Equality) {
                    // duplicate points are banished to the last pareto front
                    ranks[i] = (int)pop.size();
                    continue;
                }
                ranks[i] += d == DominanceResult::RightDominates;
                ranks[j] += d == DominanceResult::LeftDominates;
            }
        }
        return ranks;
    }
}

template <typename TInitializer>
class NSGA2 {
private:
    std::reference_wrapper<const Problem> problem_;
    std::reference_wrapper<const GeneticAlgorithmConfig> config_;

    std::reference_wrapper<const TInitializer> initializer_;
    std::reference_wrapper<const OffspringGeneratorBase> generator_;
    std::reference_wrapper<const ReinserterBase> reinserter_;

    Operon::Vector<Individual> individuals;
    Operon::Span<Individual> parents;
    Operon::Span<Individual> offspring;

    size_t generation;
    std::vector<Operon::Vector<int>> fronts;

    // best pareto front
    Operon::Vector<Individual> best;
    Operon::Vector<int> ComputeRanks(tf::Executor&, Operon::Span<Individual> pop) {
        auto n = pop.front().Size();
        if (n == 2) {
            return detail::ComputeRanks<2>(pop);
        }
        return detail::ComputeRanks<0>(pop);
    }

    void NonDominatedSort(tf::Executor& executor, Operon::Span<Individual> pop)
    {
        auto ranks = ComputeRanks(executor, pop);
        std::vector<int> indices(ranks.size());
        std::iota(indices.begin(), indices.end(), 0);
        pdqsort(indices.begin(), indices.end(), [&](auto i, auto j) { return ranks[i] < ranks[j]; });

        auto r = ranks[indices.front()];
        auto it0 = indices.begin();
        fronts.clear();
        while(true) {
            auto it = std::partition_point(it0, indices.end(), [&](auto i) { return ranks[i] == r; });
            Operon::Vector<int> front(it0, it);
            fronts.push_back(front);
            if(it == indices.end()) {
                break;
            }
            it0 = it;
            r = ranks[*it];
        }

        // assign distance. each front is sorted for each objective
        size_t n = pop.front().Fitness.size();
        auto inf = Operon::Numeric::Max<Operon::Scalar>();
        for (size_t i = 0; i < fronts.size(); ++i) {
            auto& front = fronts[i];
            for (size_t obj = 0; obj < n; ++obj) {
                pdqsort(front.begin(), front.end(), [&](auto a, auto b) { return pop[a][obj] < pop[b][obj]; });
                auto min = pop.front()[obj];
                auto max = pop.back()[obj];
                for (size_t j = 0; j < front.size(); ++j) {
                    auto idx = front[j];

                    pop[idx].Rank = i;
                    if (obj == 0) pop[idx].Distance = 0;

                    auto mPrev = j > 0 ? pop[j-1][obj] : inf;
                    auto mNext = j < front.size()-1 ? pop[j+1][obj] : inf;
                    auto distance = (mNext - mPrev) / (max - min);
                    if (!std::isfinite(distance)) {
                        distance = 0;
                    }
                    pop[idx].Distance += distance;
                }
            }
        }
        best.clear();
        std::transform(fronts.front().begin(), fronts.front().end(), std::back_inserter(best), [&](auto i) { return pop[i]; });
    }

    public:
        explicit NSGA2(Problem const& problem, GeneticAlgorithmConfig const& config, TInitializer const& initializer, OffspringGeneratorBase const& generator, ReinserterBase const& reinserter)
            : problem_(problem)
            , config_(config)
            , initializer_(initializer)
            , generator_(generator)
            , reinserter_(reinserter)
            , individuals(config.PopulationSize + config.PoolSize)
            , parents(individuals.data(), config.PopulationSize)
            , offspring(individuals.data() + config.PopulationSize, config.PoolSize)
            , generation(0UL)
        {
        }

        Operon::Span<Individual const> Parents() const { return { parents.data(), parents.size() }; }
        Operon::Span<Individual const> Offspring() const { return { offspring.data(), offspring.size() }; }

        Operon::Span<Individual const> Best() const { return { best.data(), best.size() }; }

        const Problem& GetProblem() const { return problem_.get(); }
        const GeneticAlgorithmConfig& GetConfig() const { return config_.get(); }

        const TInitializer& GetInitializer() const { return initializer_.get(); }
        const OffspringGeneratorBase& GetGenerator() const { return generator_.get(); }
        const ReinserterBase& GetReinserter() const { return reinserter_.get(); }

        size_t Generation() const { return generation; }

        void Reset()
        {
            generation = 0;
            GetGenerator().Evaluator().Reset();
        }

        void Run(Operon::RandomGenerator & random, std::function<void()> report = nullptr, size_t threads = 0)
        {
            if (!threads) {
                threads = std::thread::hardware_concurrency();
            }
            tf::Executor executor(threads);
            Run(executor, random, report);
        }

        void Run(tf::Executor & executor, Operon::RandomGenerator & random, std::function<void()> report = nullptr)
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

            //auto idx = 0;
            auto const& evaluator = generator.Evaluator();

            // we want to allocate all the memory that will be necessary for evaluation (e.g. for storing model responses)
            // in one go and use it throughout the generations in order to minimize the memory pressure
            auto trainSize = problem.TrainingRange().Size();

            ENSURE(executor.num_workers() > 0);
            std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());

            tf::Taskflow taskflow;

            std::atomic_bool terminate { false }; // flag to signal algorithm termination

            // while loop control flow
            auto [init, cond, body, back, done] = taskflow.emplace(
                [&](tf::Subflow& subflow) {
                    auto init = subflow.for_each_index(0ul, parents.size(), 1ul, [&](size_t i) {
                        auto id = executor.this_worker_id();
                        // make sure the worker has a large enough buffer
                        if (slots[id].size() < trainSize) {
                            slots[id].resize(trainSize);
                        }
                        parents[i].Genotype = initializer(rngs[i]);
                        parents[i].Fitness = evaluator(rngs[i], parents[i], slots[id]);
                    });
                    auto updateRanks = subflow.emplace([&]() { NonDominatedSort(executor, parents); });
                    init.precede(updateRanks);
                }, // init
                [&]() { return terminate || generation == config.Generations; }, // loop condition
                [&](tf::Subflow& subflow) {
                    auto prepareGenerator = subflow.emplace([&]() { generator.Prepare(parents); });
                    auto generateOffspring = subflow.for_each_index(0ul, offspring.size(), 1ul, [&](size_t i) {
                        auto buf = Operon::Span<Operon::Scalar>(slots[executor.this_worker_id()]);
                        while (!(terminate = generator.Terminate())) {
                            if (auto result = generator(rngs[i], config.CrossoverProbability, config.MutationProbability, buf); result.has_value()) {
                                offspring[i] = std::move(result.value());
                                return;
                            }
                        }
                    });
                    auto nonDominatedSort = subflow.emplace([&]() { NonDominatedSort(executor, individuals); });
                    auto reinsert = subflow.emplace([&]() { reinserter.Sort(individuals); });
                    //auto reinsert = subflow.sort(individuals.begin(), individuals.end(), CrowdedComparison{});
                    auto incrementGeneration = subflow.emplace([&]() { ++generation; });
                    auto reportProgress = subflow.emplace([&]() { if (report) std::invoke(report); });

                    // set-up subflow graph
                    prepareGenerator.precede(generateOffspring);
                    generateOffspring.precede(nonDominatedSort);
                    nonDominatedSort.precede(reinsert);
                    reinsert.precede(incrementGeneration);
                    incrementGeneration.precede(reportProgress);
                }, // loop body (evolutionary main loop)
                [&]() { return 0; }, // jump back to the next iteration
                [&]() { if(report) std::invoke(report); } // work done, report last gen and stop
            ); // evolutionary loop

            init.precede(cond);
            cond.precede(body, done);
            body.precede(back);
            back.precede(cond);

            executor.run(taskflow);
            executor.wait_for_all();
        }
    };
} // namespace operon

#endif
