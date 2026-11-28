// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef GA_BASE_HPP
#define GA_BASE_HPP

#include "config.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/generator.hpp"
#include "stoppable.hpp"
#include <atomic>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <operon/operon_export.hpp>
#include <optional>
#include <string>

namespace Operon {

// Evaluation counters of a run, split by who spent them. `Residual*`/`Jacobian*` are the units of the evaluation
// budget (Consumed()): `*Scoring` comes from evaluator scoring (EvaluatorBase::Stats()), `*Optimizer` from coefficient
// optimization (the run's OptimizerAccounting). Each pair is read from its own owner, never derived by subtraction.
// Calls (candidate scorings), OptimizerTimeUs (wall time summed over threads, not deterministic) and the cache
// counters (filled by DetailedStats(); nullopt means "no cache") never consume the budget.
struct EvaluationStats {
    std::size_t Calls {};
    std::size_t ResidualScoring {};
    std::size_t ResidualOptimizer {};
    std::size_t JacobianScoring {};
    std::size_t JacobianOptimizer {};
    std::size_t OptimizerTimeUs {};
    std::optional<std::size_t> CacheHits {};
    std::optional<std::size_t> CacheLookups {};

    [[nodiscard]] constexpr auto ResidualTotal() const -> std::size_t
    {
        return detail::SaturatingAdd(ResidualScoring, ResidualOptimizer);
    }
    [[nodiscard]] constexpr auto JacobianTotal() const -> std::size_t
    {
        return detail::SaturatingAdd(JacobianScoring, JacobianOptimizer);
    }
    [[nodiscard]] constexpr auto Consumed() const -> std::size_t
    {
        return detail::SaturatingAdd(ResidualTotal(), JacobianTotal());
    }

    [[nodiscard]] constexpr auto operator==(EvaluationStats const&) const -> bool = default;
};

// A generation attempt that was not committed. The counts describe the work executed before the workers stopped and can
// depend on the thread count. Why it was discarded is in GeneticAlgorithmBase::StopReasons().
struct OffspringProgress {
    size_t Attempted { 0 }; // offspring-generator invocations (accepted and rejected)
    size_t Completed { 0 }; // pool slots filled
    size_t PoolSize { 0 }; // pool slots to fill
};

// Budget accounting of the last Run(), in the units of the evaluation budget (see EvaluationStats::Consumed()).
// Committed work (initialization plus every committed generation) is all the budget limits, and does not depend on the
// thread count. Discarded work is what discarded generation attempts spent; Executed is everything counted since Run()
// started, including evaluator work done by a report callback. Both are informational and can depend on the thread
// count. Values saturate instead of wrapping. Reset by Reset() and at the start of Run(); refreshed after
// initialization, after every committed generation and when Run() returns.
struct BudgetAccounting {
    std::uint64_t Budget { 0 }; // GeneticAlgorithmConfig::Evaluations of the run
    std::uint64_t GenerationsCommitted { 0 }; // generations committed by this Run() (warm-start generations excluded)
    std::uint64_t EvaluationsCommitted { 0 }; // initialization (or warm-start re-evaluation) + committed generations
    std::uint64_t EvaluationsExecuted { 0 }; // everything counted since Run() started (scoring + optimizer)
    std::uint64_t EvaluationsDiscarded { 0 }; // spent by the generation attempts that were discarded
    std::uint64_t BudgetOvershoot { 0 }; // max(0, committed - budget): only initialization can overshoot

    [[nodiscard]] constexpr auto operator==(BudgetAccounting const&) const -> bool = default;
};

class Problem;
class ReinserterBase;
struct CoefficientInitializerBase;
struct TreeInitializerBase;

class GeneticAlgorithmBase : public StoppableAlgorithm {
public:
    ~GeneticAlgorithmBase() override = default;
    // Rebind the spans to the copied individuals rather than the source storage.
    GeneticAlgorithmBase(GeneticAlgorithmBase const& other)
        : StoppableAlgorithm(other)
        , config_(other.config_)
        , problem_(other.problem_)
        , treeInit_(other.treeInit_)
        , coeffInit_(other.coeffInit_)
        , generator_(other.generator_)
        , reinserter_(other.reinserter_)
        , individuals_(other.individuals_)
        , parents_(individuals_.data(), config_.PopulationSize)
        , offspring_(individuals_.data() + config_.PopulationSize, config_.PoolSize)
        , progress_(other.progress_)
        , workerRngs_(other.workerRngs_)
        , generation_(other.generation_)
        , elapsed_(other.elapsed_)
        , phaseTimes_(other.phaseTimes_)
        , isFitted_(other.isFitted_)
        , incompleteGeneration_(other.incompleteGeneration_)
        , optimizerAccounting_(other.optimizerAccounting_)
        , budgetReport_(other.budgetReport_)
    {
    }
    GeneticAlgorithmBase(GeneticAlgorithmBase&&) = delete;
    auto operator=(GeneticAlgorithmBase const& other) -> GeneticAlgorithmBase&
    {
        if (this == &other) {
            return *this;
        }
        StoppableAlgorithm::operator=(other);
        config_ = other.config_;
        problem_ = other.problem_;
        treeInit_ = other.treeInit_;
        coeffInit_ = other.coeffInit_;
        generator_ = other.generator_;
        reinserter_ = other.reinserter_;
        individuals_ = other.individuals_;
        parents_ = Operon::Span<Individual>(individuals_.data(), config_.PopulationSize);
        offspring_ = Operon::Span<Individual>(individuals_.data() + config_.PopulationSize, config_.PoolSize);
        workerRngs_ = other.workerRngs_;
        progress_ = other.progress_;
        generation_ = other.generation_;
        elapsed_ = other.elapsed_;
        phaseTimes_ = other.phaseTimes_;
        isFitted_ = other.isFitted_;
        incompleteGeneration_ = other.incompleteGeneration_;
        optimizerAccounting_ = other.optimizerAccounting_;
        budgetReport_ = other.budgetReport_;
        return *this;
    }
    auto operator=(GeneticAlgorithmBase&&) -> GeneticAlgorithmBase& = delete;

    GeneticAlgorithmBase(GeneticAlgorithmConfig config, gsl::not_null<Problem const*> problem,
        gsl::not_null<TreeInitializerBase const*> treeInit, gsl::not_null<CoefficientInitializerBase const*> coeffInit,
        gsl::not_null<OffspringGeneratorBase const*> generator, gsl::not_null<ReinserterBase const*> reinserter)
        : config_(config)
        , problem_(problem)
        , treeInit_(treeInit)
        , coeffInit_(coeffInit)
        , generator_(generator)
        , reinserter_(reinserter)
        , individuals_(config.PopulationSize + config.PoolSize)
        , parents_(individuals_.data(), config.PopulationSize)
        , offspring_(individuals_.data() + config.PopulationSize, config.PoolSize)
    {
        generator_->SetCache(config.Cache);
    }

    // Const access returns spans of const elements; const spans alone are insufficient.
    // Mutable rvalues are rejected; const rvalues can still produce dangling spans.
    template <typename Self>
    [[nodiscard]] auto Parents(this Self& self)
        -> Operon::Span<std::conditional_t<std::is_const_v<Self>, Individual const, Individual>>
    {
        return self.parents_;
    }

    template <typename Self>
    [[nodiscard]] auto Offspring(this Self& self)
        -> Operon::Span<std::conditional_t<std::is_const_v<Self>, Individual const, Individual>>
    {
        return self.offspring_;
    }

    // Preserve reference qualification when forwarding access to storage.
    template <typename Self> [[nodiscard]] auto Individuals(this Self&& self) -> auto&&
    {
        return std::forward<Self>(self).individuals_;
    }

    [[nodiscard]] auto GetProblem() const -> const Problem* { return problem_.get(); }
    [[nodiscard]] auto GetConfig() const -> GeneticAlgorithmConfig { return config_; }

    // The run-owned accounting of coefficient-optimizer work: every local search of this algorithm's run
    // (initialization and offspring generation, whether the candidate is kept, rejected or discarded) is recorded
    // here, and nowhere else. Run() passes it explicitly to ScoreIndividual and to the offspring generator; a
    // custom driver must do the same. It is reset together with the algorithm at the start of Run().
    template <typename Self> [[nodiscard]] auto&& Accounting(this Self&& self)
    {
        return std::forward<Self>(self).optimizerAccounting_;
    }

    // The combined run report: evaluation counters of the evaluator tree (scoring only), the run's optimizer
    // accounting, and the transposition-cache counters (nullopt when no cache is active, which is not the same as
    // zero). The active cache is the generator's (the pointer it actually looks up; initially config.Cache, see the
    // constructor), which SetCache may have changed since. The scoring and optimizer fields are disjoint, each read
    // from its own owner; a live report is a non-atomic cross-counter snapshot, one taken after the run's workers
    // finished is exact. Reset together with the algorithm at the start of Run().
    [[nodiscard]] auto DetailedStats() const -> EvaluationStats
    {
        auto const [residual, jacobian, calls] = generator_->Evaluator()->Stats();
        auto const optimizer = optimizerAccounting_.Snapshot();
        EvaluationStats stats {
            .Calls = calls,
            .ResidualScoring = residual,
            .ResidualOptimizer = static_cast<std::size_t>(optimizer.Residual),
            .JacobianScoring = jacobian,
            .JacobianOptimizer = static_cast<std::size_t>(optimizer.Jacobian),
            .OptimizerTimeUs = static_cast<std::size_t>(optimizer.TimeUs),
        };
        if (auto const* cache = generator_->Cache()) {
            stats.CacheHits = cache->Hits();
            stats.CacheLookups = cache->Lookups();
        }
        return stats;
    }

    // Offspring accounting of the generation body started last: generator invocations, filled pool slots and pool
    // size (see OffspringProgress). Exact whenever no generation body is running, e.g. in the report callback and
    // after Run() returns. IncompleteGeneration() is the part of it that was cut short.
    [[nodiscard]] auto Progress() const -> OffspringProgress
    {
        return { progress_.Attempted.load(std::memory_order_relaxed),
            progress_.Completed.load(std::memory_order_relaxed), progress_.PoolSize };
    }

    // Index of the last generation whose pool was filled (initialization = 0). A generation that stopped before its
    // pool was full is discarded and not counted, so this is Generation().
    [[nodiscard]] auto GenerationsCompleted() const -> std::size_t { return generation_; }

    // The stop conditions that follow from the committed state and the clock: generation limit, evaluation budget
    // (committed work reached a positive budget) and time limit. The limits a discarded or cap-crossing generation
    // exceeds are recorded by the drivers, not derived here. Does not record anything.
    [[nodiscard]] auto StopConditions(double elapsed) const -> StopMask
    {
        StopMask mask {};
        if (generation_ >= config_.Generations) {
            mask |= MaskOf(StopReason::GenerationLimit);
        }
        auto const& budget = budgetReport_;
        if (budget.EvaluationsCommitted > budget.Budget
            || (budget.EvaluationsCommitted == budget.Budget && budget.Budget != 0)) {
            mask |= MaskOf(StopReason::EvaluationLimit);
        }
        if (elapsed > static_cast<double>(config_.TimeLimit)) {
            mask |= MaskOf(StopReason::TimeLimit);
        }
        return mask;
    }

    // The reasons to report once Run() has returned: everything recorded while it ran, plus every condition that
    // holds in the final state (at the last published Elapsed()).
    [[nodiscard]] auto TerminationReasons() const -> StopMask { return StopReasons() | StopConditions(elapsed_); }

    [[nodiscard]] auto GetTreeInitializer() const -> TreeInitializerBase const* { return treeInit_.get(); }
    [[nodiscard]] auto GetCoefficientInitializer() const -> CoefficientInitializerBase const*
    {
        return coeffInit_.get();
    }
    [[nodiscard]] auto GetGenerator() const -> OffspringGeneratorBase const* { return generator_.get(); }
    [[nodiscard]] auto GetReinserter() const -> ReinserterBase const* { return reinserter_.get(); }

    template <typename Self> [[nodiscard]] auto&& WorkerRngs(this Self&& self)
    {
        return std::forward<Self>(self).workerRngs_;
    }

    // Return const scalars by value and mutable scalars by reference.
    template <typename Self>
    [[nodiscard]] auto Generation(this Self&& self)
        -> std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, size_t, size_t&>
    {
        return self.generation_;
    } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <typename Self>
    [[nodiscard]] auto Elapsed(this Self&& self)
        -> std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, double, double&>
    {
        return self.elapsed_;
    } // NOLINT(cppcoreguidelines-missing-std-forward)

    template <typename Self> [[nodiscard]] auto Timings(this Self&& self) -> auto&&
    {
        return std::forward<Self>(self).phaseTimes_;
    }

    template <typename Self>
    [[nodiscard]] auto IsFitted(this Self&& self)
        -> std::conditional_t<std::is_const_v<std::remove_reference_t<Self>>, bool, bool&>
    {
        return self.isFitted_;
    } // NOLINT(cppcoreguidelines-missing-std-forward)

    // Budget accounting of the last Run() (see BudgetAccounting); all zero before the first Run(). Not to be confused
    // with Accounting(), the run's coefficient-optimizer accounting.
    [[nodiscard]] auto BudgetReport() const -> BudgetAccounting const& { return budgetReport_; }

    // Progress of the last generation attempt that was not committed; nullopt if it was committed. Such a generation
    // is neither reinserted, counted in Generation() nor reported; Parents() and the worker RNG states are those of the
    // last committed generation. It is discarded when its pool was not filled (time limit, stop request), when it spent
    // more than the budget has left or when it made more attempts than MaxAttempts() (the pool may be full then).
    // A committed generation that crossed the attempt cap leaves this empty and records PressureLimit.
    [[nodiscard]] auto IncompleteGeneration() const -> std::optional<OffspringProgress> const&
    {
        return incompleteGeneration_;
    }
    auto SetIncompleteGeneration(std::optional<OffspringProgress> progress) -> void
    {
        incompleteGeneration_ = progress;
    }

    // StopRequested()/RequestStop() are inherited from StoppableAlgorithm.

    // Call only between runs; the active observer retains its own totals.
    auto Reset() -> void
    {
        generation_ = 0;
        incompleteGeneration_.reset();
        budgetReport_ = {};
        elapsed_ = 0;
        ClearStopRequested();
        phaseTimes_.clear();
        progress_ = {};
        GetGenerator()->Evaluator()->Reset();
        optimizerAccounting_.Reset();
        if (auto* cache = generator_->Cache()) {
            cache->ResetCounters();
        }
    }

    auto RestoreIndividuals(std::vector<Individual> inds) -> void
    {
        EXPECT(inds.size() == config_.PoolSize + config_.PopulationSize,
            "Mismatched number of individuals (must match pool/population sizes)");
        individuals_ = std::move(inds);
        parents_ = Operon::Span<Individual>(individuals_.data(), config_.PopulationSize);
        offspring_ = Operon::Span<Individual>(individuals_.data() + config_.PopulationSize, config_.PoolSize);
    }

protected:
    // Published by the drivers only: it is an input of StopConditions(), so a report callback must not set it.
    auto SetBudgetReport(BudgetAccounting const& report) -> void { budgetReport_ = report; }

    // Offspring accounting for Run(): begin a generation body, then add each worker's slot (the generator
    // invocations it needed and whether it filled the slot; thread-safe).
    auto BeginOffspringProgress() -> void
    {
        progress_.Attempted.store(0, std::memory_order_relaxed);
        progress_.Completed.store(0, std::memory_order_relaxed);
        progress_.PoolSize = offspring_.size();
    }
    auto CountOffspring(std::size_t attempts, bool completed) -> void
    {
        progress_.Attempted.fetch_add(attempts, std::memory_order_relaxed);
        if (completed) {
            progress_.Completed.fetch_add(1, std::memory_order_relaxed);
        }
    }

    using Clock = std::chrono::steady_clock;
    [[nodiscard]] static auto SecondsSince(Clock::time_point start) -> double
    {
        return std::chrono::duration<double>(Clock::now() - start).count();
    }

    // Polled by the offspring-generation workers: a stop was requested or the time limit elapsed (recorded as
    // TimeLimit). Writes only the atomic stop mask.
    [[nodiscard]] auto PollStop(Clock::time_point start) -> bool
    {
        if (StopRequested()) {
            return true;
        }
        if (SecondsSince(start) > static_cast<double>(config_.TimeLimit)) {
            RecordStop(MaskOf(StopReason::TimeLimit));
            return true;
        }
        return false;
    }

    // Loop condition, called from a single task between generations: publishes Elapsed(), records the stop conditions
    // that hold, and returns true once any stop reason is recorded.
    [[nodiscard]] auto EndOfIteration(Clock::time_point start) -> bool
    {
        elapsed_ = SecondsSince(start);
        RecordStop(StopConditions(elapsed_));
        return StopReasons() != 0;
    }

private:
    GeneticAlgorithmConfig config_;

    gsl::not_null<Problem const*> problem_;
    gsl::not_null<TreeInitializerBase const*> treeInit_;
    gsl::not_null<CoefficientInitializerBase const*> coeffInit_;
    gsl::not_null<OffspringGeneratorBase const*> generator_;
    gsl::not_null<ReinserterBase const*> reinserter_;

    Operon::Vector<Individual> individuals_;
    Operon::Span<Individual> parents_;
    Operon::Span<Individual> offspring_;

    struct ProgressState {
        std::atomic_size_t Attempted { 0 };
        std::atomic_size_t Completed { 0 };
        std::size_t PoolSize { 0 };

        ProgressState() = default;
        ProgressState(ProgressState const& other)
            : Attempted(other.Attempted.load(std::memory_order_relaxed))
            , Completed(other.Completed.load(std::memory_order_relaxed))
            , PoolSize(other.PoolSize)
        {
        }
        auto operator=(ProgressState const& other) -> ProgressState&
        {
            Attempted.store(other.Attempted.load(std::memory_order_relaxed), std::memory_order_relaxed);
            Completed.store(other.Completed.load(std::memory_order_relaxed), std::memory_order_relaxed);
            PoolSize = other.PoolSize;
            return *this;
        }
    };
    ProgressState progress_;

    std::vector<Operon::RandomGenerator> workerRngs_;
    size_t generation_ { 0 };
    double elapsed_ { 0 };
    Operon::Map<std::string, double> phaseTimes_;
    bool isFitted_ { false };
    std::optional<OffspringProgress> incompleteGeneration_;
    OptimizerAccounting optimizerAccounting_;
    BudgetAccounting budgetReport_;
};

} // namespace Operon

#endif
