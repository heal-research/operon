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
#include <cstddef>
#include <cstdint>
#include <operon/operon_export.hpp>
#include <optional>
#include <string>

namespace Operon {

// Evaluation counters of a run, split by who spent them. `Residual*`/`Jacobian*` count the units the evaluation
// budget is made of: `*Scoring` is spent by evaluator scoring (the prediction pass of Evaluate, a Jacobian taken by
// a likelihood scorer, ...; EvaluatorBase::Stats()), `*Optimizer` by coefficient optimization (recorded by the run
// in its OptimizerAccounting, from the per-fit FitOutcome diagnostics). Each pair is read from its own owner: nothing
// is derived by subtracting one from the other. Budget consumption is `Consumed()`; calls, optimizer time and cache
// counters never consume it. Calls is the number of evaluator operator() calls (candidate scorings) and
// OptimizerTimeUs the wall time spent in the optimizer, summed over threads (not deterministic, and 0 for a fit
// faster than the clock resolution). The cache counters are filled by GeneticAlgorithmBase::DetailedStats() when a
// transposition cache is configured (nullopt means "no cache", not zero).
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

// Why a generation attempt was not committed. `None` is the neutral value of a default-constructed
// OffspringProgress. ResourceLimit means the generation went over the evaluation budget and/or over the generator's
// attempt cap; the decision itself is deterministic (the same committed state gives the same decision at every
// thread count) but which of the two limits the *executed* work was seen to exceed is not when both are finite, see
// OffspringProgress. TimeLimit and StopRequested are asynchronous and are reported only if no resource limit was
// observed; StopRequested takes precedence over TimeLimit.
enum class DiscardReason : std::uint8_t {
    None,
    ResourceLimit, // the generation spent more than the budget has left and/or made more attempts than the cap
    TimeLimit, // the time limit expired before the offspring pool was filled
    StopRequested, // RequestStop() (or the report callback) ended the generation before the pool was filled
};

// A generation attempt that was not committed (informational: it describes the work executed before the workers
// stopped, which can depend on the thread count). Attempted, Completed, BudgetExceeded and AttemptCapExceeded are
// thread-count dependent in general. BudgetExceeded / AttemptCapExceeded say which limit the executed work was
// observed to exceed: they are deterministic when only one of the two limits is finite (finite = a budget below
// unlimited, a generator with MaxAttempts()), because the workers then stop on that limit alone; when both are
// finite, work that was in flight when the first one tripped may or may not have tripped the other, so the flags
// depend on the thread count. Reason is still ResourceLimit then.
struct OffspringProgress {
    size_t Attempted { 0 }; // offspring-generator invocations (accepted and rejected)
    size_t Completed { 0 }; // pool slots filled
    size_t PoolSize { 0 }; // pool slots to fill
    DiscardReason Reason { DiscardReason::None };
    bool BudgetExceeded { false }; // the executed work spent more than the budget had left
    bool AttemptCapExceeded { false }; // the executed work made more attempts than the generator's cap
};

// Why the last Run() ended. Deterministic (the same at every thread count) except for TimeLimit and StopRequested.
// If several conditions hold at once, the first of this list is reported (ResourceLimit and AttemptCap are recorded
// when they occur, so they come before the loop-condition checks).
enum class RunEnd : std::uint8_t {
    None, // no Run() yet, or it is still running
    ResourceLimit, // a generation was discarded for budget and/or attempt cap (IncompleteGeneration() is set)
    AttemptCap, // a committed generation made more attempts than the cap and ended the run (nothing discarded)
    GenerationLimit, // Generation() reached GeneticAlgorithmConfig::Generations
    BudgetSpent, // committed work reached a positive budget (initialization alone may also do so)
    StopRequested, // RequestStop(), or the report callback returned true
    TimeLimit, // GeneticAlgorithmConfig::TimeLimit expired
};

// Budget accounting of the last Run(), in the units of the evaluation budget (residual + Jacobian evaluations, see
// EvaluationStats::Consumed()). Committed work is the initialization plus every committed generation; the budget
// limits committed work only. Discarded work is what the generation attempts that were discarded spent (before the
// workers noticed that they could not commit): it is informational and can depend on the thread count, like
// Executed, which is everything counted since Run() started by the evaluator tree (scoring) and by the run's
// OptimizerAccounting: committed and discarded work, and anything else that used the evaluator, e.g. a report
// callback. Committed work does not depend on the thread count.
// All values saturate instead of wrapping. Reset by Reset() and at the start of Run(); refreshed after
// initialization, after every committed generation and when Run() returns.
struct BudgetAccounting {
    std::uint64_t Budget { 0 }; // GeneticAlgorithmConfig::Evaluations of the run
    std::uint64_t GenerationsCommitted { 0 }; // generations committed by this Run() (warm-start generations excluded)
    std::uint64_t EvaluationsCommitted { 0 }; // initialization (or warm-start re-evaluation) + committed generations
    std::uint64_t EvaluationsExecuted { 0 }; // everything counted since Run() started (scoring + optimizer)
    std::uint64_t EvaluationsDiscarded { 0 }; // spent by the generation attempts that were discarded
    std::uint64_t BudgetOvershoot { 0 }; // max(0, committed - budget): only initialization can overshoot
    RunEnd End { RunEnd::None }; // why Run() ended; None while it is running

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

    // Every stop condition that holds in the current state, derived from what the run recorded: the generation limit;
    // the evaluation budget (committed work reached a positive budget, or a generation was discarded for spending
    // more than the budget had left); selection pressure (a generation was discarded for exceeding the generator's
    // attempt cap, or a committed generation crossed it and ended the run, see BudgetReport().End); and the time
    // limit. Exact between generations and after Run() returned. Does not record anything. A custom generator's
    // Terminate() is deprecated and not consulted (see OffspringGeneratorBase::Terminate()), so the engine never
    // reports StopReason::GeneratorTermination.
    [[nodiscard]] auto StopConditions(double elapsed) const -> StopMask
    {
        StopMask mask {};
        if (generation_ >= config_.Generations) {
            mask |= MaskOf(StopReason::GenerationLimit);
        }
        auto const& budget = budgetReport_;
        auto const spent = budget.EvaluationsCommitted > budget.Budget
            || (budget.EvaluationsCommitted == budget.Budget && budget.Budget != 0);
        auto const discarded = incompleteGeneration_ && incompleteGeneration_->Reason == DiscardReason::ResourceLimit;
        if (spent || (discarded && incompleteGeneration_->BudgetExceeded)) {
            mask |= MaskOf(StopReason::EvaluationLimit);
        }
        if (budget.End == RunEnd::AttemptCap || (discarded && incompleteGeneration_->AttemptCapExceeded)) {
            mask |= MaskOf(StopReason::PressureLimit);
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

    // Budget accounting of the last Run() (see BudgetAccounting). All zero before the first Run(). Not to be confused
    // with Accounting(), the run's coefficient-optimizer accounting: this one reports committed, discarded and
    // executed work in units of the evaluation budget (scoring plus optimizer work) and why the run ended.
    [[nodiscard]] auto BudgetReport() const -> BudgetAccounting const& { return budgetReport_; }
    auto SetBudgetReport(BudgetAccounting const& report) -> void { budgetReport_ = report; }

    // Progress of the last generation attempt that was not committed, with the reason; nullopt if the last attempted
    // generation was committed. A generation is not committed if its offspring pool was not completely filled (time
    // limit or stop request), if it spent more than the budget has left, or if it made more attempts than the
    // generator allows (OffspringGeneratorBase::MaxAttempts()); in the last two cases the pool may be full
    // (Completed == PoolSize). Such a generation is neither reinserted, counted in Generation(), nor reported:
    // Parents() holds the last committed generation and the worker RNG states are those it had when the discarded
    // generation started. The counts and flags describe the work executed before the workers stopped and may depend
    // on the thread count (see OffspringProgress, DiscardReason). A run that a committed generation ended by crossing
    // the attempt cap leaves this empty: see BudgetReport().End.
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
