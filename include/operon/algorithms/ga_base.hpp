// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef GA_BASE_HPP
#define GA_BASE_HPP

#include "config.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/generator.hpp"
#include "stoppable.hpp"
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
    BudgetSpent, // committed work reached the budget (initialization alone may do it), no generation was started
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
    // Not `= default`: parents_/offspring_ are non-owning spans into
    // individuals_'s storage, so a plain memberwise copy would leave the
    // copy's spans pointing at the *source* object's individuals_ buffer
    // (dangling once the source is destroyed, and aliasing it while both
    // are alive). They're rebound here to the copy's own individuals_,
    // exactly like the primary constructor and RestoreIndividuals() do -
    // both derive them from config_.PopulationSize/PoolSize rather than
    // copying the span objects themselves. The atomic stopRequested_ still
    // gets its own handling via StoppableAlgorithm's copy ctor/assign
    // (invoked explicitly here since every other member is memberwise-safe
    // to copy or, like the spans, needs deriving instead).
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

    // Parents()/Offspring() genuinely change *type* between const and
    // non-const (Span<Individual const> vs Span<Individual>), not just
    // reference-qualification - a plain forwarding-reference return would
    // return `Operon::Span<Individual> const&` for a const Self, which
    // still exposes mutable *elements* through that span (Span's
    // operator[] isn't disabled by the span object's own constness), i.e.
    // it wouldn't actually be const-correct. The conditional_t below keeps
    // the const-correctness the original two overloads had.
    //
    // Self& (not Self&&): the original non-ref-qualified overloads were
    // technically callable on an rvalue *this, immediately returning a Span
    // into that rvalue's about-to-be-destroyed individuals_ - a dangling
    // Span by construction. Self& refuses to compile for a *mutable* rvalue
    // caller instead, a deliberate, source-incompatible tightening (no
    // in-repo or downstream caller does this). A const rvalue caller still
    // compiles here (Self deduces to a const type, so Self& becomes
    // Self const&, which binds a const rvalue same as it always could) -
    // this only closes off the mutable-rvalue case, not rvalues generally.
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

    // Individuals()/WorkerRngs()/Timings() only ever changed reference
    // qualification (T& vs T const&) between overloads (never by-value, so
    // these don't need the conditional_t treatment Generation()/Elapsed()/
    // IsFitted() below get), and ordinary forwarding-reference deduction via
    // `auto&&` reproduces both lvalue cases exactly. It's not a byte-for-
    // byte match for rvalue callers specifically - the old, non-ref-
    // qualified overloads always returned a plain T&/T const& regardless of
    // the object's value category, while `auto&&` forwarding gives T&&/
    // T const&& for a (mutable or const) rvalue caller instead. Not a new
    // hazard (both are still references, and neither version's reference
    // outlives the rvalue any better than the other - no in-repo or
    // downstream caller does this), just a different reference category
    // than before. decltype(auto) would have been the wrong choice here for
    // the lvalue/const-lvalue cases specifically: on an unparenthesized
    // member-access return expression it yields the *declared* member type
    // by value (a silent copy), losing reference identity entirely - see
    // Nodes() in tree.hpp for the full explanation.
    template <typename Self> [[nodiscard]] auto&& Individuals(this Self&& self)
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

    // Generation()/Elapsed()/IsFitted() are trivially-copyable scalars whose
    // non-const overload existed only to allow direct assignment (e.g.
    // `algo.IsFitted() = true;`); neither original overload was ref-
    // qualified, so the const one returned by value regardless of whether
    // the object was an lvalue or rvalue. A plain `auto&&` here would get
    // that wrong for a const *rvalue* specifically: Self deduces to a
    // non-reference type, and forwarding through it yields a `T const&&`
    // binding - a reference into a temporary - where the old code always
    // handed back a safe, independent copy. (Reading through a const
    // *lvalue* isn't affected: only the reference category changes there,
    // same as Individuals()/WorkerRngs()/Timings() above.)
    //
    // No caller does this today (confirmed: nothing in this repo or its
    // downstream consumers calls these getters on a const rvalue), but it's
    // one line of conditional_t to close off entirely rather than leave as
    // a latent trap. Note this deliberately does *not* forward self: naming
    // `self` inside the function body is always an lvalue expression
    // regardless of which reference type Self deduced to, so branching the
    // return *type* on constness alone (ignoring value category) is enough
    // to reproduce the original by-value-for-const/reference-for-mutable
    // split exactly, for all four Self×value-category combinations.
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

    template <typename Self> [[nodiscard]] auto&& Timings(this Self&& self)
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

    // Valid to call between runs only. The PhaseTimer observer owns its own
    // totals and is recreated each Run(), so Reset() mid-run would cause the
    // next reportProgress sync to overwrite the cleared map with stale data.
    auto Reset() -> void
    {
        generation_ = 0;
        incompleteGeneration_.reset();
        budgetReport_ = {};
        elapsed_ = 0;
        ClearStopRequested();
        phaseTimes_.clear();
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
