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

// Evaluation counters split between scoring and coefficient optimization. Consumed() is budget usage; calls, optimizer
// time, and cache counters are informational. Cache counters are nullopt when no cache is configured.
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

// Why an offspring generation was not committed. ResourceLimit covers the evaluation budget and attempt cap;
// asynchronous limits are reported only when no resource limit was observed.
enum class DiscardReason : std::uint8_t {
    None,
    ResourceLimit, // the generation spent more than the budget has left and/or made more attempts than the cap
    TimeLimit, // the time limit expired before the offspring pool was filled
    StopRequested, // RequestStop() (or the report callback) ended the generation before the pool was filled
};

// Informational work performed by an uncommitted generation. Counts and flags may depend on the thread count.
struct OffspringProgress {
    size_t Attempted { 0 }; // offspring-generator invocations (accepted and rejected)
    size_t Completed { 0 }; // pool slots filled
    size_t PoolSize { 0 }; // pool slots to fill
    DiscardReason Reason { DiscardReason::None };
    bool BudgetExceeded { false }; // the executed work spent more than the budget had left
    bool AttemptCapExceeded { false }; // the executed work made more attempts than the generator's cap
};

// Why the last Run() ended. ResourceLimit and AttemptCap take precedence over loop-condition checks.
enum class RunEnd : std::uint8_t {
    None, // no Run() yet, or it is still running
    ResourceLimit, // a generation was discarded for budget and/or attempt cap (IncompleteGeneration() is set)
    AttemptCap, // a committed generation made more attempts than the cap and ended the run (nothing discarded)
    GenerationLimit, // Generation() reached GeneticAlgorithmConfig::Generations
    BudgetSpent, // committed work reached a positive budget (initialization alone may also do so)
    StopRequested, // RequestStop(), or the report callback returned true
    TimeLimit, // GeneticAlgorithmConfig::TimeLimit expired
};

// Evaluation-budget accounting for the last Run(). The budget limits committed work; discarded and other executed
// work is reported separately. Values saturate instead of wrapping and are refreshed during and after Run().
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
    // Custom copy operations rebind the non-owning spans to the copied individuals_ storage.
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

    // Preserve const-correct element types and reject mutable rvalue access to non-owning spans.
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

    // Forward accessors for members whose reference qualification may vary with the object.
    template <typename Self> [[nodiscard]] auto&& Individuals(this Self&& self)
    {
        return std::forward<Self>(self).individuals_;
    }

    [[nodiscard]] auto GetProblem() const -> const Problem* { return problem_.get(); }
    [[nodiscard]] auto GetConfig() const -> GeneticAlgorithmConfig { return config_; }

    // Run-owned coefficient-optimizer accounting. Custom drivers must pass it to scoring and generation calls.
    template <typename Self> [[nodiscard]] auto&& Accounting(this Self&& self)
    {
        return std::forward<Self>(self).optimizerAccounting_;
    }

    // Combined scoring, optimizer, and active-cache counters. A completed run yields an exact snapshot.
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

    // Preserve value returns for const scalar accessors and references for mutable accessors.
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

    // Budget accounting of the last Run(); distinct from Accounting(), which covers optimizer diagnostics.
    [[nodiscard]] auto BudgetReport() const -> BudgetAccounting const& { return budgetReport_; }
    auto SetBudgetReport(BudgetAccounting const& report) -> void { budgetReport_ = report; }

    // Progress of the last uncommitted generation; nullopt if the last generation was committed. Counts and flags
    // describe executed work and may depend on the thread count.
    [[nodiscard]] auto IncompleteGeneration() const -> std::optional<OffspringProgress> const&
    {
        return incompleteGeneration_;
    }
    auto SetIncompleteGeneration(std::optional<OffspringProgress> progress) -> void
    {
        incompleteGeneration_ = progress;
    }

    // StopRequested()/RequestStop() are inherited from StoppableAlgorithm.

    // Valid between runs only; PhaseTimer owns per-run totals.
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
