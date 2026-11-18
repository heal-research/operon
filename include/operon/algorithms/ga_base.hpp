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

// Offspring-generation progress of one generation attempt.
struct OffspringProgress {
    size_t Attempted { 0 }; // offspring-generator invocations (accepted and rejected)
    size_t Completed { 0 }; // pool slots filled
    size_t PoolSize { 0 }; // pool slots to fill
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
        , workerRngs_(other.workerRngs_)
        , generation_(other.generation_)
        , elapsed_(other.elapsed_)
        , phaseTimes_(other.phaseTimes_)
        , isFitted_(other.isFitted_)
        , incompleteGeneration_(other.incompleteGeneration_)
        , optimizerAccounting_(other.optimizerAccounting_)
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

    // Discarded generation progress, or nullopt if the last attempt completed.
    // Parents() retains the last completed generation.
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
};

} // namespace Operon

#endif
