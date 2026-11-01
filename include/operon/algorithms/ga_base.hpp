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
};

} // namespace Operon

#endif
