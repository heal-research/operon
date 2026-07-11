// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef GA_BASE_HPP
#define GA_BASE_HPP

#include <string>
#include <operon/operon_export.hpp>
#include "operon/core/types.hpp"
#include "operon/operators/generator.hpp"
#include "config.hpp"
#include "stoppable.hpp"

namespace Operon {

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
    {
    }
    GeneticAlgorithmBase(GeneticAlgorithmBase&&) = delete;
    auto operator=(GeneticAlgorithmBase const& other) -> GeneticAlgorithmBase&
    {
        if (this == &other) { return *this; }
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
        return *this;
    }
    auto operator=(GeneticAlgorithmBase&&) -> GeneticAlgorithmBase& = delete;

    GeneticAlgorithmBase(GeneticAlgorithmConfig config, gsl::not_null<Problem const*> problem, gsl::not_null<TreeInitializerBase const*> treeInit, gsl::not_null<CoefficientInitializerBase const*> coeffInit, gsl::not_null<OffspringGeneratorBase const*> generator, gsl::not_null<ReinserterBase const*> reinserter)
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

    [[nodiscard]] auto Parents() const -> Operon::Span<Individual const> { return { parents_.data(), parents_.size() }; }
    auto Parents() -> Operon::Span<Individual> { return parents_; }

    [[nodiscard]] auto Offspring() const -> Operon::Span<Individual const> { return { offspring_.data(), offspring_.size() }; }
    auto Offspring() -> Operon::Span<Individual> { return offspring_; }

    [[nodiscard]] auto Individuals() -> Operon::Vector<Operon::Individual>& { return individuals_; }
    [[nodiscard]] auto Individuals() const -> Operon::Vector<Operon::Individual> const& { return individuals_; }

    [[nodiscard]] auto GetProblem() const -> const Problem* { return problem_.get(); }
    [[nodiscard]] auto GetConfig() const -> GeneticAlgorithmConfig { return config_; }

    [[nodiscard]] auto GetTreeInitializer() const -> TreeInitializerBase const* { return treeInit_.get(); }
    [[nodiscard]] auto GetCoefficientInitializer() const -> CoefficientInitializerBase const* { return coeffInit_.get(); }
    [[nodiscard]] auto GetGenerator() const -> OffspringGeneratorBase const* { return generator_.get(); }
    [[nodiscard]] auto GetReinserter() const -> ReinserterBase const* { return reinserter_.get(); }

    [[nodiscard]] auto WorkerRngs() const -> std::vector<Operon::RandomGenerator> const& { return workerRngs_; }
    auto WorkerRngs() -> std::vector<Operon::RandomGenerator>& { return workerRngs_; }

    [[nodiscard]] auto Generation() const -> size_t { return generation_; }
    auto Generation() -> size_t& { return generation_; }

    [[nodiscard]] auto Elapsed() const -> double { return elapsed_; }
    auto Elapsed() -> double& { return elapsed_; }

    [[nodiscard]] auto Timings() const -> Operon::Map<std::string, double> const& { return phaseTimes_; }
    auto Timings() -> Operon::Map<std::string, double>& { return phaseTimes_; }

    [[nodiscard]] auto IsFitted() const -> bool { return isFitted_; }
    auto IsFitted() -> bool& { return isFitted_; }

    // StopRequested()/RequestStop() are inherited from StoppableAlgorithm.

    // Valid to call between runs only. The PhaseTimer observer owns its own
    // totals and is recreated each Run(), so Reset() mid-run would cause the
    // next reportProgress sync to overwrite the cleared map with stale data.
    auto Reset() -> void
    {
        generation_ = 0;
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
    size_t generation_{0};
    double elapsed_{0};
    Operon::Map<std::string, double> phaseTimes_;
    bool isFitted_{false};
};

} // namespace Operon

#endif
