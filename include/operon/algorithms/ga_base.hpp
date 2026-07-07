// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef GA_BASE_HPP
#define GA_BASE_HPP

#include <atomic>
#include <functional>
#include <string>
#include <operon/operon_export.hpp>
#include "operon/core/types.hpp"
#include "operon/operators/generator.hpp"
#include "config.hpp"

namespace Operon {

class Problem;
class ReinserterBase;
struct CoefficientInitializerBase;
struct TreeInitializerBase;

// Invoked once per generation by every algorithm's Run() to report progress.
// Returning true requests early termination; each algorithm's own stop
// condition ORs StopRequested() in alongside its evaluator-budget,
// generation-count, and time-limit checks.
using ReportCallback = std::function<bool()>;

class GeneticAlgorithmBase {
public:
    virtual ~GeneticAlgorithmBase() = default;
    // std::atomic<bool> isn't copyable, so stopRequested_ needs manual
    // handling; every other member is copied exactly as the defaulted
    // versions would have done.
    GeneticAlgorithmBase(GeneticAlgorithmBase const& other)
        : config_(other.config_)
        , problem_(other.problem_)
        , treeInit_(other.treeInit_)
        , coeffInit_(other.coeffInit_)
        , generator_(other.generator_)
        , reinserter_(other.reinserter_)
        , individuals_(other.individuals_)
        , parents_(other.parents_)
        , offspring_(other.offspring_)
        , workerRngs_(other.workerRngs_)
        , generation_(other.generation_)
        , elapsed_(other.elapsed_)
        , phaseTimes_(other.phaseTimes_)
        , isFitted_(other.isFitted_)
        , stopRequested_(other.stopRequested_.load())
    {
    }
    GeneticAlgorithmBase(GeneticAlgorithmBase&&) = delete;
    auto operator=(GeneticAlgorithmBase const& other) -> GeneticAlgorithmBase&
    {
        if (this == &other) { return *this; }
        config_ = other.config_;
        problem_ = other.problem_;
        treeInit_ = other.treeInit_;
        coeffInit_ = other.coeffInit_;
        generator_ = other.generator_;
        reinserter_ = other.reinserter_;
        individuals_ = other.individuals_;
        parents_ = other.parents_;
        offspring_ = other.offspring_;
        workerRngs_ = other.workerRngs_;
        generation_ = other.generation_;
        elapsed_ = other.elapsed_;
        phaseTimes_ = other.phaseTimes_;
        isFitted_ = other.isFitted_;
        stopRequested_.store(other.stopRequested_.load());
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

    // Set by an algorithm's Run() when its ReportCallback returns true; each
    // algorithm's own stop condition ORs this in. Atomic so it's also safe to
    // call RequestStop() from outside the callback (e.g. another thread, a
    // signal handler) while Run() is in progress.
    [[nodiscard]] auto StopRequested() const -> bool { return stopRequested_.load(std::memory_order_acquire); }
    auto RequestStop() -> void { stopRequested_.store(true, std::memory_order_release); }

    // Valid to call between runs only. The PhaseTimer observer owns its own
    // totals and is recreated each Run(), so Reset() mid-run would cause the
    // next reportProgress sync to overwrite the cleared map with stale data.
    auto Reset() -> void
    {
        generation_ = 0;
        elapsed_ = 0;
        stopRequested_.store(false, std::memory_order_release);
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
    std::atomic<bool> stopRequested_{false};
};

} // namespace Operon

#endif
