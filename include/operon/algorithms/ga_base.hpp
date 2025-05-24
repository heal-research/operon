// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef GA_BASE_HPP
#define GA_BASE_HPP

#include <functional>
#include <operon/operon_export.hpp>
#include "operon/operators/generator.hpp"
#include "config.hpp"

namespace Operon {

class Problem;
class ReinserterBase;
struct CoefficientInitializerBase;
struct TreeInitializerBase;

class GeneticAlgorithmBase {
public:
    virtual ~GeneticAlgorithmBase() = default;
    GeneticAlgorithmBase(const GeneticAlgorithmBase&) = default;
    GeneticAlgorithmBase(GeneticAlgorithmBase&&) = delete;
    auto operator=(const GeneticAlgorithmBase&) -> GeneticAlgorithmBase& = default;
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

    [[nodiscard]] auto Generation() const -> size_t { return generation_; }
    auto Generation() -> size_t& { return generation_; }

    [[nodiscard]] auto Elapsed() const -> double { return elapsed_; }
    auto Elapsed() -> double& { return elapsed_; }

    [[nodiscard]] auto IsFitted() const -> bool { return isFitted_; }
    auto IsFitted() -> bool& { return isFitted_; }

    auto Reset() -> void
    {
        generation_ = 0;
        elapsed_ = 0;
        GetGenerator()->Evaluator()->Reset();
    }

    auto RestoreIndividuals(std::vector<Individual> inds) -> void
    {
        EXPECT(inds.size() == config_.PoolSize + config_.PopulationSize,
                "Mismatched number of individuals (must match pool/population sizes)");
        individuals_ = std::move(inds);
        parents_ = Operon::Span<Individual>(individuals_.data(), config_.PoolSize);
        offspring_ = Operon::Span<Individual>(individuals_.data() + config_.PoolSize, config_.PopulationSize);
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

    size_t generation_{0};
    double elapsed_{0}; // elapsed time in microseconds
    bool isFitted_{false};
};

} // namespace Operon

#endif
