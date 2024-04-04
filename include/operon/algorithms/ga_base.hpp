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

    GeneticAlgorithmBase(Problem const& problem, GeneticAlgorithmConfig const& config, TreeInitializerBase const& treeInit, CoefficientInitializerBase const& coeffInit, OffspringGeneratorBase const& generator, ReinserterBase const& reinserter)
        : problem_(problem)
        , config_(config)
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

    [[nodiscard]] auto GetProblem() const -> const Problem& { return problem_.get(); }
    [[nodiscard]] auto GetConfig() const -> const GeneticAlgorithmConfig& { return config_.get(); }

    [[nodiscard]] auto GetTreeInitializer() const -> TreeInitializerBase const& { return treeInit_.get(); }
    [[nodiscard]] auto GetCoefficientInitializer() const -> CoefficientInitializerBase const& { return coeffInit_.get(); }
    [[nodiscard]] auto GetGenerator() const -> const OffspringGeneratorBase& { return generator_.get(); }
    [[nodiscard]] auto GetReinserter() const -> const ReinserterBase& { return reinserter_.get(); }

    [[nodiscard]] auto Generation() const -> size_t { return generation_; }
    auto Generation() -> size_t& { return generation_; }

    auto Reset() -> void
    {
        generation_ = 0;
        GetGenerator().Evaluator().Reset();
    }

private:
    std::reference_wrapper<const Problem> problem_;
    std::reference_wrapper<const GeneticAlgorithmConfig> config_;
    std::reference_wrapper<const TreeInitializerBase> treeInit_;
    std::reference_wrapper<const CoefficientInitializerBase> coeffInit_;
    std::reference_wrapper<const OffspringGeneratorBase> generator_;
    std::reference_wrapper<const ReinserterBase> reinserter_;

    Operon::Vector<Individual> individuals_;
    Operon::Span<Individual> parents_;
    Operon::Span<Individual> offspring_;

    size_t generation_{0};
};

} // namespace Operon

#endif
