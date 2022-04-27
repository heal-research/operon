// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_NSGA2_HPP
#define OPERON_NSGA2_HPP

#include <functional>                      // for reference_wrapper, function
#include <operon/operon_export.hpp>        // for OPERON_EXPORT
#include <thread>                          // for thread
#include <utility>                         // for move
#include <vector>                          // for vector
#include "operon/algorithms/config.hpp"    // for GeneticAlgorithmConfig
#include "operon/core/individual.hpp"      // for Individual
#include "operon/core/types.hpp"           // for Span, Vector, RandomGenerator
#include "operon/operators/evaluator.hpp"  // for EvaluatorBase
#include "operon/operators/generator.hpp"  // for OffspringGeneratorBase

// forward declaration
namespace tf { class Executor; }

namespace Operon {

class NondominatedSorterBase; 
class Problem; 
class ReinserterBase; 
struct CoefficientInitializerBase; 
struct TreeInitializerBase; 

class OPERON_EXPORT NSGA2 {
    std::reference_wrapper<const Problem> problem_;
    std::reference_wrapper<const GeneticAlgorithmConfig> config_;

    std::reference_wrapper<const TreeInitializerBase> treeInit_;
    std::reference_wrapper<const CoefficientInitializerBase> coeffInit_;
    std::reference_wrapper<const OffspringGeneratorBase> generator_;
    std::reference_wrapper<const ReinserterBase> reinserter_;
    std::reference_wrapper<const NondominatedSorterBase> sorter_;

    Operon::Vector<Individual> individuals_;
    Operon::Span<Individual> parents_;
    Operon::Span<Individual> offspring_;

    size_t generation_{0};
    std::vector<std::vector<size_t>> fronts_;

    // best pareto front
    Operon::Vector<Individual> best_;

    auto UpdateDistance(Operon::Span<Individual> pop) -> void;
    auto Sort(Operon::Span<Individual> pop) -> void;

public:
    explicit NSGA2(Problem const& problem, GeneticAlgorithmConfig const& config, TreeInitializerBase const& treeInit, CoefficientInitializerBase const& coeffInit, OffspringGeneratorBase const& generator, ReinserterBase const& reinserter, NondominatedSorterBase const& sorter)
        : problem_(problem)
        , config_(config)
        , treeInit_(treeInit)
        , coeffInit_(coeffInit)
        , generator_(generator)
        , reinserter_(reinserter)
        , sorter_(sorter)
        , individuals_(config.PopulationSize + config.PoolSize)
        , parents_(individuals_.data(), config.PopulationSize)
        , offspring_(individuals_.data() + config.PopulationSize, config.PoolSize)
    {
    }

    [[nodiscard]] auto Parents() const -> Operon::Span<Individual const> { return { parents_.data(), parents_.size() }; }
    [[nodiscard]] auto Offspring() const -> Operon::Span<Individual const> { return { offspring_.data(), offspring_.size() }; }
    [[nodiscard]] auto Best() const -> Operon::Span<Individual const> { return { best_.data(), best_.size() }; }

    [[nodiscard]] auto GetProblem() const -> const Problem& { return problem_.get(); }
    [[nodiscard]] auto GetConfig() const -> const GeneticAlgorithmConfig& { return config_.get(); }

    [[nodiscard]] auto GetTreeInitializer() const -> TreeInitializerBase const& { return treeInit_.get(); }
    [[nodiscard]] auto GetCoefficientInitializer() const -> CoefficientInitializerBase const& { return coeffInit_.get(); }
    [[nodiscard]] auto GetGenerator() const -> const OffspringGeneratorBase& { return generator_.get(); }
    [[nodiscard]] auto GetReinserter() const -> const ReinserterBase& { return reinserter_.get(); }

    [[nodiscard]] auto Generation() const -> size_t { return generation_; }

    auto Reset() -> void
    {
        generation_ = 0;
        GetGenerator().Evaluator().Reset();
    }

    auto Run(tf::Executor& /*executor*/, Operon::RandomGenerator&/*rng*/, std::function<void()> /*report*/ = nullptr) -> void;
    auto Run(Operon::RandomGenerator& /*rng*/, std::function<void()> /*report*/ = nullptr, size_t /*threads*/= 0) -> void;
};
} // namespace Operon

#endif
