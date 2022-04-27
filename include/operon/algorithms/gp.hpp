// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef GP_HPP
#define GP_HPP

#include <cstddef>                         // for size_t
#include <functional>                      // for reference_wrapper, function
#include <nonstd/span.hpp>                 // for span<>::pointer
#include <operon/operon_export.hpp>        // for OPERON_EXPORT
#include <thread>                          // for thread
#include <utility>                         // for move
#include "operon/algorithms/config.hpp"    // for GeneticAlgorithmConfig
#include "operon/core/individual.hpp"      // for Individual
#include "operon/core/types.hpp"           // for Span, Vector, RandomGenerator
#include "operon/operators/evaluator.hpp"  // for EvaluatorBase
#include "operon/operators/generator.hpp"  // for OffspringGeneratorBase

// forward declaration
namespace tf { class Executor; }

namespace Operon {

class Problem;
class ReinserterBase;
struct CoefficientInitializerBase;
struct TreeInitializerBase;

class OPERON_EXPORT GeneticProgrammingAlgorithm {
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

public:
    explicit GeneticProgrammingAlgorithm(Problem const& problem, GeneticAlgorithmConfig const& config, TreeInitializerBase const& treeInit, CoefficientInitializerBase const& coeffInit, OffspringGeneratorBase const& generator, ReinserterBase const& reinserter)
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
    [[nodiscard]] auto Offspring() const -> Operon::Span<Individual const> { return { offspring_.data(), offspring_.size() }; }

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
        generator_.get().Evaluator().Reset();
    }

    auto Run(tf::Executor& /*executor*/, Operon::RandomGenerator&/*rng*/, std::function<void()> /*report*/ = nullptr) -> void;
    auto Run(Operon::RandomGenerator& /*rng*/, std::function<void()> /*report*/ = nullptr, size_t /*threads*/= 0) -> void;
};
} // namespace Operon

#endif

