// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_NSGA2_HPP
#define OPERON_NSGA2_HPP

#include <functional>                      // for reference_wrapper, function
#include <operon/operon_export.hpp>        // for OPERON_EXPORT
#include <thread>                          // for thread
#include <utility>                         // for move
#include <vector>                          // for vector

#include "operon/algorithms/config.hpp"    // for GeneticAlgorithmConfig
#include "operon/algorithms/ga_base.hpp"
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

class OPERON_EXPORT NSGA2 : public GeneticAlgorithmBase {
    std::reference_wrapper<const NondominatedSorterBase> sorter_;
    std::vector<std::vector<size_t>> fronts_;
    Operon::Vector<Individual> best_; // best Pareto front

    auto UpdateDistance(Operon::Span<Individual> pop) -> void;
    auto Sort(Operon::Span<Individual> pop) -> void;

public:
    NSGA2(Problem const& problem, GeneticAlgorithmConfig const& config, TreeInitializerBase const& treeInit, CoefficientInitializerBase const& coeffInit, OffspringGeneratorBase const& generator, ReinserterBase const& reinserter, NondominatedSorterBase const& sorter)
        : GeneticAlgorithmBase(problem, config, treeInit, coeffInit, generator, reinserter), sorter_(sorter)
    {
        auto const nobj { GetGenerator().Evaluator().ObjectiveCount() };
        for (auto& ind : Individuals()) {
            ind.Fitness.resize(nobj, EvaluatorBase::ErrMax);
        }
    }

    [[nodiscard]] auto Best() const -> Operon::Span<Individual const> { return { best_.data(), best_.size() }; }

    auto Run(tf::Executor& /*executor*/, Operon::RandomGenerator&/*rng*/, std::function<void()> /*report*/ = nullptr) -> void;
    auto Run(Operon::RandomGenerator& /*rng*/, std::function<void()> /*report*/ = nullptr, size_t /*threads*/= 0) -> void;
};
} // namespace Operon

#endif
