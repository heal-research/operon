// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef GP_HPP
#define GP_HPP

#include <cstddef>                         // for size_t
#include <functional>                      // for reference_wrapper, function
#include <operon/operon_export.hpp>        // for OPERON_EXPORT
#include <thread>                          // for thread
#include <utility>                         // for move

#include "operon/algorithms/config.hpp"    // for GeneticAlgorithmConfig
#include "operon/algorithms/ga_base.hpp"
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

class OPERON_EXPORT GeneticProgrammingAlgorithm : public GeneticAlgorithmBase {
public:
    GeneticProgrammingAlgorithm(GeneticAlgorithmConfig config, gsl::not_null<Problem const*> problem, gsl::not_null<TreeInitializerBase const*> treeInit, gsl::not_null<CoefficientInitializerBase const*> coeffInit, gsl::not_null<OffspringGeneratorBase const*> generator, gsl::not_null<ReinserterBase const*> reinserter)
        : GeneticAlgorithmBase(config, problem, treeInit, coeffInit, generator, reinserter)
    {
    }

    auto Run(tf::Executor& /*executor*/, Operon::RandomGenerator&/*rng*/, std::function<void()> /*report*/ = nullptr, /*warmStart*/ bool = false) -> void;
    auto Run(Operon::RandomGenerator& /*rng*/, std::function<void()> /*report*/ = nullptr, size_t /*threads*/= 0, /*warmStart*/ bool = false) -> void;
};
} // namespace Operon

#endif
