// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_POPULATION_SCORER_HPP
#define OPERON_POPULATION_SCORER_HPP

#include <cstddef>
#include <span>

#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

class CoefficientOptimizer;
struct EvaluatorBase;
struct Individual;

// Optional generation-level scoring seam. Algorithms invoke this only when
// GeneticAlgorithmConfig::PopulationScorer is non-null; scalar generator and
// evaluator APIs remain the default path. Candidates are supplied in their
// original offspring order and must leave with fitness populated in that same
// order.
class OPERON_EXPORT PopulationOffspringScorer {
public:
    virtual ~PopulationOffspringScorer() = default;

    virtual void Score(std::span<Individual> candidates,
                       std::span<RandomGenerator> random,
                       EvaluatorBase const& evaluator,
                       CoefficientOptimizer const* optimizer,
                       double localSearchProbability,
                       double lamarckianProbability,
                       std::span<Vector<Scalar>> scratch) = 0;
};

// Reference implementation for validating the opt-in generation path. It is
// intentionally equivalent to the legacy scalar scoring path, not a parallel
// backend. Accelerator implementations can replace it without changing the
// generator or evaluator contracts.
class OPERON_EXPORT CpuPopulationOffspringScorer final : public PopulationOffspringScorer {
public:
    void Score(std::span<Individual> candidates,
               std::span<RandomGenerator> random,
               EvaluatorBase const& evaluator,
               CoefficientOptimizer const* optimizer,
               double localSearchProbability,
               double lamarckianProbability,
               std::span<Vector<Scalar>> scratch) final;
};

} // namespace Operon

#endif // OPERON_POPULATION_SCORER_HPP
