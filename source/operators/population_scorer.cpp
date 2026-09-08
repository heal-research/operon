// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include "operon/operators/population_scorer.hpp"

#include <stdexcept>

#include "operon/operators/evaluator.hpp"
#include "operon/operators/local_search.hpp"

namespace Operon {

void CpuPopulationOffspringScorer::Score(std::span<Individual> candidates,
                                         std::span<RandomGenerator> random,
                                         EvaluatorBase const& evaluator,
                                         CoefficientOptimizer const* optimizer,
                                         double localSearchProbability,
                                         double lamarckianProbability,
                                         std::span<Vector<Scalar>> scratch)
{
    if (candidates.size() != random.size() || candidates.size() != scratch.size()) {
        throw std::invalid_argument("population offspring scorer requires matching candidate, RNG, and scratch spans");
    }
    auto const rows = evaluator.GetProblem()->TrainingRange().Size();
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        scratch[index].resize(rows);
        ScoreIndividual(random[index], candidates[index], evaluator, optimizer, localSearchProbability,
                        lamarckianProbability, std::span<Scalar>{scratch[index]});
    }
}

} // namespace Operon
