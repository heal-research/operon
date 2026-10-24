// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_POPULATION_STATS_HPP
#define OPERON_CLI_POPULATION_STATS_HPP

#include <cmath>
#include <cstddef>

#include <vstat/vstat.hpp>

#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/evaluator.hpp"

namespace Operon {

struct PopulationMeans {
    double Length { 0 }; // mean genotype length over the whole population
    double Fitness { 0 }; // mean of the chosen objective over the valid individuals
    std::size_t Valid { 0 }; // individuals with a finite objective below EvaluatorBase::ErrMax
};

// Means over a population, accumulated serially in double with a compensated (TwoSum) sum of per-term quotients:
// a float reduction overflows on the ErrMax sentinel and depends on the reduction order, and a plain sum overflows
// in double builds or loses small terms next to large ones. Individuals that failed evaluation (non-finite objective or
// the ErrMax sentinel) are excluded from the fitness mean; if none is valid the mean is ErrMax, so a fully failed
// population stays visible as a finite value. Length is averaged over every individual.
inline auto ComputePopulationMeans(Operon::Span<Operon::Individual const> pop, std::size_t objective) -> PopulationMeans
{
    PopulationMeans means;
    if (pop.empty()) {
        return means;
    }
    auto const valid
        = [&](Operon::Scalar fitness) { return std::isfinite(fitness) && fitness < Operon::EvaluatorBase::ErrMax; };
    double lengthSum = 0;
    for (auto const& ind : pop) {
        lengthSum += static_cast<double>(ind.Genotype.Length());
        means.Valid += static_cast<std::size_t>(valid(ind[objective]));
    }
    means.Length = lengthSum / static_cast<double>(pop.size());
    if (means.Valid == 0) {
        means.Fitness = static_cast<double>(Operon::EvaluatorBase::ErrMax);
        return means;
    }
    // Each term is divided by the count before it is added, so the running sum never exceeds the largest valid value
    // (a plain sum of n values near the sentinel overflows even in double); the sum is compensated.
    auto const n = static_cast<double>(means.Valid);
    vstat::compensated_sum<double> fitnessSum;
    for (auto const& ind : pop) {
        if (valid(ind[objective])) {
            fitnessSum(static_cast<double>(ind[objective]) / n);
        }
    }
    means.Fitness = fitnessSum.value();
    return means;
}

} // namespace Operon

#endif
