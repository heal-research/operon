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

// Means over a population, accumulated serially in double with a compensated (TwoSum) fitness sum: a float
// reduction overflows on the ErrMax sentinel and depends on the reduction order, and a plain double sum loses small
// terms next to large ones. Individuals that failed evaluation (non-finite objective or the ErrMax sentinel)
// are excluded from the fitness mean; if none is valid the mean is ErrMax, so a fully failed population stays visible
// as a finite value. Length is averaged over every individual.
inline auto ComputePopulationMeans(Operon::Span<Operon::Individual const> pop, std::size_t objective) -> PopulationMeans
{
    PopulationMeans means;
    if (pop.empty()) {
        return means;
    }
    double lengthSum = 0;
    vstat::compensated_sum<double> fitnessSum;
    for (auto const& ind : pop) {
        lengthSum += static_cast<double>(ind.Genotype.Length());
        auto const fitness = ind[objective];
        if (std::isfinite(fitness) && fitness < Operon::EvaluatorBase::ErrMax) {
            fitnessSum(static_cast<double>(fitness));
            ++means.Valid;
        }
    }
    means.Length = lengthSum / static_cast<double>(pop.size());
    means.Fitness = means.Valid > 0 ? fitnessSum.value() / static_cast<double>(means.Valid)
                                    : static_cast<double>(Operon::EvaluatorBase::ErrMax);
    return means;
}

} // namespace Operon

#endif
