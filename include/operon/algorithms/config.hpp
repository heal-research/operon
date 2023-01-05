// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_ALGORITHM_CONFIG_HPP
#define OPERON_ALGORITHM_CONFIG_HPP

#include <cstddef>

namespace Operon {
struct GeneticAlgorithmConfig {
    size_t Generations; // generation limit
    size_t Evaluations; // evaluation budget
    size_t Iterations;  // local search iterations
    size_t PopulationSize;
    size_t PoolSize;
    size_t Seed;        // random seed
    size_t TimeLimit;   // time limit
    double CrossoverProbability;
    double MutationProbability;
    double Epsilon;     // used when comparing fitness values
};
} // namespace Operon

#endif
