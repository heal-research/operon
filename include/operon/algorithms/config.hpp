// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstddef>

namespace Operon {
struct GeneticAlgorithmConfig {
    size_t Generations; // generation limit
    size_t Evaluations; // evaluation budget
    size_t Iterations;  // local search iterations
    size_t PopulationSize;
    size_t PoolSize;
    double CrossoverProbability;
    double MutationProbability;
    size_t Seed;        // random seed
    size_t TimeLimit;   // time limit
};
} // namespace Operon

#endif
