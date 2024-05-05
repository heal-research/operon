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
    size_t TimeLimit{~std::size_t{0}}; // time limit
    double CrossoverProbability{1.0};
    double MutationProbability{0.25};
    double LocalSearchProbability{1.0};
    double LamarckianProbability{1.0};
    double Epsilon{0};     // used when comparing fitness values
};
} // namespace Operon

#endif
