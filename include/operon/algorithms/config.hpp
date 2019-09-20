#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstddef>

struct GeneticAlgorithmConfig {
    size_t Generations;
    size_t Evaluations;
    size_t Iterations;
    size_t PopulationSize;
    double CrossoverProbability;
    double MutationProbability;
    // offspring selection recombinator
    size_t MaxSelectionPressure;
    // brood recombinator
    size_t BroodSize;
    size_t BroodTournamentSize;
};

#endif
