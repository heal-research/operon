#ifndef CONFIG_HPP
#define CONFIG_HPP

#include <cstdint>

struct GeneticAlgorithmConfig
{
    size_t Generations;
    size_t Evaluations;
    size_t Iterations;
    size_t PopulationSize;
    double CrossoverProbability;
    double MutationProbability;
};

struct OffspringSelectionGeneticAlgorithmConfig : public GeneticAlgorithmConfig
{
    size_t MaxSelectionPressure;
};

#endif

