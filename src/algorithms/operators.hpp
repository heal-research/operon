#include <random>

#include "jsf.hpp"
#include "tree.hpp"

namespace Operon
{
    using Rand = Random::JsfRand<64>;

    // it is useful to have a data structure holding additional attributes 
    template<size_t D = 1UL>
    struct Individual 
    {
        Tree   Genotype;
        double Fitness[D];
    };

    Tree Cross(Rand& random, const Tree& lhs, const Tree& rhs, double internalProb, int maxLength, int maxDepth); 
    Tree MutateOnePoint(Rand& random, const Tree& tree);

    // T is an individual and Index is the index of the objective to consider (for multi-objective)
    template<typename T, size_t Index>
    std::vector<T> SelectTournament(Rand& random, const std::vector<T>& population, size_t tournamentSize, bool maximization = true);
}

