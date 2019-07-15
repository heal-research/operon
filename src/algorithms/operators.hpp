#include <random>

#include "jsf.hpp"
#include "tree.hpp"

namespace Operon
{
    using Rand = Random::JsfRand<64>;

    // declare some useful data structures
    template<typename Ret, typename... Args>
    struct OperatorBase 
    {
        virtual Ret operator()(Rand& random, Args... args) const = 0;
    };
    // it's useful to have a data structure holding additional attributes for a solution candidate 
    template<size_t D = 1UL>
    struct Individual 
    {
        Tree   Genotype;
        double Fitness[D];
    };

    using CrossoverBase = OperatorBase<Tree, const Tree&, const Tree&>;
    using MutatorBase   = OperatorBase<Tree, const Tree&>;
}

