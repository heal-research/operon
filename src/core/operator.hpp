#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <random>

#include "jsf.hpp"
#include "tree.hpp"
#include "grammar.hpp"
#include "dataset.hpp"

namespace Operon
{
    using RandomDevice = Random::JsfRand<64>;

    // declare some useful data structures
    template<typename Ret, typename... Args>
    struct OperatorBase 
    {
        // all operators take a random device (source of randomness) as the first parameter
        virtual Ret operator()(RandomDevice& random, Args... args) const = 0;
    };

    // it's useful to have a data structure holding additional attributes for a solution candidate 
    template<size_t D = 1UL>
    struct Individual 
    {
        Tree   Genotype;
        double Fitness[D];
        static constexpr size_t Dimension = D; 
    };

    // crossover takes two parent trees and returns a child
    struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> { };

    // the selector a vector of individuals and returns the index of a selected individual per each call of operator() 
    // this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
    template<typename T, size_t I>
    struct SelectorBase : public OperatorBase<size_t> 
    { 
        using TSelectable = T;
        static constexpr size_t Index = I;
        virtual void Reset(const std::vector<TSelectable>& population) = 0;
    };

    // the mutator takes one parent tree and returns a child
    struct MutatorBase :   public OperatorBase<Tree, const Tree&> { };

    // the creator builds a new tree using the existing grammar and allowed inputs
    struct CreatorBase :   public OperatorBase<Tree, const Grammar&, const std::vector<Variable>&> { };
}
#endif

