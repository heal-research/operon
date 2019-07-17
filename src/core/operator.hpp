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
        // all operators take a source of randomness as the first parameter
        virtual Ret operator()(RandomDevice& random, Args... args) const = 0;
    };

    // it's useful to have a data structure holding additional attributes for a solution candidate 
    template<size_t D = 1UL>
    struct Individual 
    {
        Tree   Genotype;
        double Fitness[D];
        size_t Dimension = D;
    };

    // crossover takes two parent trees and returns a child
    using CrossoverBase = OperatorBase<Tree, const Tree&, const Tree&>;
    // the mutator takes one parent tree and returns a child
    using MutatorBase   = OperatorBase<Tree, const Tree&>;
    // the selector a vector of individuals and returns the index of a selected individual per each call of operator() 
    // this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
    using SelectorBase  = OperatorBase<size_t>;
    // the creator builds a new tree using the existing grammar and allowed inputs
    using CreatorBase   = OperatorBase<Tree, const Grammar&, const std::vector<Variable>&>;
}

#endif

