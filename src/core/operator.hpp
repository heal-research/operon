#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <random>
#include "gsl/span"

#include "jsf.hpp"
#include "tree.hpp"
#include "grammar.hpp"
#include "dataset.hpp"

namespace Operon
{
    using RandomDevice = Random::JsfRand<64>;

    template<typename Ret, typename... Args>
    struct OperatorBase 
    {
        // all operators take a random device (source of randomness) as the first parameter
        virtual Ret operator()(RandomDevice& random, Args... args) const = 0;
    };

    // it's useful to have a data structure holding additional attributes for a solution candidate 
    // maybe we should have an array of trees here? 
    template<size_t D = 1UL>
    struct Individual 
    {
        Tree                  Genotype;
        std::array<double, D> Fitness;
    };

    // the creator builds a new tree using the existing grammar and allowed inputs
    struct CreatorBase : public OperatorBase<Tree, const Grammar&, const std::vector<Variable>&> { };

    // crossover takes two parent trees and returns a child
    struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> { };

    // the mutator takes one parent tree and returns a child
    struct MutatorBase :   public OperatorBase<Tree, const Tree&> { };

    // the selector a vector of individuals and returns the index of a selected individual per each call of operator() 
    // this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
    template<typename T, size_t Idx, bool Max>
    class SelectorBase : public OperatorBase<size_t> 
    {
        public:
            using Ind                          = T;
            static constexpr size_t Index      = Idx;
            static constexpr bool Maximization = Max;

            virtual void Reset(const std::vector<T>& pop) = 0;

        protected:
            inline bool Compare(size_t lhs, size_t rhs) const noexcept 
            {
                if constexpr (Max) return population[lhs].Fitness[Idx] < population[rhs].Fitness[Idx];
                else return population[lhs].Fitness[Idx] > population[rhs].Fitness[Idx];
            }
            gsl::span<const T> population;
    };
}
#endif

