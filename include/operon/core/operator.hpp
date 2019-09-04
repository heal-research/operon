#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <random>
#include "gsl/gsl"

#include "common.hpp"
#include "tree.hpp"
#include "grammar.hpp"
#include "dataset.hpp"

namespace Operon
{
    template<typename Ret, typename... Args>
    struct OperatorBase 
    {
        // all operators take a random device (source of randomness) as the first parameter
        virtual Ret operator()(operon::rand_t& random, Args... args) const = 0;
    };

    // it's useful to have a data structure holding additional attributes for a solution candidate 
    // maybe we should have an array of trees here? 
    template<size_t D = 1UL>
    struct Individual 
    {
        Tree                    Genotype;
        std::array<double, D>   Fitness;
        static constexpr size_t Dimension = D;

        inline double operator[](gsl::index i) const { return Fitness[i]; }
    };

    // the creator builds a new tree using the existing grammar and allowed inputs
    struct CreatorBase : public OperatorBase<Tree, const Grammar&, const gsl::span<const Variable>> { };

    // crossover takes two parent trees and returns a child
    struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> { };

    // the mutator can work in place or return a copy (child) 
    template<bool InPlace, typename RetType = std::conditional_t<InPlace, void, Tree>, typename ArgType = std::conditional_t<InPlace, Tree&, const Tree&>>
    struct MutatorBase  : public OperatorBase<RetType, ArgType> 
    {
        RetType operator()(operon::rand_t& random, ArgType tree) const override
        {
            if constexpr (InPlace)
            {
                Mutate(random, tree);
            }
            else
            {
                auto child = tree;
                Mutate(random, child);
                return child;
            }
        }

        virtual void Mutate(operon::rand_t& random, Tree& tree) const = 0;
    };

    // the selector a vector of individuals and returns the index of a selected individual per each call of operator() 
    // this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
    template<typename T, size_t Idx, bool Max>
    class SelectorBase : public OperatorBase<gsl::index> 
    {
        public:
            using  SelectableType                   = T;
            static constexpr size_t SelectableIndex = Idx;
            static constexpr bool   Maximization    = Max;

            virtual void Reset(const gsl::span<const T> pop) = 0;

        protected:
            gsl::span<const T> population;
    };
}
#endif

