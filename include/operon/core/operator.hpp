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
        Tree                  Genotype;
        std::array<double, D> Fitness;

        static constexpr size_t Dimension = D;
    };

    // the creator builds a new tree using the existing grammar and allowed inputs
    struct CreatorBase : public OperatorBase<Tree, const Grammar&, const gsl::span<const Variable>> { };

    // crossover takes two parent trees and returns a child
    struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> { };

    // the mutator takes one parent tree and returns a child
    struct MutatorBase :   public OperatorBase<void, Tree&> { };

    // the selector a vector of individuals and returns the index of a selected individual per each call of operator() 
    // this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
    template<typename T, size_t Idx, bool Max>
    class SelectorBase : public OperatorBase<gsl::index> 
    {
        public:
            using Ind                          = T;
            static constexpr size_t Index      = Idx;
            static constexpr bool Maximization = Max;

            virtual void Reset(const gsl::span<const T> pop) = 0;
            const T& Get(gsl::index i) const { return population[i]; }

        protected:
            inline bool Compare(gsl::index lhs, gsl::index rhs) const noexcept 
            {
                auto fit = [&](gsl::index i) { return population[i].Fitness[Idx]; };
                if constexpr (Max) return fit(lhs) < fit(rhs);
                else return fit(lhs) > fit(rhs);
            }
            gsl::span<const T> population;
    };
}
#endif

