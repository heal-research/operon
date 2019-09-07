#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <random>
#include <atomic>
#include "gsl/gsl"

#include "common.hpp"
#include "tree.hpp"
#include "grammar.hpp"
#include "dataset.hpp"
#include "problem.hpp"

namespace Operon
{
    // operator base classes for two types of operators: stateless and stateful
    template<typename Ret, typename... Args>
    struct StatelessOperator 
    {
        using return_type = Ret;
        using argument_type = std::tuple<Args...>;
        // all operators take a random device (source of randomness) as the first parameter
        virtual Ret operator()(operon::rand_t& random, Args... args) const = 0;
    };

    template<typename Ret, typename... Args>
    struct StatefulOperator
    {
        using return_type = Ret;
        using argument_type = std::tuple<Args...>;
        virtual Ret operator()(operon::rand_t& random, Args... args) = 0;
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
    struct CreatorBase : public StatelessOperator<Tree, const Grammar&, const gsl::span<const Variable>> { };

    // crossover takes two parent trees and returns a child
    struct CrossoverBase : public StatelessOperator<Tree, const Tree&, const Tree&> { };

    // the mutator can work in place or return a copy (child) 
    template<bool InPlace, typename RetType = std::conditional_t<InPlace, void, Tree>, typename ArgType = std::conditional_t<InPlace, Tree&, const Tree&>>
    struct MutatorBase  : public StatelessOperator<RetType, ArgType> 
    {
        static constexpr bool inPlace = InPlace;

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
    template<typename T, gsl::index Idx, bool Max>
    class SelectorBase : public StatelessOperator<gsl::index> 
    {
        public:
            using  SelectableType                       = T;
            static constexpr gsl::index SelectableIndex = Idx;
            static constexpr bool Maximization          = Max;

            virtual void Prepare(const gsl::span<const T> pop) = 0;

            gsl::span<const T> Population() const { return population; }

        protected:
            gsl::span<const T> population;
    };

    template<typename T>
    class EvaluatorBase : public StatefulOperator<double, T&, size_t> 
    {
        // some fitness measures are relative to the whole population (eg. diversity) 
        // and the evaluator needs to do some preparation work using the entire pop
        public:
            EvaluatorBase(Problem& p) : problem(p) { }

            virtual void Prepare(const gsl::span<const T> pop) = 0;
            size_t TotalEvaluations()   const { return fitnessEvaluations + localEvaluations; }
            size_t FitnessEvaluations() const { return fitnessEvaluations;                    }
            size_t LocalEvaluations()   const { return localEvaluations;                      }

        protected: 
            gsl::span<const T> population;
            std::reference_wrapper<const Problem> problem;
            std::atomic_ulong fitnessEvaluations = 0;
            std::atomic_ulong localEvaluations = 0;
    };

    template<typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
    class RecombinatorBase : public StatefulOperator<std::optional<typename TSelector::SelectableType>, double, double>
    {
        public:
            using EvaluatorType = TEvaluator;
            using SelectorType  = TSelector;
            using CrossoverType = TCrossover;
            using MutatorType   = TMutator;

            RecombinatorBase(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut)
                : evaluator(eval), selector(sel), crossover(cx), mutator(mut) { } 

            TSelector&  Selector()  const { return selector.get();  }
            TCrossover& Crossover() const { return crossover.get(); }
            TMutator&   Mutator()   const { return mutator.get();   }
            TEvaluator& Evaluator() const { return evaluator.get(); }

            virtual void Prepare(gsl::span<const typename TSelector::SelectableType> pop) = 0;
        
        protected:
            std::reference_wrapper<TEvaluator> evaluator;
            std::reference_wrapper<TSelector>  selector;
            std::reference_wrapper<TCrossover> crossover;
            std::reference_wrapper<TMutator>   mutator;
    };
}
#endif

