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
    struct OperatorBase 
    {
        using return_type = Ret;
        using argument_type = std::tuple<Args...>;
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

        double& operator[](gsl::index i) noexcept { return Fitness[i]; }
        double operator[](gsl::index i) const noexcept { return Fitness[i]; }
    };

    // the creator builds a new tree using the existing grammar and allowed inputs
    struct CreatorBase : public OperatorBase<Tree, const Grammar&, const gsl::span<const Variable>> { };

    // crossover takes two parent trees and returns a child
    struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> { };

    // the mutator can work in place or return a copy (child) 
    struct MutatorBase  : public OperatorBase<Tree, Tree> { };

    // the selector a vector of individuals and returns the index of a selected individual per each call of operator() 
    // this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
    template<typename T, gsl::index Idx, bool Max>
    class SelectorBase : public OperatorBase<gsl::index> 
    {
        public:
            using  SelectableType                       = T;
            static constexpr gsl::index SelectableIndex = Idx;
            static constexpr bool Maximization          = Max;

            virtual void Prepare(const gsl::span<const T> pop) = 0;

            gsl::span<const T> Population() const { return population; }

        protected:
            mutable gsl::span<const T> population;
    };

    template<typename T>
    class EvaluatorBase : public OperatorBase<double, T&> 
    {
        // some fitness measures are relative to the whole population (eg. diversity) 
        // and the evaluator needs to do some preparation work using the entire pop
        public:
            static constexpr size_t DefaultLocalOptimizationIterations = 50;
            static constexpr size_t DefaultEvaluationBudget            = 100'000;

            EvaluatorBase(Problem& p) : problem(p) { }

            virtual void Prepare(const gsl::span<const T> pop) = 0;
            size_t TotalEvaluations()   const { return fitnessEvaluations + localEvaluations; }
            size_t FitnessEvaluations() const { return fitnessEvaluations;                    }
            size_t LocalEvaluations()   const { return localEvaluations;                      }

            void   LocalOptimizationIterations(size_t value) { iterations = value; }
            size_t LocalOptimizationIterations() const       { return iterations;  }

            void   Budget(size_t value)    { budget = value;                       }
            size_t Budget() const          { return budget;                        }
            bool   BudgetExhausted() const { return TotalEvaluations() > Budget(); }

            void Reset()    
            {
                fitnessEvaluations = 0;
                localEvaluations = 0;
            }

        protected: 
            gsl::span<const T> population;
            std::reference_wrapper<const Problem> problem;
            mutable std::atomic_ulong fitnessEvaluations = 0;
            mutable std::atomic_ulong localEvaluations = 0;
            size_t iterations = DefaultLocalOptimizationIterations;
            size_t budget     = DefaultEvaluationBudget;
    };

    template<typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
    class RecombinatorBase : public OperatorBase<std::optional<typename TSelector::SelectableType>, double, double>
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

            virtual void Prepare(gsl::span<const typename TSelector::SelectableType> pop) const = 0;
            virtual bool Terminate() const { return evaluator.get().BudgetExhausted(); }
        
        protected:
            std::reference_wrapper<TEvaluator> evaluator;
            std::reference_wrapper<TSelector>  selector;
            std::reference_wrapper<TCrossover> crossover;
            std::reference_wrapper<TMutator>   mutator;
    };

    template<typename T>
    class PopulationAnalyzerBase : public OperatorBase<double, gsl::index>
    {
        public:
            virtual void Prepare(const gsl::span<T> pop) = 0;
    };
}
#endif

