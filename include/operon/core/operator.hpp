// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include <atomic>
#include <random>
#include <type_traits>

#include "core/dataset.hpp"
#include "core/pset.hpp"
#include "core/individual.hpp"
#include "core/problem.hpp"
#include "core/tree.hpp"

namespace Operon {
// it's useful to have a data structure holding additional attributes for a solution candidate
// maybe we should have an array of trees here?

// operator base classes for two types of operators: stateless and stateful
template <typename Ret, typename... Args>
struct OperatorBase {
    using ReturnType = Ret;
    using ArgumentType = std::tuple<Args...>;
    // all operators take a random device (source of randomness) as the first parameter
    virtual Ret operator()(Operon::RandomGenerator& random, Args... args) const = 0;
    virtual ~OperatorBase() {}
};

// the creator builds a new tree using the existing pset and allowed inputs
struct CreatorBase : public OperatorBase<Tree, size_t, size_t, size_t> {
public:
    CreatorBase(const PrimitiveSet& pset, const Operon::Span<const Variable> variables)
        : pset_(pset)
        , variables_(variables)
    {
    }

protected:
    std::reference_wrapper<const PrimitiveSet> pset_;
    const Operon::Span<const Variable> variables_;
};

// crossover takes two parent trees and returns a child
struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> {
};

// the mutator can work in place or return a copy (child)
struct MutatorBase : public OperatorBase<Tree, Tree> {
};

// the selector a vector of individuals and returns the index of a selected individual per each call of operator()
// this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
class SelectorBase : public OperatorBase<size_t> {
public:
    using SelectableType = Individual;
    SelectorBase() { }
    explicit SelectorBase(ComparisonCallback&& cb)
        : comp(std::move(cb))
    {
    }

    explicit SelectorBase(ComparisonCallback const& cb) : comp(cb) { }

    virtual void Prepare(const Operon::Span<const Individual> pop) const
    {
        this->population = Operon::Span<const Individual>(pop);
    };

    Operon::Span<const Individual> Population() const { return population; }

    bool Compare(Individual const& lhs, Individual const& rhs) const { 
        return comp(lhs, rhs); 
    }

protected:
    mutable Operon::Span<const Individual> population;
    ComparisonCallback comp;
};

class ReinserterBase : public OperatorBase<void, Operon::Span<Individual>, Operon::Span<Individual>> {
public:
    explicit ReinserterBase(ComparisonCallback&& cb)
        : comp(std::move(cb))
    {
    }

    explicit ReinserterBase(ComparisonCallback const& cb)
        : comp(cb)
    {
    }

    inline void Sort(Operon::Span<Individual> inds) const { pdqsort(inds.begin(), inds.end(), comp); }

protected:
    ComparisonCallback comp;
};

class EvaluatorBase : public OperatorBase<Operon::Vector<Operon::Scalar>, Individual&, Operon::Span<Operon::Scalar>> {
    // some fitness measures are relative to the whole population (eg. diversity)
    // and the evaluator needs to do some preparation work using the entire pop
public:
    static constexpr size_t DefaultLocalOptimizationIterations = 50;
    static constexpr size_t DefaultEvaluationBudget = 100'000;

    using ReturnType = OperatorBase::ReturnType;

    EvaluatorBase(Problem& p)
        : problem(p)
    {
    }

    virtual void Prepare(const Operon::Span<const Individual> pop)
    {
        population = pop;
    }

    size_t TotalEvaluations() const { return fitnessEvaluations + localEvaluations; }
    size_t FitnessEvaluations() const { return fitnessEvaluations; }
    size_t LocalEvaluations() const { return localEvaluations; }

    void SetLocalOptimizationIterations(size_t value) { iterations = value; }
    size_t GetLocalOptimizationIterations() const { return iterations; }

    void SetBudget(size_t value) { budget = value; }
    size_t GetBudget() const { return budget; }
    bool BudgetExhausted() const { return TotalEvaluations() > GetBudget(); }

    void Reset()
    {
        fitnessEvaluations = 0;
        localEvaluations = 0;
    }

protected:
    Operon::Span<const Individual> population;
    std::reference_wrapper<const Problem> problem;
    mutable std::atomic_ulong fitnessEvaluations = 0;
    mutable std::atomic_ulong localEvaluations = 0;
    size_t iterations = DefaultLocalOptimizationIterations;
    size_t budget = DefaultEvaluationBudget;
};

// TODO: Maybe remove all the template parameters and go for accepting references to operator bases
class OffspringGeneratorBase : public OperatorBase<std::optional<Individual>, /* crossover prob. */ double, /* mutation prob. */ double, /* memory buffer */ Operon::Span<Operon::Scalar>> {
public:
    OffspringGeneratorBase(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : evaluator(eval)
        , crossover(cx)
        , mutator(mut)
        , femaleSelector(femSel)
        , maleSelector(maleSel)
    {
    }

    SelectorBase& FemaleSelector() const { return femaleSelector.get(); }
    SelectorBase& MaleSelector() const { return maleSelector.get(); }
    CrossoverBase& Crossover() const { return crossover.get(); }
    MutatorBase& Mutator() const { return mutator.get(); }
    EvaluatorBase& Evaluator() const { return evaluator.get(); }

    virtual void Prepare(Operon::Span<const Individual> pop) const
    {
        this->FemaleSelector().Prepare(pop);
        this->MaleSelector().Prepare(pop);
    }
    virtual bool Terminate() const { return evaluator.get().BudgetExhausted(); }

protected:
    std::reference_wrapper<EvaluatorBase> evaluator;
    std::reference_wrapper<CrossoverBase> crossover;
    std::reference_wrapper<MutatorBase> mutator;
    std::reference_wrapper<SelectorBase> femaleSelector;
    std::reference_wrapper<SelectorBase> maleSelector;
};

template <typename T>
class PopulationAnalyzerBase : public OperatorBase<double> {
public:
    virtual void Prepare(Operon::Span<const T> pop) = 0;
};
} // namespace Operon
#endif
