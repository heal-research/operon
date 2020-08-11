/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef OPERATOR_HPP
#define OPERATOR_HPP

#include "gsl/gsl"
#include <atomic>
#include <random>
#include <type_traits>

#include "common.hpp"
#include "dataset.hpp"
#include "grammar.hpp"
#include "problem.hpp"
#include "tree.hpp"

namespace Operon {
// it's useful to have a data structure holding additional attributes for a solution candidate
// maybe we should have an array of trees here?
struct Individual {
    Tree Genotype;
    std::vector<Operon::Scalar> Fitness;

    Operon::Scalar& operator[](gsl::index i) noexcept { return Fitness[i]; }
    Operon::Scalar operator[](gsl::index i) const noexcept { return Fitness[i]; }

    Individual()
        : Individual(1)
    {
    }
    Individual(size_t fitDim)
        : Fitness(fitDim)
    {
    }
};

//using ComparisonCallback = std::add_pointer<bool(Individual const&, Individual const&)>::type;
using ComparisonCallback = std::add_pointer_t<bool(Individual const&, Individual const&)>;

// operator base classes for two types of operators: stateless and stateful
template <typename Ret, typename... Args>
struct OperatorBase {
    using ReturnType = Ret;
    using ArgumentType = std::tuple<Args...>;
    // all operators take a random device (source of randomness) as the first parameter
    virtual Ret operator()(Operon::Random& random, Args... args) const = 0;
    virtual ~OperatorBase() {}
};

// the creator builds a new tree using the existing grammar and allowed inputs
struct CreatorBase : public OperatorBase<Tree, size_t, size_t, size_t> {
public:
    CreatorBase(const Grammar& grammar, const gsl::span<const Variable> variables)
        : grammar_(grammar)
        , variables_(variables)
    {
    }

protected:
    std::reference_wrapper<const Grammar> grammar_;
    const gsl::span<const Variable> variables_;
};

// crossover takes two parent trees and returns a child
struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> {
};

// the mutator can work in place or return a copy (child)
struct MutatorBase : public OperatorBase<Tree, Tree> {
};

// the selector a vector of individuals and returns the index of a selected individual per each call of operator()
// this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
class SelectorBase : public OperatorBase<gsl::index> {
public:
    using SelectableType = Individual;

    explicit SelectorBase(ComparisonCallback cb)
        : comp(cb)
    {
    }

    virtual void Prepare(const gsl::span<const Individual> pop) const
    {
        this->population = gsl::span<const Individual>(pop);
    };

    gsl::span<const Individual> Population() const { return population; }

    bool Compare(Individual const& lhs, Individual const& rhs) const { 
        EXPECT(comp != nullptr);
        return comp(lhs, rhs); 
    }

protected:
    mutable gsl::span<const Individual> population;
    ComparisonCallback comp;
};

class ReinserterBase : public OperatorBase<void, std::vector<Individual>&, std::vector<Individual>&> {
public:
    explicit ReinserterBase(ComparisonCallback cb)
        : comp(cb)
    {
    }

protected:
    ComparisonCallback comp;
};

class EvaluatorBase : public OperatorBase<Operon::Scalar, Individual&> {
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

    virtual void Prepare(const gsl::span<const Individual> pop, gsl::index idx = 0)
    {
        population = pop;
        objIndex = idx;
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
    gsl::span<const Individual> population;
    std::reference_wrapper<const Problem> problem;
    mutable std::atomic_ulong fitnessEvaluations = 0;
    mutable std::atomic_ulong localEvaluations = 0;
    size_t iterations = DefaultLocalOptimizationIterations;
    size_t budget = DefaultEvaluationBudget;
    mutable gsl::index objIndex;
};

// TODO: Maybe remove all the template parameters and go for accepting references to operator bases
class OffspringGeneratorBase : public OperatorBase<std::optional<Individual>, /* crossover prob. */ double, /* mutation prob. */ double> {
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

    virtual void Prepare(gsl::span<const Individual> pop) const
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
    virtual void Prepare(gsl::span<const T> pop) = 0;
};
} // namespace Operon
#endif
