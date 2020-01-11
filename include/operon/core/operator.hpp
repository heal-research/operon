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

#include "common.hpp"
#include "dataset.hpp"
#include "grammar.hpp"
#include "problem.hpp"
#include "tree.hpp"

namespace Operon {
// it's useful to have a data structure holding additional attributes for a solution candidate
// maybe we should have an array of trees here?
template <size_t D = 1UL>
struct Individual {
    Tree Genotype;
    std::array<Operon::Scalar, D> Fitness;
    static constexpr size_t Dimension = D;

    Operon::Scalar& operator[](gsl::index i) noexcept { return Fitness[i]; }
    Operon::Scalar operator[](gsl::index i) const noexcept { return Fitness[i]; }

    // returns true if this dominates rhs
    inline bool operator<(const Individual& rhs) const noexcept
    {
        for (size_t i = 0; i < D; ++i) {
            if (Fitness[i] < rhs.Fitness[i]) {
                continue;
            }
            return false;
        }
        return true;
    }
};

// operator base classes for two types of operators: stateless and stateful
template <typename Ret, typename... Args>
struct OperatorBase {
    using ReturnType = Ret;
    using ArgumentType = std::tuple<Args...>;
    // all operators take a random device (source of randomness) as the first parameter
    virtual Ret operator()(Operon::Random& random, Args... args) const = 0;
};

// the creator builds a new tree using the existing grammar and allowed inputs
struct CreatorBase : public OperatorBase<Tree, const Grammar&, const gsl::span<const Variable>> {
};

// crossover takes two parent trees and returns a child
struct CrossoverBase : public OperatorBase<Tree, const Tree&, const Tree&> {
};

// the mutator can work in place or return a copy (child)
struct MutatorBase : public OperatorBase<Tree, Tree> {
};

// the selector a vector of individuals and returns the index of a selected individual per each call of operator()
// this operator is meant to be a lightweight object that is initialized with a population and some other parameters on-the-fly
template <typename T, gsl::index Idx>
class SelectorBase : public OperatorBase<gsl::index> {
public:
    using SelectableType = T;
    static constexpr gsl::index SelectableIndex = Idx;

    virtual void Prepare(const gsl::span<const T> pop) const
    {
        this->population = gsl::span<const T>(pop);
    };

    gsl::span<const T> Population() const { return population; }

protected:
    mutable gsl::span<const T> population;
};

template <typename T, gsl::index Idx>
class ReinserterBase : public OperatorBase<void, std::vector<T>&, std::vector<T>&> {
};

template <typename T>
class EvaluatorBase : public OperatorBase<Operon::Scalar, T&> {
    // some fitness measures are relative to the whole population (eg. diversity)
    // and the evaluator needs to do some preparation work using the entire pop
public:
    static constexpr size_t DefaultLocalOptimizationIterations = 50;
    static constexpr size_t DefaultEvaluationBudget = 100'000;

    EvaluatorBase(Problem& p)
        : problem(p)
    {
    }

    virtual void Prepare(const gsl::span<const T> pop) = 0;

    size_t TotalEvaluations() const { return fitnessEvaluations + localEvaluations; }
    size_t FitnessEvaluations() const { return fitnessEvaluations; }
    size_t LocalEvaluations() const { return localEvaluations; }

    void LocalOptimizationIterations(size_t value) { iterations = value; }
    size_t LocalOptimizationIterations() const { return iterations; }

    void Budget(size_t value) { budget = value; }
    size_t Budget() const { return budget; }
    bool BudgetExhausted() const { return TotalEvaluations() > Budget(); }

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
    size_t budget = DefaultEvaluationBudget;
};

// TODO: Maybe remove all the template parameters and go for accepting references to operator bases
template <typename TEvaluator, typename TCrossover, typename TMutator, typename TFemaleSelector, typename TMaleSelector = TFemaleSelector>
class OffspringGeneratorBase : public OperatorBase<std::optional<typename TFemaleSelector::SelectableType>, /* crossover prob. */ double, /* mutation prob. */ double> {
public:
    using EvaluatorType = TEvaluator;
    using FemaleSelectorType = TFemaleSelector;
    using MaleSelectorType = TMaleSelector;
    using CrossoverType = TCrossover;
    using MutatorType = TMutator;

    using T = typename TFemaleSelector::SelectableType;
    using U = typename TMaleSelector::SelectableType;

    OffspringGeneratorBase(TEvaluator& eval, TCrossover& cx, TMutator& mut, TFemaleSelector& femSel, TMaleSelector& maleSel)
        : evaluator(eval)
        , crossover(cx)
        , mutator(mut)
        , femaleSelector(femSel)
        , maleSelector(maleSel)
    {
    }

    TFemaleSelector& FemaleSelector() const { return femaleSelector.get(); }
    TMaleSelector& MaleSelector() const { return maleSelector.get(); }
    TCrossover& Crossover() const { return crossover.get(); }
    TMutator& Mutator() const { return mutator.get(); }
    TEvaluator& Evaluator() const { return evaluator.get(); }

    virtual void Prepare(gsl::span<const T> pop) const
    {
        static_assert(std::is_same_v<T, U>);
        this->FemaleSelector().Prepare(pop);
        this->MaleSelector().Prepare(pop);
    }
    virtual bool Terminate() const { return evaluator.get().BudgetExhausted(); }

protected:
    std::reference_wrapper<TEvaluator> evaluator;
    std::reference_wrapper<TFemaleSelector> femaleSelector;
    std::reference_wrapper<TFemaleSelector> maleSelector;
    std::reference_wrapper<TCrossover> crossover;
    std::reference_wrapper<TMutator> mutator;
};

template <typename T>
class PopulationAnalyzerBase : public OperatorBase<double> {
public:
    virtual void Prepare(gsl::span<const T> pop) = 0;

};
} // namespace Operon
#endif
