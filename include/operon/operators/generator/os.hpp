// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OS_GENERATOR_HPP
#define OS_GENERATOR_HPP

#include "core/operator.hpp"
#include "algorithms/pareto.hpp"

namespace Operon {
class OffspringSelectionGenerator : public OffspringGeneratorBase {
public:
    explicit OffspringSelectionGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
        , maxSelectionPressure(100)
        , comparisonFactor(1.0)
    {
    }

    std::optional<Individual> operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf = Operon::Span<Operon::Scalar>{}) const override
    {
        std::uniform_real_distribution<double> uniformReal;
        bool doCrossover = uniformReal(random) < pCrossover;
        bool doMutation = uniformReal(random) < pMutation;

        if (!(doCrossover || doMutation))
            return std::nullopt;

        auto population = this->FemaleSelector().Population();

        size_t first = this->femaleSelector(random);


        std::optional<Individual> p1{ population[first] };
        std::optional<Individual> p2;

        Individual child(p1.value().Fitness.size());

        if (doCrossover) {
            auto second = this->maleSelector(random);
            child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
            p2 = population[second];
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? this->mutator(random, std::move(child.Genotype))
                : this->mutator(random, population[first].Genotype);
        }

        child.Fitness = this->evaluator(random, child, buf);
        bool accept{false};

        if (p2.has_value()) {
            Individual q(child.Fitness.size());
            for (size_t i = 0; i < child.Fitness.size(); ++i) {
                auto f1 = p1.value()[i];
                auto f2 = p2.value()[i];
                q[i] = std::max(f1, f2) - static_cast<Operon::Scalar>(comparisonFactor) * std::abs(f1 - f2);
                accept = DominanceCalculator::Compare(child, q) == DominanceResult::NoDomination;
            }
        } else {
            accept = DominanceCalculator::Compare(child, p1.value()) == DominanceResult::NoDomination;
        }
        if (accept) return { child };
        return { };
    }

    void MaxSelectionPressure(size_t value) { maxSelectionPressure = value; }
    size_t MaxSelectionPressure() const { return maxSelectionPressure; }

    void ComparisonFactor(double value) { comparisonFactor = value; }
    double ComparisonFactor() const { return comparisonFactor; }

    void Prepare(const Operon::Span<const Individual> pop) const override
    {
        OffspringGeneratorBase::Prepare(pop);
        lastEvaluations = this->Evaluator().FitnessEvaluations();
    }

    double SelectionPressure() const
    {
        if (this->FemaleSelector().Population().empty()) {
            return 0;
        }
        return static_cast<double>(this->Evaluator().FitnessEvaluations() - lastEvaluations) / static_cast<double>(this->FemaleSelector().Population().size());
    }

    bool Terminate() const override
    {
        return OffspringGeneratorBase::Terminate() || SelectionPressure() > static_cast<double>(maxSelectionPressure);
    };

private:
    mutable size_t lastEvaluations;
    size_t maxSelectionPressure;
    double comparisonFactor;
};
} // namespace Operon

#endif
