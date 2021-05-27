// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OS_GENERATOR_HPP
#define OS_GENERATOR_HPP

#include "core/operator.hpp"

namespace Operon {
class OffspringSelectionGenerator : public OffspringGeneratorBase {
public:
    explicit OffspringSelectionGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
        , maxSelectionPressure(100)
        , comparisonFactor(1.0)
    {
    }

    std::optional<Individual> operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation) const override
    {
        std::uniform_real_distribution<double> uniformReal;
        bool doCrossover = uniformReal(random) < pCrossover;
        bool doMutation = uniformReal(random) < pMutation;

        if (!(doCrossover || doMutation))
            return std::nullopt;

        auto population = this->FemaleSelector().Population();

        size_t first = this->femaleSelector(random);

        Individual child(1);

        auto f1 = population[first][0];
        auto f2 = f1;

        if (doCrossover) {
            auto second = this->maleSelector(random);
            child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
            f2 = population[second][0];
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? this->mutator(random, std::move(child.Genotype))
                : this->mutator(random, population[first].Genotype);
        }

        auto f = this->evaluator(random, child);

        if (std::isfinite(f) && f < (std::max(f1, f2) - comparisonFactor * std::abs(f1 - f2))) {
            child[0] = f;
            return std::make_optional(child);
        }

        return std::nullopt;
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
        return (this->Evaluator().FitnessEvaluations() - lastEvaluations) / static_cast<double>(this->FemaleSelector().Population().size());
    }

    bool Terminate() const override
    {
        return OffspringGeneratorBase::Terminate() || SelectionPressure() > maxSelectionPressure;
    };

private:
    mutable size_t lastEvaluations;
    size_t maxSelectionPressure;
    double comparisonFactor;
};
} // namespace Operon

#endif
