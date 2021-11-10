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

    std::optional<Individual> operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf = Operon::Span<Operon::Scalar>{}) const override;

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
        auto n = this->FemaleSelector().Population().size();
        if (!n) { return 0; }
        auto e = this->Evaluator().FitnessEvaluations() - lastEvaluations;
        return static_cast<double>(e) / static_cast<double>(n);
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
