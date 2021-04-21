// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef BASIC_GENERATOR_HPP
#define BASIC_GENERATOR_HPP

#include "core/operator.hpp"

namespace Operon {
// TODO: think of a way to eliminate duplicated code between the different recombinators
class BasicOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit BasicOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
    {
    }

    std::optional<Individual> operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation) const override
    {
        std::uniform_real_distribution<double> uniformReal;
        bool doCrossover = std::bernoulli_distribution(pCrossover)(random);
        bool doMutation = std::bernoulli_distribution(pMutation)(random);

        if (!(doCrossover || doMutation))
            return std::nullopt;

        auto population = this->FemaleSelector().Population();

        auto first = this->femaleSelector(random);
        Individual child(1);

        if (doCrossover) {
            auto second = this->maleSelector(random);
            child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? this->mutator(random, std::move(child.Genotype))
                : this->mutator(random, population[first].Genotype);
        }

        auto f = this->evaluator(random, child);
        if (!std::isfinite(f)) { f = Operon::Numeric::Max<Operon::Scalar>(); }
        child[0] = f;
        return std::make_optional(child);
    }
};

} // namespace Operon

#endif
