// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef BROOD_GENERATOR_HPP
#define BROOD_GENERATOR_HPP

#include "core/operator.hpp"
#include "algorithms/pareto.hpp"

namespace Operon {
class BroodOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit BroodOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
    {
    }

    std::optional<Individual> operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf = Operon::Span<Operon::Scalar>{}) const override
    {
        std::uniform_real_distribution<double> uniformReal;

        auto population = this->FemaleSelector().Population();

        auto first = this->femaleSelector(random);
        auto second = this->maleSelector(random);


        // assuming the basic generator never fails
        auto makeOffspring = [&]() {
            Individual child(population[first].Fitness.size());
            bool doCrossover = std::bernoulli_distribution(pCrossover)(random);
            bool doMutation = std::bernoulli_distribution(pMutation)(random);

            if (doCrossover) {
                child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
            }

            if (doMutation) {
                child.Genotype = doCrossover
                    ? this->mutator(random, std::move(child.Genotype))
                    : this->mutator(random, population[first].Genotype);
            }

            auto f = this->evaluator(random, child, buf);
            for (size_t i = 0; i < f.size(); ++i) {
                child[i] = std::isfinite(f[i]) ? f[i] : Operon::Numeric::Max<Operon::Scalar>();
            }
            return child;
        };

        auto best = makeOffspring();

        for (size_t i = 1; i < broodSize; ++i) {
            auto other = makeOffspring();
            if (other[0] < best[0]) {
                std::swap(best, other);
            }
        }

        return std::make_optional(best);
    }

    void BroodSize(size_t value) { broodSize = value; }
    size_t BroodSize() const { return broodSize; }

private:
    size_t broodSize;
};
} // namespace Operon
#endif
