// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operators/generator/basic.hpp"

namespace Operon {
    std::optional<Individual> BasicOffspringGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const
    {
        std::uniform_real_distribution<double> uniformReal;
        bool doCrossover = std::bernoulli_distribution(pCrossover)(random);
        bool doMutation = std::bernoulli_distribution(pMutation)(random);

        if (!(doCrossover || doMutation))
            return std::nullopt;

        auto population = this->FemaleSelector().Population();

        auto first = this->femaleSelector(random);
        Individual child;

        if (doCrossover) {
            auto second = this->maleSelector(random);
            child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? this->mutator(random, std::move(child.Genotype))
                : this->mutator(random, population[first].Genotype);
        }

        child.Fitness = this->evaluator(random, child, buf);
        for (auto& v : child.Fitness) {
            if (!std::isfinite(v)) { v = Operon::Numeric::Max<Operon::Scalar>(); }
        }
        return std::make_optional(child);
    }
}
