// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon/operators/generator.hpp"

namespace Operon {
    auto BasicOffspringGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual>
    {
        std::uniform_real_distribution<double> uniformReal;
        bool doCrossover = std::bernoulli_distribution(pCrossover)(random);
        bool doMutation = std::bernoulli_distribution(pMutation)(random);

        if (!(doCrossover || doMutation)) {
            return std::nullopt;
        }

        auto population = this->FemaleSelector().Population();

        auto first = this->FemaleSelector()(random);
        Individual child;

        if (doCrossover) {
            auto second = this->MaleSelector()(random);
            child.Genotype = this->Crossover()(random, population[first].Genotype, population[second].Genotype);
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? this->Mutator()(random, std::move(child.Genotype))
                : this->Mutator()(random, population[first].Genotype);
        }

        child.Fitness = this->Evaluator()(random, child, buf);
        for (auto& v : child.Fitness) {
            if (!std::isfinite(v)) { v = Operon::Numeric::Max<Operon::Scalar>(); }
        }
        return std::make_optional(child);
    }
} // namespace Operon
