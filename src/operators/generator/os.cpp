// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operators/generator/os.hpp"

namespace Operon {

    std::optional<Individual> OffspringSelectionGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const
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
            Individual q(child.Size());
            for (size_t i = 0; i < child.Size(); ++i) {
                auto f1 = p1.value()[i];
                auto f2 = p2.value()[i];
                q[i] = std::max(f1, f2) - static_cast<Operon::Scalar>(comparisonFactor) * std::abs(f1 - f2);
                accept = child.ParetoCompare(q) != Dominance::Right;
            }
        } else {
            accept = child.ParetoCompare(p1.value()) != Dominance::Right;
        }
        if (accept) return { child };
        return { };
    }

} // namespace Operon
