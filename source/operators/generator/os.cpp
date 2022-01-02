// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operon/operators/generator.hpp"

namespace Operon {

    auto OffspringSelectionGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual>
    {
        std::uniform_real_distribution<double> uniformReal;
        bool doCrossover = uniformReal(random) < pCrossover;
        bool doMutation = uniformReal(random) < pMutation;

        if (!(doCrossover || doMutation)) {
            return std::nullopt;
        }

        auto population = FemaleSelector().Population();

        size_t first = FemaleSelector()(random);


        std::optional<Individual> p1{ population[first] };
        std::optional<Individual> p2;

        Individual child(p1.value().Fitness.size());

        if (doCrossover) {
            auto second = MaleSelector()(random);
            child.Genotype = Crossover()(random, population[first].Genotype, population[second].Genotype);
            p2 = population[second];
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? Mutator()(random, std::move(child.Genotype))
                : Mutator()(random, population[first].Genotype);
        }

        child.Fitness = Evaluator()(random, child, buf);
        bool accept{false};

        if (p2.has_value()) {
            Individual q(child.Size());
            for (size_t i = 0; i < child.Size(); ++i) {
                auto f1 = p1.value()[i];
                auto f2 = p2.value()[i];
                q[i] = std::max(f1, f2) - static_cast<Operon::Scalar>(comparisonFactor_) * std::abs(f1 - f2);
                accept = child.ParetoCompare(q) != Dominance::Right;
            }
        } else {
            accept = child.ParetoCompare(p1.value()) != Dominance::Right;
        }
        return accept ? std::make_optional(child) : std::nullopt;
    }

} // namespace Operon
