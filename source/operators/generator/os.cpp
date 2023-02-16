// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/operators/generator.hpp"
#include "operon/core/comparison.hpp"

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

        Individual child(p1->Size());

        if (doCrossover) {
            p2 = population[ MaleSelector()(random) ];
            child.Genotype = Crossover()(random, p1->Genotype, p2->Genotype);
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? Mutator()(random, std::move(child.Genotype))
                : Mutator()(random, p1->Genotype);
        }

        child.Fitness = Evaluator()(random, child, buf);
        bool accept{false};

        if (p2) {
            Individual q(child.Size());
            for (size_t i = 0; i < child.Size(); ++i) {
                auto f1 = (*p1)[i];
                auto f2 = (*p2)[i];
                q[i] = std::max(f1, f2) - static_cast<Operon::Scalar>(comparisonFactor_) * std::abs(f1 - f2);
                accept = Operon::ParetoDominance{}(child.Fitness, q.Fitness) != Dominance::Right;
            }
        } else {
            accept = Operon::ParetoDominance{}(child.Fitness, p1->Fitness) != Dominance::Right;
        }
        return accept ? std::make_optional(child) : std::nullopt;
    }

} // namespace Operon
