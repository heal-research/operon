// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/operators/generator.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

namespace Operon {
    auto BroodOffspringGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, double pLocal, double pLamarck, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual>
    {
        auto const& fsel = *FemaleSelector();
        auto const& msel = *MaleSelector();

        auto const pop = fsel.Population();
        auto const& p1 = pop[ fsel(random) ];
        auto const& p2 = pop[ msel(random) ];

        // the brood offspring generator creates a brood of offspring from the same two parents
        auto makeOffspring = [&]() {
            RecombinationResult res{ {}, p1, p2 };
            OffspringGeneratorBase::Generate(random, pCrossover, pMutation, pLocal, pLamarck, buf, res);
            return res ? res.Child.value() : res.Parent1.value();
        };

        std::vector<Individual> offspring(broodSize_);
        std::generate(offspring.begin(), offspring.end(), makeOffspring);
        SingleObjectiveComparison comp{0};

        if (pop.front().Size() > 1) {
            std::stable_sort(offspring.begin(), offspring.end(), LexicographicalComparison{});
            auto fronts = RankIntersectSorter{}(offspring);
            auto best = *std::min_element(fronts[0].begin(), fronts[0].end(), [&](auto i, auto j) { return comp(offspring[i], offspring[j]); });
            return std::make_optional(offspring[best]);
        }
        auto best = *std::min_element(offspring.begin(), offspring.end(), comp);
        return std::make_optional(best);
    }
} // namespace Operon
