// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/operators/generator.hpp"
#include "operon/operators/non_dominated_sorter.hpp"

namespace Operon {
    auto BroodOffspringGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const -> std::optional<Individual>
    {
        std::uniform_real_distribution<double> uniformReal;

        auto population = this->FemaleSelector().Population();

        auto first = FemaleSelector()(random);
        auto second = MaleSelector()(random);

        // assuming the basic generator never fails
        auto makeOffspring = [&]() {
            Individual child(population[first].Fitness.size());
            bool doCrossover = std::bernoulli_distribution(pCrossover)(random);
            bool doMutation = std::bernoulli_distribution(pMutation)(random);

            if (doCrossover) {
                child.Genotype = Crossover()(random, population[first].Genotype, population[second].Genotype);
            }

            if (doMutation) {
                child.Genotype = doCrossover
                    ? Mutator()(random, std::move(child.Genotype))
                    : Mutator()(random, population[first].Genotype);
            }

            auto f = Evaluator()(random, child, buf);
            for (size_t i = 0; i < f.size(); ++i) {
                child[i] = std::isfinite(f[i]) ? f[i] : std::numeric_limits<Operon::Scalar>::max();
            }
            return child;
        };

        std::vector<Individual> offspring(broodSize_);
        std::generate(offspring.begin(), offspring.end(), makeOffspring);
        SingleObjectiveComparison comp{0};

        if (population.front().Size() > 1) {
            std::stable_sort(offspring.begin(), offspring.end(), LexicographicalComparison{});
            auto fronts = RankIntersectSorter{}(offspring);
            auto best = *std::min_element(fronts[0].begin(), fronts[0].end(), [&](auto i, auto j) { return comp(offspring[i], offspring[j]); });
            return std::make_optional(offspring[best]);
        }
        auto best = *std::min_element(offspring.begin(), offspring.end(), comp);
        return std::make_optional(best);
    }
} // namespace Operon
