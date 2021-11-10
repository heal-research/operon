// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operators/non_dominated_sorter.hpp"
#include "operators/generator/poly.hpp"

namespace Operon {
    std::optional<Individual> PolygenicOffspringGenerator::operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation, Operon::Span<Operon::Scalar> buf) const
    {
        std::uniform_real_distribution<double> uniformReal;
        auto population = this->FemaleSelector().Population();

        // assuming the basic generator never fails
        auto makeOffspring = [&]() {
            auto first = this->femaleSelector(random);
            auto second = this->maleSelector(random);
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

        std::vector<Individual> offspring;
        offspring.reserve(broodSize);
        for (size_t i = 0; i < broodSize; ++i) {
            offspring.push_back(makeOffspring());
        }
        SingleObjectiveComparison comp{0};

        if (population.front().Size() > 1) {
            std::stable_sort(offspring.begin(), offspring.end(), LexicographicalComparison{});
            auto fronts = RankSorter {}(offspring);
            auto best = *std::min_element(fronts[0].begin(), fronts[0].end(), [&](auto i, auto j) { return comp(offspring[i], offspring[j]); });
            return std::make_optional(offspring[best]);
        } else {
            auto best = *std::min_element(offspring.begin(), offspring.end(), comp);
            return std::make_optional(best);
        }
    }

}
