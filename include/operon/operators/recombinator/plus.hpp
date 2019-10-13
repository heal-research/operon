/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef PLUS_RECOMBINATOR_HPP
#define PLUS_RECOMBINATOR_HPP

#include "core/operator.hpp"

namespace Operon {
template <typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
class PlusRecombinator : public RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator> {
public:
    explicit PlusRecombinator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut)
        : RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut)
    {
    }

    using T = typename TSelector::SelectableType;
    std::optional<T> operator()(operon::rand_t& random, double pCrossover, double pMutation) const override
    {
        std::uniform_real_distribution<double> uniformReal;
        bool doCrossover = uniformReal(random) < pCrossover;
        bool doMutation = uniformReal(random) < pMutation;

        if (!(doCrossover || doMutation))
            return std::nullopt;

        constexpr bool Max = TSelector::Maximization;
        constexpr gsl::index Idx = TSelector::SelectableIndex;

        auto population = this->Selector().Population();

        auto first = this->selector(random);
        auto second = this->selector(random);

        typename TSelector::SelectableType child;

        if (doCrossover) {
            child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? this->mutator(random, std::move(child.Genotype))
                : this->mutator(random, population[first].Genotype);
        }

        auto f = this->evaluator(random, child);
        child.Fitness[Idx] = std::isfinite(f) ? f : (Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max());

        if (doCrossover) {
            // we have two parents
            if (Max && child.Fitness[Idx] < std::max(population[first].Fitness[Idx], population[second].Fitness[Idx])) {
                child = population[first].Fitness[Idx] > population[second].Fitness[Idx] ? population[first] : population[second];
            } else if (!Max && child.Fitness[Idx] > std::min(population[first].Fitness[Idx], population[second].Fitness[Idx])) {
                child = population[first].Fitness[Idx] < population[second].Fitness[Idx] ? population[first] : population[second];
            }
        } else {
            // we have one parent
            if (Max && child.Fitness[Idx] < population[first].Fitness[Idx]) {
                child = population[first];
            } else if (!Max && child.Fitness[Idx] > population[first].Fitness[Idx]) {
                child = population[first];
            }
        }

        return std::make_optional(child);
    }

    void Prepare(const gsl::span<const T> pop) const override
    {
        this->Selector().Prepare(pop);
    }
};

}

#endif
