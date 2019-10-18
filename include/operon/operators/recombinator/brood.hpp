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

#ifndef BROOD_RECOMBINATOR_HPP
#define BROOD_RECOMBINATOR_HPP

#include "core/operator.hpp"

namespace Operon {
template <typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
class BroodRecombinator : public RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator> {
public:
    explicit BroodRecombinator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut)
        : RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut)
    {
    }

    using T = typename TSelector::SelectableType;
    std::optional<T> operator()(operon::rand_t& random, double pCrossover, double pMutation) const override
    {
        std::uniform_real_distribution<double> uniformReal;

        constexpr bool Max = TSelector::Maximization;
        constexpr gsl::index Idx = TSelector::SelectableIndex;

        auto population = this->Selector().Population();
        auto first = this->selector(random);
        auto second = this->selector(random);

        T child;

        std::vector<T> brood;
        brood.reserve(broodSize);
        for (size_t i = 0; i < broodSize; ++i) {
            bool doCrossover = uniformReal(random) < pCrossover;
            bool doMutation = uniformReal(random) < pMutation;

            if (!(doCrossover || doMutation)) {
                child = population[first];
            } else {
                if (doCrossover) {
                    child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
                }

                if (doMutation) {
                    child.Genotype = doCrossover
                        ? this->mutator(random, std::move(child.Genotype))
                        : this->mutator(random, population[first].Genotype);
                }
            }
            brood.push_back(child);
        }

        auto eval = [&](gsl::index idx) {
            auto f = this->evaluator(random, brood[idx]);
            brood[idx].Fitness[Idx] = std::isfinite(f) ? f : (Max ? operon::scalar::min() : operon::scalar::max());
        };

        std::uniform_int_distribution<gsl::index> uniformInt(0, brood.size() - 1);
        auto bestIdx = uniformInt(random);
        eval(bestIdx);

        std::conditional_t<Max, std::less<>, std::greater<>> comp;
        for (size_t i = 1; i < broodTournamentSize; ++i) {
            auto currIdx = uniformInt(random);
            eval(currIdx);
            if (comp(brood[bestIdx][Idx], brood[currIdx][Idx])) {
                bestIdx = currIdx;
            }
        }
        return std::make_optional(brood[bestIdx]);
    }

    void Prepare(const gsl::span<const T> pop) const override
    {
        this->Selector().Prepare(pop);
    }

    void BroodSize(size_t value) { broodSize = value; }
    size_t BroodSize() const { return broodSize; }

    void BroodTournamentSize(size_t value) { broodTournamentSize = value; }
    size_t BroodTournamentSize() const { return broodTournamentSize; }

private:
    size_t broodSize;
    size_t broodTournamentSize;
};
} // namespace Operon
#endif
