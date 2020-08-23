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

#ifndef BASIC_GENERATOR_HPP
#define BASIC_GENERATOR_HPP

#include "core/operator.hpp"

namespace Operon {
// TODO: think of a way to eliminate duplicated code between the different recombinators
class BasicOffspringGenerator : public OffspringGeneratorBase {
public:
    explicit BasicOffspringGenerator(EvaluatorBase& eval, CrossoverBase& cx, MutatorBase& mut, SelectorBase& femSel, SelectorBase& maleSel)
        : OffspringGeneratorBase(eval, cx, mut, femSel, maleSel)
    {
    }

    std::optional<Individual> operator()(Operon::RandomGenerator& random, double pCrossover, double pMutation) const override
    {
        std::uniform_real_distribution<double> uniformReal;
        bool doCrossover = std::bernoulli_distribution(pCrossover)(random);
        bool doMutation = std::bernoulli_distribution(pMutation)(random);

        if (!(doCrossover || doMutation))
            return std::nullopt;

        auto population = this->FemaleSelector().Population();

        auto first = this->femaleSelector(random);
        Individual child(1);

        if (doCrossover) {
            auto second = this->maleSelector(random);
            child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
        }

        if (doMutation) {
            child.Genotype = doCrossover
                ? this->mutator(random, std::move(child.Genotype))
                : this->mutator(random, population[first].Genotype);
        }

        auto f = this->evaluator(random, child);
        if (!std::isfinite(f)) { f = Operon::Numeric::Max<Operon::Scalar>(); }
        child[0] = f;
        return std::make_optional(child);
    }
};

} // namespace Operon

#endif
