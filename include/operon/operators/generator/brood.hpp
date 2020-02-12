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

#ifndef BROOD_GENERATOR_HPP
#define BROOD_GENERATOR_HPP

#include "core/operator.hpp"

namespace Operon {
template <typename TEvaluator, typename TCrossover, typename TMutator, typename TFemaleSelector, typename TMaleSelector = TFemaleSelector>
class BroodOffspringGenerator : public OffspringGeneratorBase<TEvaluator, TCrossover, TMutator, TFemaleSelector, TMaleSelector> {
public:
    explicit BroodOffspringGenerator(TEvaluator& eval, TCrossover& cx, TMutator& mut, TFemaleSelector& femSel, TMaleSelector& maleSel)
        : OffspringGeneratorBase<TEvaluator, TCrossover, TMutator, TFemaleSelector, TMaleSelector>(eval, cx, mut, femSel, maleSel)
          , basicGenerator(eval, cx, mut, femSel, maleSel)
    {
    }

    using T = typename TFemaleSelector::SelectableType;
    std::optional<T> operator()(Operon::Random& random, double pCrossover, double pMutation) const override
    {
        std::uniform_real_distribution<double> uniformReal;

        constexpr gsl::index Idx = TFemaleSelector::SelectableIndex;
        auto population = this->FemaleSelector().Population();

        // assuming the basic generator never fails
        auto best = basicGenerator(random, pCrossover, pMutation).value();

        for (size_t i = 1; i < broodSize; ++i) {
            auto other = basicGenerator(random, pCrossover, pMutation).value();
            if (other[Idx] < best[Idx]) {
                std::swap(best, other);
            }
        }

        return std::make_optional(best);
    }

    void BroodSize(size_t value) { broodSize = value; }
    size_t BroodSize() const { return broodSize; }

private:
    BasicOffspringGenerator<TEvaluator, TCrossover, TMutator, TFemaleSelector, TMaleSelector> basicGenerator;
    size_t broodSize;
};
} // namespace Operon
#endif
