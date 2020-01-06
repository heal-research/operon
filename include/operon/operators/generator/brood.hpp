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
template <typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
class BroodOffspringGenerator : public OffspringGeneratorBase<TEvaluator, TSelector, TCrossover, TMutator> {
public:
    explicit BroodOffspringGenerator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut)
        : OffspringGeneratorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut)
          , basicGenerator(eval, sel, cx, mut)
    {
    }

    using T = typename TSelector::SelectableType;
    std::optional<T> operator()(Operon::Random& random, double pCrossover, double pMutation) const override
    {
        std::uniform_real_distribution<double> uniformReal;

        constexpr gsl::index Idx = TSelector::SelectableIndex;

        auto population = this->Selector().Population();

        // assuming the basic generator never fails
        auto best = basicGenerator(random, pCrossover, pMutation).value();

        for (size_t i = 1; i < broodSize; ++i) {
            auto other = basicGenerator(random, pCrossover, pMutation).value();
            if (other[0] < best[0]) {
                std::swap(best, other);
            }
        }

        return std::make_optional(best);
    }

    void BroodSize(size_t value) { broodSize = value; }
    size_t BroodSize() const { return broodSize; }

private:
    BasicOffspringGenerator<TEvaluator, TSelector, TCrossover, TMutator> basicGenerator;
    size_t broodSize;
};
} // namespace Operon
#endif
