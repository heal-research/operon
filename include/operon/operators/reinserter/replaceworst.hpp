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

#ifndef OPERON_REINSERTER_REPLACE_WORST
#define OPERON_REINSERTER_REPLACE_WORST

#include "core/operator.hpp"

namespace Operon {
template <typename T, gsl::index Idx, bool Max, typename ExecutionPolicy = std::execution::parallel_unsequenced_policy>
class ReplaceWorstReinserter : public ReinserterBase<T, Idx> {
    public:
        // replace the worst individuals in pop with the best individuals from pool
        virtual void operator()(operon::rand_t&, std::vector<T>& pop, std::vector<T>& pool) const override {
            ExecutionPolicy ep;
            auto comp = [&](const auto& lhs, const auto& rhs) { return lhs[Idx] < rhs[Idx]; };
            if (pop.size() > pool.size()) {
                std::sort(ep, pop.begin(), pop.end(), comp);
            } else if (pop.size() < pool.size()) {
                std::sort(ep, pool.begin(), pool.end(), comp);
            }
            auto offset = std::min(pop.size(), pool.size());
            std::copy_if(ep, std::make_move_iterator(pool.begin()), std::make_move_iterator(pool.begin() + offset), pop.begin() + pop.size() - offset, [](const auto& ind) { return !ind.Genotype.Empty(); });
        }
};
} // namespace operon

#endif
