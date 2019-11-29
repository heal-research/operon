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

#ifndef OPERON_REINSERTER_KEEP_BEST
#define OPERON_REINSERTER_KEEP_BEST

#include "core/operator.hpp"

namespace Operon {
template <typename T, gsl::index Idx, bool Max, typename ExecutionPolicy = std::execution::parallel_unsequenced_policy>
class KeepBestReinserter : public ReinserterBase<T, Idx, Max> {
    public:
        // keep the best |pop| individuals from pop+pool
        virtual void operator()(operon::rand_t&, std::vector<T>& pop, std::vector<T>& pool) const override {
            std::conditional_t<Max, std::less<>, std::greater<>> comp;
            ExecutionPolicy ep;
            // sort the population and the recombination pool
            std::sort(ep, pop.begin(), pop.end(), [&](const auto& lhs, const auto& rhs) { return comp(lhs[Idx], rhs[Idx]); });
            std::sort(ep, pool.begin(), pool.end(), [&](const auto& lhs, const auto& rhs) { return comp(lhs[Idx], rhs[Idx]); });

            for (size_t i = 0, j = 0; i < pool.size() && j < pop.size();) {
                if (comp(pop[j][Idx], pool[i][Idx])) {
                    pop[j++] = std::move(pool[i]);
                }
                ++i;
            }
        }
};
} // namespace operon

#endif

