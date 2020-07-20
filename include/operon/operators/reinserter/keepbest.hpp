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

#include <execution>

#include "core/operator.hpp"

namespace Operon {
template <typename ExecutionPolicy = std::execution::unsequenced_policy>
class KeepBestReinserter : public ReinserterBase {
    public:
        explicit KeepBestReinserter(ComparisonCallback cb) : ReinserterBase(cb) { }
        // keep the best |pop| individuals from pop+pool
        virtual void operator()(Operon::Random&, std::vector<Individual>& pop, std::vector<Individual>& pool) const override {
            ExecutionPolicy ep;
            // sort the population and the recombination pool
            std::sort(ep, pop.begin(), pop.end(), this->comp);
            std::sort(ep, pool.begin(), pool.end(), this->comp);

            for (size_t i = 0, j = 0; i < pool.size() && j < pop.size();) {
                //if (pop[j][idx] > pool[i][idx]) {
                if (this->comp(pop[i], pop[j])) {
                    pop[j++] = std::move(pool[i]);
                }
                ++i;
            }
        }
};
} // namespace operon

#endif

