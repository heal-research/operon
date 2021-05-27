// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_REINSERTER_KEEP_BEST
#define OPERON_REINSERTER_KEEP_BEST

#include "core/operator.hpp"
#include "pdqsort.h"

namespace Operon {
class KeepBestReinserter : public ReinserterBase {
    public:
        explicit KeepBestReinserter(ComparisonCallback&& cb) : ReinserterBase(cb) { }
        explicit KeepBestReinserter(ComparisonCallback const& cb) : ReinserterBase(cb) { }
        // keep the best |pop| individuals from pop+pool
        void operator()(Operon::RandomGenerator&, std::vector<Individual>& pop, std::vector<Individual>& pool) const override {
            // sort the population and the recombination pool
            pdqsort(pop.begin(), pop.end(), this->comp);
            pdqsort(pool.begin(), pool.end(), this->comp);

            // merge the best individuals from pop+pool into pop
            size_t i = 0, j = 0;
            while (i < pool.size() && j < pop.size()) {
                if (this->comp(pool[i], pop[j])) {
                    std::swap(pool[i], pop[j]);
                    ++i;
                }
                ++j;
            }
        }
};
} // namespace operon

#endif

