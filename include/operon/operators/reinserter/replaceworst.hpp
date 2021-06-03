// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_REINSERTER_REPLACE_WORST
#define OPERON_REINSERTER_REPLACE_WORST

#include "core/operator.hpp"
#include "pdqsort.h"

namespace Operon {
class ReplaceWorstReinserter : public ReinserterBase {
    public:
        explicit ReplaceWorstReinserter(ComparisonCallback&& cb) : ReinserterBase(cb) { }
        explicit ReplaceWorstReinserter(ComparisonCallback const& cb) : ReinserterBase(cb) { }
        // replace the worst individuals in pop with the best individuals from pool
        void operator()(Operon::RandomGenerator&, Operon::Span<Individual> pop, Operon::Span<Individual> pool) const override {
            // typically the pool and the population are the same size
            if (pop.size() == pool.size()) {
                std::swap_ranges(pop.begin(), pop.end(), pool.begin());
                return;
            }
            if (pop.size() > pool.size()) Sort(pop); else Sort(pool);
            auto offset = static_cast<std::ptrdiff_t>(std::min(pop.size(), pool.size()));
            std::swap_ranges(pool.begin(), pool.begin() + offset, pop.end() - offset);
        }
};
} // namespace operon

#endif
