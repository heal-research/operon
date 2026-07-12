// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_REINSERTER_HPP
#define OPERON_REINSERTER_HPP

#include <algorithm>
#include "operon/core/concepts.hpp"
#include "operon/core/operator.hpp"
#include "operon/core/individual.hpp"

namespace Operon {
class ReinserterBase : public OperatorBase<void, Operon::Span<Individual>, Operon::Span<Individual>> {
public:
    explicit ReinserterBase(ComparisonCallback cb)
        : comp_(std::move(cb))
    {
    }

    inline void Sort(Operon::Span<Individual> inds) const { std::stable_sort(inds.begin(), inds.end(), comp_); }

    [[nodiscard]] inline auto Compare(Individual const& lhs, Individual const& rhs) const -> bool
    {
        return comp_(lhs, rhs);
    }

private:
    ComparisonCallback comp_;
};

class OPERON_EXPORT KeepBestReinserter : public ReinserterBase {
public:
    explicit KeepBestReinserter(ComparisonCallback const& cb)
        : ReinserterBase(cb)
    {
    }
    // keep the best |pop| individuals from pop+pool
    void operator()(Operon::RandomGenerator& /*random*/, Operon::Span<Individual> pop, Operon::Span<Individual> pool) const override
    {
        // sort the population and the recombination pool
        Sort(pop);
        Sort(pool);

        // merge the best individuals from pop+pool into pop
        size_t i = 0;
        size_t j = 0;
        while (i < pool.size() && j < pop.size()) {
            if (Compare(pool[i], pop[j])) {
                std::swap(pool[i], pop[j]);
                ++i;
            }
            ++j;
        }
    }
};

class OPERON_EXPORT ReplaceWorstReinserter : public ReinserterBase {
public:
    explicit ReplaceWorstReinserter(ComparisonCallback const& cb)
        : ReinserterBase(cb)
    {
    }
    // replace the worst individuals in pop with the best individuals from pool
    void operator()(Operon::RandomGenerator& /*random*/, Operon::Span<Individual> pop, Operon::Span<Individual> pool) const override
    {
        // typically the pool and the population are the same size
        if (pop.size() > pool.size()) {
            Sort(pop);
        } else if (pop.size() < pool.size()) {
            Sort(pool);
        }
        auto offset = static_cast<std::ptrdiff_t>(std::min(pop.size(), pool.size()));
        std::swap_ranges(pool.begin(), pool.begin() + offset, pop.end() - offset);
    }
};

// The concrete reinserters satisfy Concepts::Reinserter; ReinserterBase itself
// stays virtual-dispatch (pyoperon/CLI factories need dynamic wiring), but
// pinning these asserts here catches signature drift against the concept at
// compile time.
static_assert(Concepts::Reinserter<KeepBestReinserter>);
static_assert(Concepts::Reinserter<ReplaceWorstReinserter>);

} // namespace Operon

#endif
