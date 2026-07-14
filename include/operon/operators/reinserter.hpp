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
// Elitism lives here (rather than as a separate per-algorithm mechanism,
// e.g. GP's old offspring[0]-overwrite hack) because "which individuals
// carry over between generations unconditionally" is exactly what
// reinsertion already governs. operator() is `final`: it protects the top
// EliteCount() individuals of `pop` (by the shared comparator) from ever
// being touched, then hands the remaining, still-contiguous subspan of pop
// to Combine() - the merge/replace strategy each subclass implements. Any
// algorithm (GP, NSGA2, ...) gets elitism uniformly this way, generalizing
// from "best-by-one-objective" (GP's SingleObjectiveComparison) to
// "best-by-rank-then-crowding" (NSGA2's CrowdedComparison) for free, since
// both are just the comparator already passed in.
class ReinserterBase : public OperatorBase<void, Operon::Span<Individual>, Operon::Span<Individual>> {
public:
    explicit ReinserterBase(ComparisonCallback cb, size_t eliteCount = 0)
        : comp_(std::move(cb))
        , eliteCount_(eliteCount)
    {
    }

    void operator()(Operon::RandomGenerator& random, Operon::Span<Individual> pop, Operon::Span<Individual> pool) const final
    {
        auto elites = std::min(eliteCount_, pop.size());
        if (elites > 0) { Sort(pop); }
        Combine(random, pop.subspan(elites), pool);
    }

    inline void Sort(Operon::Span<Individual> inds) const { std::stable_sort(inds.begin(), inds.end(), comp_); }

    [[nodiscard]] inline auto Compare(Individual const& lhs, Individual const& rhs) const -> bool
    {
        return comp_(lhs, rhs);
    }

    [[nodiscard]] auto EliteCount() const -> size_t { return eliteCount_; }

protected:
    // Merge/replace strategy applied to the non-elite tail of pop and the
    // full pool. `pop` here excludes whatever operator() already protected.
    virtual void Combine(Operon::RandomGenerator& random, Operon::Span<Individual> pop, Operon::Span<Individual> pool) const = 0;

private:
    ComparisonCallback comp_;
    size_t eliteCount_;
};

class OPERON_EXPORT KeepBestReinserter : public ReinserterBase {
public:
    explicit KeepBestReinserter(ComparisonCallback const& cb, size_t eliteCount = 0)
        : ReinserterBase(cb, eliteCount)
    {
    }

protected:
    // keep the best |pop| individuals from pop+pool
    void Combine(Operon::RandomGenerator& /*random*/, Operon::Span<Individual> pop, Operon::Span<Individual> pool) const override
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
    explicit ReplaceWorstReinserter(ComparisonCallback const& cb, size_t eliteCount = 0)
        : ReinserterBase(cb, eliteCount)
    {
    }

protected:
    // replace the worst individuals in pop with the best individuals from pool
    void Combine(Operon::RandomGenerator& /*random*/, Operon::Span<Individual> pop, Operon::Span<Individual> pool) const override
    {
        // Both spans must be sorted by the comparator regardless of their
        // relative sizes - when pop.size() == pool.size() (the common case),
        // skipping this would make offset cover the *entire* span and this
        // become an unconditional full swap, never consulting the
        // comparator at all.
        Sort(pop);
        Sort(pool);
        auto offset = static_cast<std::ptrdiff_t>(std::min(pop.size(), pool.size()));
        std::swap_ranges(pool.begin(), pool.begin() + offset, pop.end() - offset);
    }
};

// See core/concepts.hpp for why these are asserted here rather than constraining a template.
static_assert(Concepts::Reinserter<KeepBestReinserter>);
static_assert(Concepts::Reinserter<ReplaceWorstReinserter>);

} // namespace Operon

#endif
