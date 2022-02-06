// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_PARETO_NONDOMINATED_SORTER_BASE
#define OPERON_PARETO_NONDOMINATED_SORTER_BASE

#include <vector>

#include "operon/core/types.hpp"
#include "operon/core/individual.hpp"

namespace Operon {

struct Individual;

namespace detail {
} // namespace detail

class NondominatedSorterBase {
    public:
        using Result = std::vector<std::vector<size_t>>;

        explicit NondominatedSorterBase(Operon::Scalar eps = 0) : eps_(eps) { }

        mutable struct {
            size_t LexicographicalComparisons = 0; // both lexicographical and single-objective
            size_t SingleValueComparisons = 0;
            size_t DominanceComparisons = 0;
            size_t RankComparisons = 0;
            size_t InnerOps = 0;
            double MeanRank  = 0;
            double MeanND = 0;
        } Stats;

        void Reset() { Stats = {0, 0, 0, 0, 0, 0., 0.}; }

        virtual auto Sort(Operon::Span<Operon::Individual const>) const -> Result = 0;

        auto operator()(Operon::Span<Operon::Individual const> pop) const -> Result {
            size_t m = pop.front().Fitness.size();
            ENSURE(m > 1);
            return Sort(pop);
        }

        auto Epsilon() const -> Operon::Scalar { return eps_; }

    private:
        Operon::Scalar eps_;
};

} // namespace Operon

#endif
