// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_NONDOMINATED_SORT
#define OPERON_PARETO_NONDOMINATED_SORT

#include "core/individual.hpp"
#include "core/operator.hpp"

#include "robin_hood.h"

namespace Operon {

namespace detail {
}

struct FastNondominatedSorter : public NondominatedSorterBase {
    inline std::vector<std::vector<size_t>>
    operator()(Operon::RandomGenerator&, Operon::Span<Operon::Individual const> pop) const
    {
        size_t m = pop.front().Fitness.size();
        ENSURE(m > 1);

        switch (m) {
        case 2:
            return Sort<2>(pop);
        case 3:
            return Sort<3>(pop);
        case 4:
            return Sort<4>(pop);
        case 5:
            return Sort<5>(pop);
        case 6:
            return Sort<6>(pop);
        case 7:
            return Sort<7>(pop);
        default:
            return Sort<0>(pop);
        }
    }

    private:
    template <size_t N, int Storage = Eigen::ColMajor>
    using Mat = std::conditional_t<
        N == 0,
        Eigen::Array<size_t, Eigen::Dynamic, Eigen::Dynamic, Storage>,
        Eigen::Array<size_t, Eigen::Dynamic, N, Storage>>;

    using Vec = Eigen::Array<size_t, Eigen::Dynamic, 1, Eigen::ColMajor>;

    template<size_t N = 0>
    inline std::vector<std::vector<size_t>> Sort(Operon::Span<Individual const> pop) const noexcept
    {
        const size_t n = pop.size();
        //const size_t m = pop.front().Fitness.size();

        std::vector<std::pair<size_t, size_t>> dom; dom.reserve(n * (n-1) / 2);
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0ul);

        Vec rank = Vec::Zero(n);
        std::vector<std::vector<size_t>> fronts;
        while(!idx.empty()) {
            std::vector<size_t> front;
            for (size_t i = 0; i < idx.size()-1; ++i) {
                auto x = idx[i];
                for (size_t j = i + 1; j < idx.size(); ++j) {
                    ++this->Stats.InnerOps;
                    ++this->Stats.DominanceComparisons;
                    auto y = idx[j];
                    auto d = pop[x].ParetoCompare<N>(pop[y]);
                    rank(x) += d == Dominance::Right;
                    rank(y) += d == Dominance::Left || d == Dominance::Equal;
                }
            }
            std::copy_if(idx.begin(), idx.end(), std::back_inserter(front), [&](auto x) { return rank(x) == 0; });
            idx.erase(std::remove_if(idx.begin(), idx.end(), [&](auto x) { return rank(x) == 0; }), idx.end());
            std::for_each(idx.begin(), idx.end(), [&](auto x) { rank(x) = 0; });
            fronts.push_back(front);
        }

        return fronts;
    }
};

// fast non-dominated sort by Deb et al.
} // namespace operon

#endif
