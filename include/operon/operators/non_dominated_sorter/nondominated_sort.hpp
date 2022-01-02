// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_NONDOMINATED_SORT
#define OPERON_PARETO_NONDOMINATED_SORT

#include <robin_hood.h>

#include "sorter_base.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

template<bool DominateOnEqual = false>
struct OPERON_EXPORT FastNondominatedSorter : public NondominatedSorterBase {
    private:
    using Vec = Eigen::Array<size_t, -1, 1>;
    using Mat = Eigen::Array<size_t, -1, -1>;

    inline auto Sort(Operon::Span<Individual const> pop) const noexcept -> NondominatedSorterBase::Result override
    {
        const size_t n = pop.size();
        //const size_t m = pop.front().Fitness.size();

        std::vector<std::pair<size_t, size_t>> dom; dom.reserve(n * (n-1) / 2);
        std::vector<size_t> idx(n);
        std::iota(idx.begin(), idx.end(), 0UL);

        Vec rank = Vec::Zero(n);
        std::vector<std::vector<size_t>> fronts;
        while(!idx.empty()) {
            std::vector<size_t> front;
            for (size_t i = 0; i < idx.size()-1; ++i) {
                auto x = idx[i];
                for (size_t j = i + 1; j < idx.size(); ++j) {
                    ++Stats.InnerOps;
                    ++Stats.DominanceComparisons;
                    auto y = idx[j];
                    auto d = pop[x].ParetoCompare(pop[y]);
                    rank(x) += static_cast<uint64_t>(d == Dominance::Right);
                    rank(y) += static_cast<uint64_t>(d == Dominance::Left || (DominateOnEqual && d == Dominance::Equal));
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
} // namespace Operon

#endif
