// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <Eigen/Core>
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/core/individual.hpp"

#include <cpp-sort/sorters.h>

namespace Operon {
    auto RankOrdinalSorter::Sort(Operon::Span<Operon::Individual const> pop) const -> NondominatedSorterBase::Result
    {
        static_assert(EIGEN_VERSION_AT_LEAST(3,4,0), "RankOrdinal requires Eigen version >= 3.4.0");
        using Vec = Eigen::Array<Eigen::Index, -1, 1>;
        using Mat = Eigen::Array<Eigen::Index, -1, -1>;

        const auto n = static_cast<Eigen::Index>(pop.size());
        const auto m = static_cast<Eigen::Index>(pop.front().Size());
        assert(m >= 2);

        // 1) sort indices according to the stable sorting rules
        Mat p(n, m); // permutation matrix
        Mat r(m, n); // ordinal rank matrix
        p.col(0) = Vec::LinSpaced(n, 0, n-1);
        r(0, p.col(0)) = Vec::LinSpaced(n, 0, n-1);

        cppsort::spin_sorter sorter;

        for (auto i = 1; i < m; ++i) {
            p.col(i) = p.col(i - 1); // this is a critical part of the approach
            sorter(p.col(i), [&](auto idx) { return pop[idx][i]; });
            //std::stable_sort(p.col(i).begin(), p.col(i).end(), [&](auto a, auto b) { return pop[a][i] < pop[b][i]; });
            r(i, p.col(i)) = Vec::LinSpaced(n, 0, n - 1);
        }
        // 2) save min and max positions as well as the column index for the max position
        Vec maxc(n);
        Vec minp(n);
        Vec maxp(n);
        for (auto i = 0; i < n; ++i) {
            auto c = r.col(i);
            auto max = std::max_element(c.begin(), c.end());
            maxp(i) = *max;
            maxc(i) = std::distance(c.begin(), max);
        }

        // 3) compute ranks / fronts
        Vec rank = Vec::Zero(n); // individual ranks
        for (auto i : p(Eigen::seq(0, n - 2), 0)) {
            if (maxp(i) == n - 1) {
                continue;
            }
            for (auto j : p(Eigen::seq(maxp(i) + 1, n - 1), maxc(i))) {
                rank(j) += static_cast<int64_t>(rank(i) == rank(j) && (r.col(i) < r.col(j)).all());
            }
        }
        std::vector<std::vector<size_t>> fronts(rank.maxCoeff() + 1);
        for (auto i = 0; i < n; ++i) {
            fronts[rank(i)].push_back(i);
        }
        return fronts;
    }
} // namespace Operon
