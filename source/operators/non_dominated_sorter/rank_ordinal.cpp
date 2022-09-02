// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/individual.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include <Eigen/Core>

namespace Operon {

// rank-based non-dominated sorting - ordinal version - see https://arxiv.org/abs/2203.13654
auto RankOrdinalSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar eps) const -> NondominatedSorterBase::Result
{
#if EIGEN_VERSION_AT_LEAST(3, 4, 0)
    using Vec = Eigen::Array<Eigen::Index, -1, 1>;
    using Mat = Eigen::Array<Eigen::Index, -1, -1>;

    const auto n = static_cast<Eigen::Index>(pop.size());
    const auto m = static_cast<Eigen::Index>(pop.front().Size());
    assert(m >= 2);

    // 1) sort indices according to the stable sorting rules
    Mat p(n, m); // permutation matrix
    Mat r(m, n); // ordinal rank matrix
    p.col(0) = Vec::LinSpaced(n, 0, n - 1);
    r(0, p.col(0)) = Vec::LinSpaced(n, 0, n - 1);

    Operon::Less cmp;
    std::vector<Operon::Scalar> buf(n); // buffer to store fitness values to avoid pointer indirections during sorting
    for (auto i = 1; i < m; ++i) {
        std::transform(pop.begin(), pop.end(), buf.begin(), [i](auto const& ind) { return ind[i]; });
        p.col(i) = p.col(i - 1); // this is a critical part of the approach
        std::stable_sort(p.col(i).begin(), p.col(i).end(), [&](auto a, auto b) { return cmp(buf[a], buf[b], eps); });
        r(i, p.col(i)) = Vec::LinSpaced(n, 0, n - 1);
    }

    // 2) save min and max positions as well as the column index for the max position
    Vec maxc(n);
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
#else
    throw std::runtime_error("RankOrdinal requires Eigen >= 3.4.0");
#endif
}
} // namespace Operon
