// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/core/individual.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include <cpp-sort/sorters/merge_sorter.h>
#include <Eigen/Core>
#include <ranges>
#include <eve/module/algo.hpp>

#include <iostream>

namespace Operon {

// rank-based non-dominated sorting - ordinal version - see https://arxiv.org/abs/2203.13654
auto RankOrdinalSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*not used*/) const -> NondominatedSorterBase::Result
{
    static_assert(EIGEN_VERSION_AT_LEAST(3, 4, 0), "RankOrdinal requires Eigen >= 3.4.0");
    using Vec = Eigen::Array<int, -1, 1>;
    using Mat = Eigen::Array<int, -1, -1>;

    const auto n = static_cast<int>(pop.size());
    const auto m = static_cast<int>(pop.front().Size());
    assert(m >= 2);

    // 1) sort indices according to the stable sorting rules
    Mat p(n, m); // permutation matrix
    Mat r(m, n); // ordinal rank matrix
    p.col(0) = Vec::LinSpaced(n, 0, n - 1);
    r(0, p.col(0)) = Vec::LinSpaced(n, 0, n - 1);

    std::vector<Operon::Scalar> buf(n); // buffer to store fitness values to avoid pointer indirections during sorting
    cppsort::merge_sorter sorter;
    for (auto i = 1; i < m; ++i) {
        std::transform(pop.begin(), pop.end(), buf.begin(), [i](auto const& ind) { return ind[i]; });
        p.col(i) = p.col(i - 1); // this is a critical part of the approach
        sorter(p.col(i), [&](auto j) { return buf[j]; });
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

    auto dominated = [&](int i, int j) {
        // std::span<int> a(r.col(i).data(), r.col(i).size());
        // std::span<int> b(r.col(j).data(), r.col(j).size());
        return m == 2
            ? (r.col(i) < r.col(j)).all()
            : eve::algo::all_of(eve::views::zip(std::span(r.col(i).data(), m), std::span(r.col(j).data(), m)), [](auto t) { return kumi::apply(std::less{}, t); });
    };

    // 3) compute ranks / fronts
    // Vec rank = Vec::Zero(n); // individual ranks
    std::vector<int> rank(n, 0);
    for (auto i : p(Eigen::seq(0, n - 2), 0)) {
        if (maxp(i) == n-1) {
            continue;
        }
        for (auto j : p(Eigen::seq(maxp(i)+1, n-1), maxc(i))) {
            rank[j] += (rank[i] == rank[j] && dominated(i, j));
        }
    }
    std::vector<std::vector<size_t>> fronts(std::ranges::max(rank) + 1);
    for (auto i = 0; i < n; ++i) {
        fronts[rank[i]].push_back(i);
    }
    return fronts;
}
} // namespace Operon
