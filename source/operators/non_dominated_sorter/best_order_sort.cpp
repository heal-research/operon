// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/operators/non_dominated_sorter.hpp"

#include <cpp-sort/sorters/merge_sorter.h>
#include <ranges>
#include <eve/module/algo.hpp>

namespace Operon {
// best order sort https://doi.org/10.1145/2908961.2931684
auto BestOrderSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) const -> NondominatedSorterBase::Result
{
    auto const n = static_cast<int>(pop.size());
    auto const m = static_cast<int>(pop.front().Size());

    // initialization
    std::vector<std::vector<std::vector<int>>> solutionSets(m);

    // std::vector<std::vector<int>> comparisonSets(n);
    // for (auto& cset : comparisonSets) {
    //     cset.resize(m);
    //     std::iota(cset.begin(), cset.end(), 0);
    // }

    std::vector<bool> isRanked(n, false); // rank status
    std::vector<int> rank(n, 0);          // rank of solutions

    int sc{0}; // number of solutions already ranked
    int rc{1}; // number of fronts so far (at least one front)

    std::vector<std::vector<int>> sortedByObjective(m);
    std::vector<std::vector<int>> sortedIndices(n);

    auto& idx = sortedByObjective[0];
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);
    for(auto i : idx) { sortedIndices[i].push_back(i); }

    // sort the individuals for each objective
    cppsort::merge_sorter sorter;
    for (auto j = 1; j < m; ++j) {
        sortedByObjective[j] = sortedByObjective[j-1];
        sorter(sortedByObjective[j], [&](auto i) { return pop[i][j]; });

        for (int i = 0; i < n; ++i) {
            sortedIndices[sortedByObjective[j][i]].push_back(i);
        }
    }

    // utility method
    auto addSolutionToRankSet = [&](auto s, auto j) {
        auto r = rank[s];
        auto& ss = solutionSets[j];
        if (r >= std::ssize(ss)) {
            ss.resize(r+1UL);
        }
        ss[r].push_back(s);
    };

    // algorithm 4 in the original paper
    auto dominationCheck = [&](auto s, auto t) {
        auto const& a = sortedIndices[s];
        auto const& b = sortedIndices[t];
        return m == 2
            ? std::ranges::none_of(std::ranges::iota_view{0, m}, [&](auto i) { return a[i] < b[i]; })
            : eve::algo::none_of(eve::views::zip(a, b), [](auto t) { auto [x, y] = t; return x < y; });
    };

    // algorithm 3 in the original paper
    auto findRank = [&](auto s, auto j) {
        bool done{false};

        for (auto k = 0; k < rc; ++k) {
            bool dominated = false;

            if (k >= std::ssize(solutionSets[j])) {
                solutionSets[j].resize(k+1UL);
            }

            for (auto t : solutionSets[j][k]) {
                // check if s is dominated by t
                if (dominated = dominationCheck(s, t); dominated) {
                    break;
                }
            }

            if (!dominated) {
                rank[s] = k;
                done = true;
                addSolutionToRankSet(s, j);
                break;
            }
        }

        if (!done) {
            rank[s] = rc;
            addSolutionToRankSet(s, j);
            ++rc;
        }
    };

    // main loop
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < m; ++j) {
            auto s = sortedByObjective[j][i]; // take i-th element from qj
            // auto cs = comparisonSets[s];
            // std::ranges::remove(cs, j); // reduce comparison set
            if (isRanked[s]) {
                addSolutionToRankSet(s, j);
            } else {
                findRank(s, j);
                isRanked[s] = true;
                ++sc;
            }
        }

        if (sc == n) {
            break; // all done, sorting ended
        }
    }

    // return fronts
    std::vector<std::vector<std::size_t>> fronts;
    fronts.resize(*std::max_element(rank.begin(), rank.end()) + 1UL);
    for (std::size_t i = 0UL; i < rank.size(); ++i) {
        fronts[rank[i]].push_back(i);
    }
    return fronts;
}
} // namespace Operon
