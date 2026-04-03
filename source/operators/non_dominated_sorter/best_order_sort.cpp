// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/operators/non_dominated_sorter.hpp"

#include <cpp-sort/sorters/merge_sorter.h>
#include <eve/module/algo.hpp>
#include <fmt/core.h>
#include <ranges>

namespace {

struct BestOrderState {
    Operon::Span<Operon::Individual const> Pop;
    Operon::Vector<Operon::Vector<int>> const& ComparisonSets; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    int M;
    int& Rc;                                                   // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    Operon::Vector<int>& Rank;                                 // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    Operon::Vector<Operon::Vector<Operon::Vector<int>>>& SolutionSets; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

auto AddToRankSet(int s, int j, BestOrderState& st) -> void {
    auto r = st.Rank[s];
    auto& ss = st.SolutionSets[j];
    if (r >= std::ssize(ss)) {
        ss.resize(r + 1UL);
    }
    ss[r].push_back(s);
}

auto DominationCheck(int s, int t, BestOrderState const& st) -> bool {
    auto const& a = st.Pop[s].Fitness;
    auto const& b = st.Pop[t].Fitness;
    return st.M == 2
        ? std::ranges::none_of(st.ComparisonSets[t], [&](auto i) -> auto { return a[i] < b[i]; })
        : eve::algo::none_of(eve::views::zip(a, b), [](auto p) -> auto { return kumi::apply(std::less{}, p); });
}

auto FindRank(int s, int j, BestOrderState& st) -> void {
    bool done{false};
    for (auto k = 0; k < st.Rc; ++k) {
        bool dominated = false;
        if (k >= std::ssize(st.SolutionSets[j])) {
            st.SolutionSets[j].resize(k + 1UL);
        }
        for (auto t : st.SolutionSets[j][k]) {
            if (dominated = DominationCheck(s, t, st); dominated) {
                break;
            }
        }
        if (!dominated) {
            st.Rank[s] = k;
            done = true;
            AddToRankSet(s, j, st);
            break;
        }
    }
    if (!done) {
        st.Rank[s] = st.Rc;
        AddToRankSet(s, j, st);
        ++st.Rc;
    }
}

} // anonymous namespace

namespace Operon {
// best order sort https://doi.org/10.1145/2908961.2931684
auto BestOrderSorter::Sort(Operon::Span<Operon::Individual const> pop, Operon::Scalar /*unused*/) const -> NondominatedSorterBase::Result
{
    auto const n = static_cast<int>(pop.size());
    auto const m = static_cast<int>(pop.front().Size());

    // initialization
    Operon::Vector<Operon::Vector<Operon::Vector<int>>> solutionSets(m);

    Operon::Vector<Operon::Vector<int>> comparisonSets(n);
    for (auto& cset : comparisonSets) {
        cset.resize(m);
        std::iota(cset.begin(), cset.end(), 0);
    }

    Operon::Vector<bool> isRanked(n, false); // rank status
    Operon::Vector<int> rank(n, 0);          // rank of solutions

    int sc{0}; // number of solutions already ranked
    int rc{1}; // number of fronts so far (at least one front)

    Operon::Vector<Operon::Vector<int>> sortedByObjective(m);

    auto& idx = sortedByObjective[0];
    idx.resize(n);
    std::iota(idx.begin(), idx.end(), 0);

    // sort the individuals for each objective
    cppsort::merge_sorter const sorter;
    for (auto j = 1; j < m; ++j) {
        sortedByObjective[j] = sortedByObjective[j-1];
        sorter(sortedByObjective[j], [&](auto i) -> auto { return pop[i][j]; });
    }

    BestOrderState st{ .Pop=pop, .ComparisonSets=comparisonSets, .M=m, .Rc=rc, .Rank=rank, .SolutionSets=solutionSets };

    // main loop
    for (auto i = 0; i < n; ++i) {
        for (auto j = 0; j < m; ++j) {
            auto s = sortedByObjective[j][i]; // take i-th element from qj
            std::erase(comparisonSets[s], j); // reduce comparison set
            if (isRanked[s]) {
                AddToRankSet(s, j, st);
            } else {
                FindRank(s, j, st);
                isRanked[s] = true;
                ++sc;
            }
        }

        if (sc == n) {
            break; // all done, sorting ended
        }
    }

    // return fronts
    Operon::Vector<Operon::Vector<std::size_t>> fronts;
    fronts.resize(*std::max_element(rank.begin(), rank.end()) + 1UL);
    for (std::size_t i = 0UL; i < rank.size(); ++i) {
        fronts[rank[i]].push_back(i);
    }
    return fronts;
}
} // namespace Operon
