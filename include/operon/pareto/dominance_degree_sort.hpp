// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_DOMINANCE_DEGREE_SORT
#define OPERON_PARETO_DOMINANCE_DEGREE_SORT

#include "core/individual.hpp"
#include "core/operator.hpp"
#include <core/types.hpp>

namespace Operon {

// Zhou et al. 2016 - "Ranking vectors by means of dominance degree matrix"
// https://doi.org/10.1109/TEVC.2016.2567648
// BB: this implementation is slow although I can't find any obvious faults,
// it is an exact reproduction of the paper algorithm with a tiny optimization (line 40)
struct DominanceDegreeSorter : public NondominatedSorterBase {
    inline std::vector<std::vector<size_t>> operator()(Operon::RandomGenerator&, Operon::Span<Operon::Individual const> pop) const
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
    using Vec = Eigen::Matrix<size_t, -1, 1, Eigen::ColMajor>;
    using Mat = Eigen::Matrix<size_t, -1, -1, Eigen::ColMajor>;

    inline auto ComputeComparisonMatrix(Operon::Span<Operon::Individual const> pop, Mat const& idx, size_t colIdx) const noexcept
    {
        size_t const n = pop.size();
        Mat c = Mat::Zero(n, n);
        Mat::ConstColXpr b = idx.col(colIdx);
        c.row(b(0)).fill(1);
        for (size_t i = 1; i < n; ++i) {
            if (pop[b(i)][colIdx] == pop[b(i-1)][colIdx]) {
                c(b(i), Eigen::all) = c(b(i-1), Eigen::all);
            } else {
                c(b(i), b(Eigen::seq(i,n-1))).fill(1);
            }
        }
        return c;
    }

    template<size_t... indices>
    inline auto ComparisonMatrixSumFused(Operon::Span<Operon::Individual const> pop, Mat const& idx, std::index_sequence<indices...>) const noexcept
    {
        Mat sum = Mat::Zero(pop.size(), pop.size());
        sum.noalias() += (ComputeComparisonMatrix(pop, idx, indices) + ...);
        return sum;
    }

    inline auto ComparisonMatrixSum(Operon::Span<Operon::Individual const> pop, Mat const& idx) const noexcept {
        Mat d = ComputeComparisonMatrix(pop, idx, 0);
        for (int i = 1; i < idx.cols(); ++i) {
            d.noalias() += ComputeComparisonMatrix(pop, idx, i);
        }
        return d;
    }

    template<size_t M>
    inline auto ComputeDegreeMatrix(Operon::Span<Operon::Individual const> pop, Mat const& idx) const noexcept
    {
        size_t const n = pop.size();
        size_t const m = pop.front().Fitness.size();
        Mat d;
        if constexpr (M == 0) {
            // iter sum version cost: 1 assignment + (m-1)*n*n additions (2 ints)
            d = ComparisonMatrixSum(pop, idx);
        } else {
            // fused version cost: n*n additions (m ints)
            d = ComparisonMatrixSumFused(pop, idx, std::make_index_sequence<M>{});
        }
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                if (d(i, j) == m && d(j, i) == m) {
                    d(i, j) = d(j, i) = 0;
                }
            }
        }
        return d;
    }

    template<size_t M = 0>
    inline std::vector<std::vector<size_t>> Sort(Operon::Span<Operon::Individual const> pop) const noexcept
    {
        const size_t n = pop.size();
        const size_t m = pop.front().Fitness.size();

        Mat idx = Vec::LinSpaced(n, 0, n-1).replicate(1, m);
        for (size_t i = 0; i < m; ++i) {
            std::sort(idx.col(i).begin(), idx.col(i).end(), [&](auto a, auto b) { return pop[a][i] < pop[b][i]; });
        }
        Mat d = ComputeDegreeMatrix<M>(pop, idx);
        Vec indices = Vec::LinSpaced(n, 0, n-1);
        size_t count = 0; // number of assigned solutions
        std::vector<size_t> remaining;
        std::vector<std::vector<size_t>> fronts;

        while (count < n) {
            std::vector<size_t> front;
            for (int i = 0; i < indices.size(); ++i) {
                if (std::all_of(indices.begin(), indices.end(), [&](auto j) { return d(j, indices(i)) < m; })) {
                    front.push_back(indices(i));
                } else {
                    remaining.push_back(i);
                }
            }
            if (!remaining.empty()) {
                indices = indices(remaining).eval();
                remaining.clear();
            }
            count += front.size();
            fronts.push_back(front);
        }
        return fronts;
    }
};

} // namespace operon

#endif
