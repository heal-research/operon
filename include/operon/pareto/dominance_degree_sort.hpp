// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_DOMINANCE_DEGREE_SORT
#define OPERON_PARETO_DOMINANCE_DEGREE_SORT

#include "core/individual.hpp"
#include "core/operator.hpp"

#include <core/types.hpp>

#include <iomanip>

namespace Operon {

// Zhou et al. 2016 - "Ranking vectors by means of dominance degree matrix"
// https://doi.org/10.1109/TEVC.2016.2567648
// BB: this implementation is slow although I can't find any obvious faults,
// it is an exact reproduction of the paper algorithm.
template<bool DominateOnEqual = false>
struct DominanceDegreeSorter : public NondominatedSorterBase {
    inline std::vector<std::vector<size_t>> operator()(Operon::RandomGenerator&, Operon::Span<Operon::Individual const> pop) const
    {
        size_t m = pop.front().Fitness.size();
        ENSURE(m > 1);
        return Sort(pop);
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
                c.row(b(i)) = c.row(b(i-1));
            } else {
                for (size_t j = i; j < n; ++j) {
                    c(b(i), b(j)) = 1;
                }
            }
        }
        return c;
    }

    inline auto ComparisonMatrixSum(Operon::Span<Operon::Individual const> pop, Mat const& idx) const noexcept {
        Mat d = ComputeComparisonMatrix(pop, idx, 0);
        for (int i = 1; i < idx.cols(); ++i) {
            d.noalias() += ComputeComparisonMatrix(pop, idx, i);
        }
        return d;
    }

    inline auto ComputeDegreeMatrix(Operon::Span<Operon::Individual const> pop, Mat const& idx) const noexcept
    {
        size_t const n = pop.size();
        size_t const m = pop.front().Fitness.size();
        Mat d = ComparisonMatrixSum(pop, idx);
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                if (d(i, j) == m && d(j, i) == m) {
                    d(i, j) = d(j, i) = 0;
                }
            }
        }
        return d;
    }

    inline std::vector<std::vector<size_t>> Sort(Operon::Span<Operon::Individual const> pop) const noexcept
    {
        const size_t n = pop.size();
        const size_t m = pop.front().Fitness.size();

        Mat idx = Vec::LinSpaced(n, 0, n-1).replicate(1, m);
        for (size_t i = 0; i < m; ++i) {
            auto data = idx.col(i).data();
            std::sort(data, data + n, [&](auto a, auto b) { return pop[a][i] < pop[b][i]; });
        }
        Mat d = ComputeDegreeMatrix(pop, idx);
        size_t count = 0; // number of assigned solutions
        std::vector<std::vector<size_t>> fronts;
        std::vector<size_t> indices(n);
        std::iota(indices.begin(), indices.end(), 0ul);

        std::vector<size_t> remaining;
        while (count < n) {
            std::vector<size_t> front;
            for (auto i : indices) {
                if (std::all_of(indices.begin(), indices.end(), [&](auto j) { return d(j, i) < m; })) {
                    front.push_back(i);
                } else {
                    remaining.push_back(i);
                }
            }
            indices.swap(remaining);
            remaining.clear();
            count += front.size();
            fronts.push_back(front);
        }
        return fronts;
    }
};

} // namespace operon

#endif
