// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARETO_DOMINANCE_DEGREE_SORT
#define OPERON_PARETO_DOMINANCE_DEGREE_SORT

#include "core/individual.hpp"
#include "core/operator.hpp"

namespace Operon {

namespace detail {
} // namespace detail

// Zhou et al. 2016 - "Ranking vectors by means of dominance degree matrix"
// https://doi.org/10.1109/TEVC.2016.2567648
// BB: this implementation is slow although I can't find any obvious faults,
// it is an exact reproduction of the paper algorithm
struct DominanceDegreeSorter : public NondominatedSorterBase {
    inline std::vector<std::vector<size_t>> operator()(Operon::RandomGenerator&, Operon::Span<Operon::Individual const> pop) const
    {
        return Sort(pop);
    }

private:
    using Vec = Eigen::Array<size_t, Eigen::Dynamic, 1, Eigen::ColMajor>;

    template <typename W, typename C>
    inline void ComputeComparisonMatrix(W const& w, C& c) const noexcept
    {
        size_t const n = w.size();
        Vec b = Vec::LinSpaced(n, 0, n - 1);
        auto p = b.data();
        pdqsort(p, p + n, [&](auto a, auto b) { return w(a) < w(b); });

        c.row(b(0)).setConstant(1);
        for (size_t i = 1; i < n; ++i) {
            c.row(b(i)).setConstant(0);
        }

        for (size_t i = 1; i < n; ++i) {
            if (w(b(i)) == w(b(i - 1))) {
                c.row(b(i)) = c.row(b(i - 1));
            } else {
                for (size_t j = i; j < n; ++j) {
                    c(b(i), b(j)) = 1;
                }
            }
        }
    }

    inline std::vector<std::vector<size_t>> Sort(Operon::Span<Operon::Individual const> pop) const noexcept
    {
        const size_t n = pop.size();
        const int m = (int)pop.front().Fitness.size();

        // build matrix a using fitness values
        Eigen::Array<Operon::Scalar, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> a(n, m);
        for (size_t i = 0; i < n; ++i) {
            auto ptr = pop[i].Fitness.data();
            a.row(i) = Eigen::Map<Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1, Eigen::ColMajor> const>(ptr, m);
        }

        Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> c(n, n);
        Eigen::Array<int, Eigen::Dynamic, Eigen::Dynamic, Eigen::ColMajor> d(n, n);

        ComputeComparisonMatrix(a.col(0), d);
        for (int i = 1; i < m; ++i) {
            ComputeComparisonMatrix(a.col(i), c);
            d += c;
        }

        // algorithm 3
        for (size_t i = 0; i < n; ++i) {
            for (size_t j = i; j < n; ++j) {
                if (d(i, j) == m && d(j, i) == m) {
                    d(i, j) = d(j, i) = 0;
                }
            }
        }

        size_t count = 0; // number of assigned solutions
        std::vector<std::vector<size_t>> fronts;
        Vec sorted = Vec::Zero(n);

        while (count < n) {
            std::vector<size_t> front;
            auto b = d.block(0, 0, n, n);
            for (size_t i = 0; i < n; ++i) {
                if (!sorted(i) && b.col(i).maxCoeff() < m) {
                    front.push_back(i);
                    sorted(i) = 1;
                }
            }
            for (auto i : front) {
                d.row(i).setConstant(0);
                //d.col(i).setConstant(0);
            }
            count += front.size();
            fronts.push_back(front);
        }

        return fronts;
    }
};

} // namespace operon

#endif
