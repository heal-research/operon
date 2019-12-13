/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef DIVERSITY_HPP
#define DIVERSITY_HPP

#include <Eigen/Core>

#include "core/operator.hpp"
#include "core/stats.hpp"
#include <execution>
#include <unordered_set>

namespace Operon {
template <typename T, typename Scalar = uint32_t>
class PopulationDiversityAnalyzer final : PopulationAnalyzerBase<T> {
public:
    double operator()(operon::rand_t&, gsl::index i) const
    {
        MeanVarianceCalculator calc;
        for (const auto& ind : population) {
            calc.Add(ind.Genotype.Length());
        }
        return 1 - sim.row(i).mean() / calc.Mean();
    }

    double HybridDiversity() const
    {
        MeanVarianceCalculator calc;
        for (Eigen::Index i = 0; i < sim.rows() - 1; ++i) {
            for (Eigen::Index j = i + 1; j < sim.rows(); ++j) {
                double total = population[i].Genotype.Length() + population[j].Genotype.Length();
                auto distance = (total - sim(i, j)) / total;
                calc.Add(distance);
            }
        }
        return calc.Mean();
    }

    double StructuralDiversity() const
    {
        MeanVarianceCalculator calc;
        for (Eigen::Index i = 0; i < sim.rows() - 1; ++i) {
            for (Eigen::Index j = i + 1; j < sim.rows(); ++j) {
                double total = population[i].Genotype.Length() + population[j].Genotype.Length();
                auto distance = (total - sim(j, i)) / total;
                calc.Add(distance);
            }
        }
        return calc.Mean();
    }

    void Prepare(gsl::span<const T> pop)
    {
        population = pop;

        // warning: this will fail if the population size is too large
        // (not to mention this whole analyzer will be SLOW)
        // for large population sizes it is recommended to use the sampling analyzer
        sim = MatrixType::Zero(pop.size(), pop.size());

        std::vector<std::vector<operon::hash_t>> hybrid(pop.size());
        std::vector<std::vector<operon::hash_t>> strukt(pop.size());

        std::vector<gsl::index> indices(pop.size());
        std::iota(indices.begin(), indices.end(), 0);

        // hybrid (strict) hashing
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end(), [&](gsl::index i) {
            auto tree = population[i].Genotype;
            hybrid[i] = HashTree(tree, true);
            strukt[i] = HashTree(tree, false);
        });

        // calculate intersections
        std::for_each(std::execution::par_unseq, indices.begin(), indices.end() - 1, [&](gsl::index i) {
            for (size_t j = i + 1; j < indices.size(); ++j) {
                sim(i, j) = Intersect(hybrid[i], hybrid[j]);
                sim(j, i) = Intersect(strukt[i], strukt[j]);
            }
        });
    }

    static inline Scalar Intersect(const std::vector<operon::hash_t>& lhs, const std::vector<operon::hash_t>& rhs)
    {
        Scalar count = 0;
        size_t i = 0;
        size_t j = 0;

        operon::hash_t lmax = lhs.back();
        operon::hash_t rmax = rhs.back();

        while(i < lhs.size() && j < rhs.size()) {
            auto a  = lhs[i];
            auto b  = rhs[j];
            count  += a == b;
            i      += a <= b;
            j      += a >= b;

            if (a > rmax || b > lmax) { break; }
        }
        return count;
    }

    static inline std::vector<operon::hash_t> HashTree(Tree& tree, bool strict = true)
    {
        tree.Sort(strict);
        const auto& nodes = tree.Nodes();
        std::vector<operon::hash_t> hashes(tree.Length());
        std::transform(nodes.begin(), nodes.end(), hashes.begin(), [](const auto& n) { return n.CalculatedHashValue; });
        std::sort(hashes.begin(), hashes.end());
        return hashes;
    }

private:
    using MatrixType = Eigen::Matrix<Scalar, Eigen::Dynamic, Eigen::Dynamic>;
    MatrixType sim;
    gsl::span<const T> population;
};
} // namespace Operon

#endif
