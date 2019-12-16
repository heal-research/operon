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
    namespace detail {
        using hash_vector_t = std::vector<operon::hash_t, Eigen::aligned_allocator<operon::hash_t>>;
    }

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

        std::vector<detail::hash_vector_t> hybrid(pop.size());
        std::vector<detail::hash_vector_t> strukt(pop.size());

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
                sim(i, j) = IntersectVector(hybrid[i], hybrid[j]);
                sim(j, i) = IntersectVector(strukt[i], strukt[j]);
            }
        });
    }

    static Scalar Intersect(const detail::hash_vector_t& lhs, const detail::hash_vector_t& rhs)
    {
        Scalar count = 0;
        size_t i = 0;
        size_t j = 0;

        operon::hash_t lmax = lhs.back();
        operon::hash_t rmax = rhs.back();

        while (i < lhs.size() && j < rhs.size()) {
            auto a = lhs[i];
            auto b = rhs[j];
            count += a == b;
            i += a <= b;
            j += a >= b;

            if (a > rmax || b > lmax) {
                break;
            }
        }
        return count;
    }

    static Scalar IntersectVector(detail::hash_vector_t const& lhs, detail::hash_vector_t const& rhs)
    {
        Scalar count = 0;
        size_t i = 0, j = 0;

        uint64_t const* a = lhs.data();
        uint64_t const* b = rhs.data();

        // trim lengths to be a multiple of 4
        size_t aLen = (lhs.size() / 4) * 4;
        size_t bLen = (rhs.size() / 4) * 4;

        while (i < aLen && j < bLen) {
            // load segments of four 64-bit elements
            __m256i v_a                  = _mm256_load_si256((__m256i*)&a[i]);
            __m256i v_b                  = _mm256_load_si256((__m256i*)&b[j]);

            // move pointers
            uint64_t a_max               = _mm256_extract_epi64(v_a, 3);
            uint64_t b_max               = _mm256_extract_epi64(v_b, 3);
            i                           += (a_max <= b_max) * 4;
            j                           += (a_max >= b_max) * 4;

            // compute mask of common elements
            constexpr auto cyclic_shift  = _MM_SHUFFLE(0, 3, 2, 1);
            __m256i cmp_mask1            = _mm256_cmpeq_epi64(v_a, v_b);

            v_b                          = _mm256_permute4x64_epi64(v_b, cyclic_shift); // shuffling
            __m256i cmp_mask2            = _mm256_cmpeq_epi64(v_a, v_b);

            v_b                          = _mm256_permute4x64_epi64(v_b, cyclic_shift);
            __m256i cmp_mask3            = _mm256_cmpeq_epi64(v_a, v_b);

            v_b                          = _mm256_permute4x64_epi64(v_b, cyclic_shift);
            __m256i cmp_mask4            = _mm256_cmpeq_epi64(v_a, v_b);

            __m256i cmp_mask             = _mm256_or_si256(
                _mm256_or_si256(cmp_mask1, cmp_mask2),
                _mm256_or_si256(cmp_mask3, cmp_mask4)); // OR-ing of comparison masks

            // convert the 256-bit mask to 4-bit
            int mask                     = _mm256_movemask_pd((__m256d)cmp_mask);
            auto cnt                     = __builtin_popcount(mask); // a number of elements is a weight of the mask
            count                       += cnt;
        }

        // intersect the tail using scalar intersection
        while(i < lhs.size() && j < rhs.size()) {
            if(a[i] < b[j]) {
                ++i;
            } else if(b[j] < a[i]) {
                ++j;
            } else {
                ++count;
                ++i;
                ++j;
            }
        }
        return count;
    }

    static inline std::vector<operon::hash_t, Eigen::aligned_allocator<operon::hash_t>> HashTree(Tree& tree, bool strict = true)
    {
        tree.Sort(strict);
        const auto& nodes = tree.Nodes();
        detail::hash_vector_t hashes(tree.Length());
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
