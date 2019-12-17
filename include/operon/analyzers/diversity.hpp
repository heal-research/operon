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
        std::atomic<size_t> mismatches = 0;
        std::for_each(std::execution::seq, indices.begin(), indices.end() - 1, [&](gsl::index i) {
            for (size_t j = i + 1; j < indices.size(); ++j) {
                sim(i, j) = IntersectVector(hybrid[i], hybrid[j]);
                sim(j, i) = IntersectVector(strukt[i], strukt[j]);

                auto s = Intersect(strukt[i], strukt[j]);

                if (s != sim(j, i)) {
                    ++mismatches;
                    for (auto v : strukt[i]) fmt::print("{} ", v); fmt::print("\n");
                    for (auto v : strukt[j]) fmt::print("{} ", v); fmt::print("\n");
                    fmt::print("Hash count mismatch {} (scalar) != {} (vector)\n", s, sim(j, i));
                    throw std::runtime_error("");
                    
                }
            }
        });
        fmt::print("mismatches = {}\n", mismatches);
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
        Scalar count                     = 0;
        size_t i                         = 0;
        size_t j                         = 0;

        uint64_t const* a                = lhs.data();
        uint64_t const* b                = rhs.data();

        // trim lengths to be a multiple of 4
        size_t aLen                      = (lhs.size() / 4) * 4;
        size_t bLen                      = (rhs.size() / 4) * 4;

        __m256i bmask_prev               = _mm256_set_epi64x(0,0,0,0);
        __m256i cmp_prev                 = _mm256_set_epi64x(0,0,0,0);

        int mask;

        while (i < aLen && j < bLen) {
            // load segments of four 64-bit elements
            __m256i va                   = _mm256_load_si256((__m256i*)&a[i]);
            __m256i vb                   = _mm256_load_si256((__m256i*)&b[j]);

            size_t i_prev                = i; // track i
            size_t j_prev                = j; // track j

            // move pointers
            uint64_t a_max               = _mm256_extract_epi64(va, 3);
            uint64_t b_max               = _mm256_extract_epi64(vb, 3);
            i                           += (a_max <= b_max) * 4;
            j                           += (a_max >= b_max) * 4;

            // part 1
            __m256i bmask                = _mm256_set_epi64x(0,0,0,0);
            bmask                        = _mm256_or_si256(bmask, bmask_prev);

            __m256i cmp_mask1            = _mm256_cmpeq_epi64(va, vb);
            cmp_mask1                    = _mm256_andnot_si256(bmask, cmp_mask1);
            bmask                        = _mm256_or_si256(cmp_mask1, bmask);

            // part 2
            constexpr auto cyclic_shift  = _MM_SHUFFLE(0,3,2,1);
            vb                           = _mm256_permute4x64_epi64(vb, cyclic_shift);
            bmask_prev                   = _mm256_permute4x64_epi64(bmask_prev, cyclic_shift);
            bmask                        = _mm256_permute4x64_epi64(bmask, cyclic_shift);
            bmask                        = _mm256_or_si256(bmask, bmask_prev);
            __m256i cmp_mask2            = _mm256_cmpeq_epi64(va, vb);
            cmp_mask2                    = _mm256_andnot_si256(bmask, cmp_mask2);
            bmask                        = _mm256_or_si256(cmp_mask2, bmask);

            // part 3
            vb                           = _mm256_permute4x64_epi64(vb, cyclic_shift);
            bmask_prev                   = _mm256_permute4x64_epi64(bmask_prev, cyclic_shift);
            bmask                        = _mm256_permute4x64_epi64(bmask, cyclic_shift);
            bmask                        = _mm256_or_si256(bmask, bmask_prev);
            __m256i cmp_mask3            = _mm256_cmpeq_epi64(va, vb);
            cmp_mask3                    = _mm256_andnot_si256(bmask, cmp_mask3);
            bmask                        = _mm256_or_si256(cmp_mask3, bmask);

            // part 4
            vb                           = _mm256_permute4x64_epi64(vb, cyclic_shift);
            bmask_prev                   = _mm256_permute4x64_epi64(bmask_prev, cyclic_shift);
            bmask                        = _mm256_permute4x64_epi64(bmask, cyclic_shift);
            bmask                        = _mm256_or_si256(bmask, bmask_prev);
            __m256i cmp_mask4            = _mm256_cmpeq_epi64(va, vb);
            cmp_mask4                    = _mm256_andnot_si256(bmask, cmp_mask4);
            bmask                        = _mm256_or_si256(cmp_mask4, bmask);

            // finish bmask cycle 
            bmask = _mm256_permute4x64_epi64(bmask, cyclic_shift);

            __m256i cmp_mask             = _mm256_or_si256(_mm256_or_si256(cmp_mask1, cmp_mask2), _mm256_or_si256(cmp_mask3, cmp_mask4)); // OR-ing of comparison masks
            cmp_mask                     = _mm256_andnot_si256(cmp_prev, cmp_mask);

            // convert the 256-bit mask to 4-bit
            mask                         = _mm256_movemask_pd((__m256d)cmp_mask);
            auto cnt                     = __builtin_popcount(mask); // a number of elements is a weight of the mask
            count                       += cnt;

            if (j_prev < j) {
                j_prev                   = j;
                bmask_prev               = _mm256_set_epi64x(0,0,0,0);
            } else {
                bmask_prev               = bmask;
            }
            if (i_prev < i) {
                i_prev                   = i;
                cmp_prev                 = _mm256_set_epi64x(0,0,0,0);
            } else {
                cmp_prev                 = cmp_mask;
            }
        }
        
        int bm = _mm256_movemask_pd((__m256d)bmask_prev); 
        j += __builtin_popcount(bm);

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
        detail::hash_vector_t hashes(nodes.size());
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
