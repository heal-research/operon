// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_DISTANCE_HPP
#define OPERON_DISTANCE_HPP

#include "types.hpp"

#define VCL_NAMESPACE vcl
#include "vectorclass.h"

#include <immintrin.h>

#include <Eigen/Core>

namespace Operon {
namespace Distance {
    namespace {
        // returns true if lhs and rhs have _zero_ elements in common
        inline bool NullIntersectProbe(uint64_t const* lhs, uint64_t const* rhs) noexcept
        {
            // this can be done either with broadcasts or permutations
            // the permutations version seems faster 
            using vec = vcl::Vec4uq;
            auto a = vec().load(lhs);
            auto b0 = vec().load(rhs);
            auto b1 = vcl::permute4<1, 2, 3, 0>(b0);
            auto b2 = vcl::permute4<2, 3, 0, 1>(b0);
            auto b3 = vcl::permute4<3, 0, 1, 2>(b0);
            return !vcl::horizontal_or(a == b0 | a == b1 | a == b2 | a == b3);
        }

        // returns true if lhs and rhs have _zero_ elements in common
        inline bool NullIntersectProbe(uint32_t const* lhs, uint32_t const* rhs) noexcept
        {
            using vec = vcl::Vec8ui;
            auto a = vec().load(lhs);
            auto b0 = vec().load(rhs);
            auto b1 = vcl::permute8<1, 2, 3, 4, 5, 6, 7, 0>(b0);
            auto b2 = vcl::permute8<2, 3, 4, 5, 6, 7, 0, 1>(b0);
            auto b3 = vcl::permute8<3, 4, 5, 6, 7, 0, 1, 2>(b0);
            auto b4 = vcl::permute8<4, 5, 6, 7, 0, 1, 2, 3>(b0);
            auto b5 = vcl::permute8<5, 6, 7, 0, 1, 2, 3, 4>(b0);
            auto b6 = vcl::permute8<6, 7, 0, 1, 2, 3, 4, 5>(b0);
            auto b7 = vcl::permute8<7, 0, 1, 2, 3, 4, 5, 6>(b0);
            return !vcl::horizontal_or(a == b0 | a == b1 | a == b2 | a == b3 | a == b4 | a == b5 | a == b6 | a == b7);
        }

        // this method only works when the hash vectors are sorted
        inline size_t CountIntersect(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept
        {
            size_t ls = lhs.size();
            size_t rs = rhs.size();

            constexpr auto s = sizeof(Operon::Hash);
            auto lt = ls & (-s);
            auto rt = rs & (-s);

            Operon::Hash const* p = lhs.data();
            Operon::Hash const* q = rhs.data();
            size_t count = 0;
            size_t i = 0;
            size_t j = 0;
            while (i < lt && j < rt && NullIntersectProbe(p + i, q + j)) {
                auto a = p[i + 3];
                auto b = q[j + 3];
                // we cannot have a == b because then NullIntersectProbe would return false
                if (a < b) i += 4;
                if (a > b) j += 4;
            }

            auto lm = lhs.back();
            auto rm = rhs.back();

            while (i < ls && j < rs) {
                auto a = lhs[i];
                auto b = rhs[j];

                count += a == b;
                i += a <= b;
                j += b <= a;

                if (a > rm || b > lm) {
                    break;
                }
            }
            return count;
        }
    } // namespace

    inline double Jaccard(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept
    {
        size_t c = CountIntersect(lhs, rhs);
        size_t n = lhs.size() + rhs.size();
        return static_cast<double>(n - 2 * c) / static_cast<double>(n);
    }

    inline double SorensenDice(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept
    {
        size_t n = lhs.size() + rhs.size();
        size_t c = CountIntersect(lhs, rhs);
        return 1.0 - 2.0 * static_cast<double>(c) / static_cast<double>(n);
    }

} // namespace Distance
} // namespace Operon

#endif
