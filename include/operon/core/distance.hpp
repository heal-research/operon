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
        template<typename T, std::enable_if_t<std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8), bool> = true>
        constexpr bool Intersect(T const* lhs, T const* rhs) noexcept
        {
            auto a = std::conditional_t<sizeof(T) == 4, vcl::Vec4ui, vcl::Vec4uq>().load(lhs);
            return vcl::horizontal_add(a == rhs[0] | a == rhs[1] | a == rhs[2] | a == rhs[3]);
        }

        // this method only works when the hash vectors are sorted
        template<typename T>
        size_t CountIntersect(Operon::Span<T> lhs, Operon::Span<T> rhs) noexcept
        {
            size_t ls = lhs.size();
            size_t rs = rhs.size();

            constexpr auto s = sizeof(T);
            auto lt = ls & (-s);
            auto rt = rs & (-s);

            T const* p = lhs.data();
            T const* q = rhs.data();
            size_t count = 0;
            size_t i = 0;
            size_t j = 0;
            while (i < lt && j < rt) {
                if(Intersect(p + i, q + j)) {
                    break;
                }
                auto a = p[i + 3];
                auto b = q[j + 3];
                // we cannot have a == b because of !Intersect 
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

        template<typename Container>
        size_t CountIntersect(Container const& lhs, Container const& rhs) noexcept
        {
            using T = typename Container::value_type;
            return CountIntersect(Operon::Span<T const>(lhs.data(), lhs.size()), Operon::Span<T const>(rhs.data(), rhs.size()));
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
