// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_DISTANCE_HPP
#define OPERON_DISTANCE_HPP

#include "core/types.hpp"

#define VCL_NAMESPACE vcl
#include "vectorclass.h"

#include <immintrin.h>

#include <Eigen/Core>

namespace Operon {
namespace Distance {
    namespace {
        template<typename T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 8, bool> = true>
        constexpr inline auto check(T const* lhs, T const* rhs) noexcept
        {
            auto a = vcl::Vec4uq().load(lhs);
            return a == rhs[0] | a == rhs[1] | a == rhs[2] | a == rhs[3];
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T> && sizeof(T) == 4, bool> = true>
        constexpr inline auto check(T const* lhs, T const* rhs) noexcept
        {
            auto a = vcl::Vec8ui().load(lhs);
            return a == rhs[0] | a == rhs[1] | a == rhs[2] | a == rhs[3] | a == rhs[4] | a == rhs[5] | a == rhs[6] | a == rhs[7];
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T> && (sizeof(T) == 4 || sizeof(T) == 8), bool> = true>
        constexpr inline bool Intersect(T const* lhs, T const* rhs) noexcept
        {
            return vcl::horizontal_add(check(lhs, rhs));
        }

        // this method only works when the hash vectors are sorted
        template<typename T, size_t S = size_t{32} / sizeof(T)>
        inline size_t CountIntersect(Operon::Span<T> lhs, Operon::Span<T> rhs) noexcept
        {
            T const *p0 = lhs.data(), *pS = p0 + (lhs.size() & (-S)), *pN = p0 + lhs.size();
            T const *q0 = rhs.data(), *qS = q0 + (rhs.size() & (-S)), *qN = q0 + rhs.size();
            T const *p = p0, *q = q0;

            while(p < pS && q < qS) {
                if (Intersect(p, q)) {
                    break;
                } 
                auto const a = *(p + S - 1);
                auto const b = *(q + S - 1);
                if (a < b) p += S;
                if (a > b) q += S;
            }

            auto const aN = *(pN - 1);
            auto const bN = *(qN - 1);
            size_t count{0};
            while(p < pN && q < qN) {
                auto const a = *p;
                auto const b = *q;

                if (a > bN || b > aN) {
                    break;
                }

                count += a == b;
                p += a <= b;
                q += a >= b;
            }

            return count;
        }

        template<typename Container>
        inline size_t CountIntersect(Container const& lhs, Container const& rhs) noexcept
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
