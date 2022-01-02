// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_DISTANCE_HPP
#define OPERON_DISTANCE_HPP

#include <algorithm>
#include <vectorclass/vectorclass.h>
#include <Eigen/Core>

#include "types.hpp"

namespace Operon::Distance {
    namespace detail {
        template<typename T, size_t I = 0>
        auto Check(T const* lhs, T const* rhs) {
            using Vec = std::conditional_t<sizeof(T) == sizeof(uint32_t), Vec8ui, Vec4uq>;
            static_assert(I >= 0 && I < Vec::size());
            if constexpr(I == Vec::size() - 1) {
                return Vec().load(lhs) == rhs[I];
            } else {
                return (Vec().load(lhs) == rhs[I]) | Check<T, I+1>(lhs, rhs); 
            }
        }

        template<typename T, std::enable_if_t<std::is_integral_v<T> && (sizeof(T) == sizeof(uint32_t) || sizeof(T) == sizeof(uint64_t)), bool> = true>
        constexpr inline auto Intersect(T const* lhs, T const* rhs) noexcept -> bool
        {
            return horizontal_add(Check(lhs, rhs));
        }

        // this method only works when the hash vectors are sorted
        template<typename T, size_t S = 4 * sizeof(uint64_t) / sizeof(T)>
        inline auto CountIntersect(Operon::Span<T> lhs, Operon::Span<T> rhs) noexcept -> size_t
        {
            T const *p0 = lhs.data();
            T const *pS = p0 + (lhs.size() & (-S));
            T const *pN = p0 + lhs.size();

            T const *q0 = rhs.data();
            T const *qS = q0 + (rhs.size() & (-S));
            T const *qN = q0 + rhs.size();

            T const *p = p0;
            T const *q = q0;

            while(p < pS && q < qS) {
                if (Intersect(p, q)) {
                    break;
                } 
                auto const a = *(p + S - 1);
                auto const b = *(q + S - 1);
                if (a < b) { p += S; }
                if (a > b) { q += S; }
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
        inline auto CountIntersect(Container const& lhs, Container const& rhs) noexcept -> size_t
        {
            using T = typename Container::value_type;
            return CountIntersect(Operon::Span<T const>(lhs.data(), lhs.size()), Operon::Span<T const>(rhs.data(), rhs.size()));
        }
    } // namespace detail

    inline auto Jaccard(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept -> double
    {
        size_t c = detail::CountIntersect(lhs, rhs);
        size_t n = lhs.size() + rhs.size();
        return static_cast<double>(n - 2 * c) / static_cast<double>(n);
    }

    inline auto SorensenDice(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept -> double
    {
        size_t n = lhs.size() + rhs.size();
        size_t c = detail::CountIntersect(lhs, rhs);
        return 1 - 2 * static_cast<double>(c) / static_cast<double>(n);
    }

} // namespace Operon::Distance

#endif
