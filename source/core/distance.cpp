// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/core/distance.hpp"

#include <eve/wide.hpp>
#include <eve/module/algo.hpp>

namespace Operon::Distance {
    namespace detail {
        template<typename T>
        auto Intersect(T const* lhs, T const* rhs) {
            eve::wide<T> const a(lhs);
            return [&]<auto... Idx>(std::index_sequence<Idx...>){
                return eve::any(((a == rhs[Idx]) || ...));
            }(std::make_index_sequence<eve::wide<T>::size()>{});
        }

        // this method only works when the hash vectors are sorted
        template<typename T>
        inline auto CountIntersect(Operon::Span<T const> lhs, Operon::Span<T const> rhs) noexcept -> size_t
        {
            size_t constexpr S = eve::wide<T>::size();
            T const *p0 = lhs.data();
            T const *pS = p0 + (lhs.size() & (-S));
            T const *pN = p0 + lhs.size();

            T const *q0 = rhs.data();
            T const *qS = q0 + (rhs.size() & (-S));
            T const *qN = q0 + rhs.size();

            T const *p = p0;
            T const *q = q0;

            while(p < pS && q < qS && !Intersect(p, q)) {
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

    auto Jaccard(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept -> double
    {
        size_t n = lhs.size() + rhs.size();
        size_t c = detail::CountIntersect(lhs, rhs);
        return static_cast<double>(n - 2 * c) / static_cast<double>(n);
    }

    auto SorensenDice(Operon::Vector<Operon::Hash> const& lhs, Operon::Vector<Operon::Hash> const& rhs) noexcept -> double
    {
        size_t n = lhs.size() + rhs.size();
        size_t c = detail::CountIntersect(lhs, rhs);
        return 1 - 2 * static_cast<double>(c) / static_cast<double>(n);
    }

} // namespace Operon::Distance
