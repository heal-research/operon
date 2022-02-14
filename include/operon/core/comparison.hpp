// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_COMPARISON_HPP
#define OPERON_COMPARISON_HPP

#include <cmath>
#include <iterator>
#include <type_traits>

namespace Operon {

enum class Dominance : int { Left = 0,
    Equal = 1,
    Right = 2,
    None = 3 };

template <bool CheckNan = false>
struct Less {
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    auto operator()(T a, T b, T eps = 0.0) const noexcept -> bool
    {
        static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
        if constexpr (CheckNan) {
            if (std::isnan(a)) {
                return false;
            }
            if (std::isnan(b)) {
                return true;
            }
        }
        return a < b && b - a > eps;
    }

    template<typename Input1, typename Input2>
    auto operator()(Input1 first1, Input1 last1, Input2 first2, Input2 last2, typename std::iterator_traits<Input1>::value_type eps = 0.0) const noexcept -> bool {
        return std::lexicographical_compare(first1, last1, first2, last2, [&](auto a, auto b) { return (*this)(a, b, eps); });
    }

    template<typename Cont1, typename Cont2>
    auto operator()(Cont1 const& c1, Cont2 const& c2, typename Cont1::value_type eps = 0.0) const noexcept -> bool 
    {
        return (*this)(c1.cbegin(), c1.cend(), c2.cbegin(), c2.cend(), eps);
    }
};

template <bool CheckNan = false>
struct LessEqual {
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    auto operator()(T a, T b, T eps = 0.0) const noexcept -> bool
    {
        static_assert(std::is_floating_point_v<T>, "T must be a floating-point type");
        if constexpr (CheckNan) {
            if (std::isnan(a)) {
                return false;
            }
            if (std::isnan(b)) {
                return true;
            }
        }
        return a <= b && b - a > eps;
    }
};

struct Equal {
    template <typename T, std::enable_if_t<std::is_floating_point_v<T>, bool> = true>
    auto operator()(T a, T b, T eps = 0.0) const noexcept -> bool
    {
        return std::abs(a - b) <= eps;
    }

    template<typename Input1, typename Input2>
    auto operator()(Input1 first1, Input1 last1, Input2 first2, Input2 last2, typename std::iterator_traits<Input1>::value_type eps = 0.0) const noexcept -> bool {
        return std::equal(first1, last1, first2, last2, [&](auto a, auto b) { return (*this)(a, b, eps); });
    }

    template<typename Cont1, typename Cont2>
    auto operator()(Cont1 const& c1, Cont2 const& c2, typename Cont1::value_type eps = 0.0) const noexcept -> bool 
    {
        return (*this)(c1.cbegin(), c1.cend(), c2.cbegin(), c2.cend(), eps);
    }
};

template <bool Strict = false, bool CheckNan = false>
struct ParetoDominance {
    template <typename Input1, typename Input2>
    auto operator()(Input1 first1, Input1 last1, Input2 first2, Input2 last2, typename std::iterator_traits<Input1>::value_type eps = 0.0)const noexcept -> Dominance
    {
        bool better { false };
        bool worse { false };
        std::conditional_t<Strict, Less<CheckNan>, LessEqual<CheckNan>> cmp;
        for (; first1 != last1 && first2  != last2; ++first1, ++first2) {
            better |= cmp(*first1, *first2, eps);
            worse |= cmp(*first2, *first1, eps);
        }
        if (better) {
            return worse ? Dominance::None : Dominance::Left;
        }
        if (worse) {
            return better ? Dominance::None : Dominance::Right;
        }
        return Dominance::Equal;
    }

    template<typename Cont1, typename Cont2>
    auto operator()(Cont1 const& c1, Cont2 const& c2, typename Cont1::value_type eps = 0.0) const noexcept -> Dominance
    {
        return (*this)(c1.cbegin(), c1.cend(), c2.cbegin(), c2.cend(), eps);
    }
};

} // namespace Operon

#endif
