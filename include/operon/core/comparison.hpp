// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_COMPARISON_HPP
#define OPERON_COMPARISON_HPP

#include <cmath>
#include <iterator>
#include <type_traits>

#include "contracts.hpp"
#include "types.hpp"

namespace Operon {

enum class Dominance : int { Equal = 0,
    Left = 1,
    Right = 2,
    None = 3 };

struct Equal {
    template <std::floating_point T>
    inline auto operator()(T a, T b, T eps = 0.0) const noexcept -> bool
    {
        return std::abs(a - b) <= eps;
    }

    template<std::forward_iterator Input1, std::forward_iterator Input2>
    inline auto operator()(Input1 first1, Input1 last1, Input2 first2, Input2 last2, typename std::iterator_traits<Input1>::value_type eps = 0.0) const noexcept -> bool {
        return std::equal(first1, last1, first2, last2, [&](auto a, auto b) { return (*this)(a, b, eps); });
    }

    template<std::ranges::forward_range R1, std::ranges::forward_range R2>
    inline auto operator()(R1&& r1, R2&& r2, Operon::Scalar eps = 0.0) const noexcept -> bool
    {
        return (*this)(std::begin(r1), std::end(r1), std::begin(r2), std::end(r2), eps);
    }
};

template <bool CheckNan = false>
struct Less {
    template <std::floating_point T>
    inline auto operator()(T a, T b, T eps = 0.0) const noexcept -> bool
    {
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

    template<std::forward_iterator Input1, std::forward_iterator Input2>
    inline auto operator()(Input1 first1, Input1 last1, Input2 first2, Input2 last2, typename std::iterator_traits<Input1>::value_type eps = 0.0) const noexcept -> bool {
        return std::lexicographical_compare(first1, last1, first2, last2, [&](auto a, auto b) { return (*this)(a, b, eps); });
    }

    template<std::ranges::forward_range R1, std::ranges::forward_range R2>
    inline auto operator()(R1&& r1, R2&& r2, Operon::Scalar eps = 0.0) const noexcept -> bool
    {
        return (*this)(std::begin(r1), std::end(r1), std::begin(r2), std::end(r2), eps);
    }
};

template <bool CheckNan = false>
struct LessEqual {
    template <std::floating_point T>
    auto operator()(T a, T b, T eps = 0.0) const noexcept -> bool
    {
        return Less<CheckNan>{}(a, b, eps) || Equal{}(a, b, eps);
    }
};

template <bool CheckNan = false>
struct ParetoDominance {
    template<std::forward_iterator Input1, std::forward_iterator Input2>
    inline auto operator()(Input1 first1, Input1 last1, Input2 first2, Input2 last2) const noexcept -> Dominance
    {
        uint8_t r{0};
        uint8_t v{0};
        for (; first1 != last1 && first2 != last2; ++first1, ++first2) {
            auto const a = *first1;
            auto const b = *first2;
            r |= (a < b);
            v |= (a > b);
        }
        return static_cast<Dominance>(r | static_cast<uint8_t>(v << 1U));
    }

    template<std::forward_iterator Input1, std::forward_iterator Input2>
    inline auto operator()(Input1 first1, Input1 last1, Input2 first2, Input2 last2, typename std::iterator_traits<Input1>::value_type eps) const noexcept -> Dominance
    {
        Operon::Less<CheckNan> cmp;
        uint8_t r{0};
        uint8_t v{0};
        for (; first1 != last1 && first2 != last2; ++first1, ++first2) {
            auto const a = *first1;
            auto const b = *first2;
            r |= cmp(a, b, eps);
            v |= cmp(b, a, eps);
        }
        return static_cast<Dominance>(r | static_cast<uint8_t>(v << 1U));
    }

    template<std::ranges::forward_range R1, std::ranges::forward_range R2>
    inline auto operator()(R1&& r1, R2&& r2) const noexcept -> Dominance
    {
        return (*this)(std::begin(r1), std::end(r1), std::begin(r2), std::end(r2));
    }

    template<std::ranges::forward_range R1, std::ranges::forward_range R2>
    inline auto operator()(R1&& r1, R2&& r2, Operon::Scalar eps) const noexcept -> Dominance
    {
        return (*this)(std::begin(r1), std::end(r1), std::begin(r2), std::end(r2), eps);
    }
};

} // namespace Operon
#endif