// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research
//
#ifndef OPERON_RANDOM_HPP
#define OPERON_RANDOM_HPP

#include <algorithm>
#include <random>
#include <type_traits> // NOLINT
#include "operon/core/contracts.hpp" // NOLINT

#include "jsf.hpp" // NOLINT
#include "romu.hpp" // NOLINT
#include "sfc64.hpp" // NOLINT
#include "wyrand.hpp" // NOLINT

namespace Operon::Random { // NOLINT
template<typename R, typename T> // NOLINT
auto Uniform(R& random, T a, T b) -> T // NOLINT
{ // NOLINT
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type."); // NOLINT
    using Dist = std::conditional_t<std::is_integral_v<T>, std::uniform_int_distribution<T>, std::uniform_real_distribution<T>>; // NOLINT
    return Dist(a,b)(random); // NOLINT
} // NOLINT
 // NOLINT
template <typename R, typename InputIterator> // NOLINT
auto Sample(R& random, InputIterator start, InputIterator end) -> InputIterator // NOLINT
{ // NOLINT
    auto dist = std::distance(start, end); // NOLINT
    if (dist <= 1) { return start; } // NOLINT
    std::advance(start, Uniform(random, decltype(dist){0}, dist-1)); // NOLINT
    return start; // NOLINT
} // NOLINT
 // NOLINT
template <typename R, typename InputIterator> // NOLINT
auto Sample(R& random, InputIterator start, InputIterator end, // NOLINT
    std::add_pointer_t<bool(typename InputIterator::value_type const&)> condition) -> InputIterator // NOLINT
{ // NOLINT
    auto n = std::count_if(start, end, condition); // NOLINT
 // NOLINT
    if (n == 0) { // NOLINT
        return end; // no element satisfies the condition // NOLINT
    } // NOLINT
 // NOLINT
    auto m = Uniform(random, decltype(n){0}, n-1); // NOLINT
    InputIterator it; // NOLINT
    for (it = start; it < end; ++it) { // NOLINT
        if (condition(*it) && 0 == m--) { // NOLINT
            break; // NOLINT
        } // NOLINT
    } // NOLINT
 // NOLINT
    ENSURE(start <= it && it <= end); // NOLINT
    return it; // NOLINT
} // NOLINT
 // NOLINT
// sample n elements and write them to the output iterator // NOLINT
template <typename R, typename InputIterator, typename OutputIterator> // NOLINT
auto Sample(R& random, InputIterator start, InputIterator end, OutputIterator out, size_t n) -> OutputIterator // NOLINT
{ // NOLINT
    EXPECT(start < end); // NOLINT
    return std::sample(start, end, out, n, random); // NOLINT
} // NOLINT
} // namespace Operon::Random // NOLINT
 // NOLINT
#endif // NOLINT

