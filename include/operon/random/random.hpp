#ifndef OPERON_RANDOM_HPP
#define OPERON_RANDOM_HPP

#include "core/contracts.hpp"
#include "random/jsf.hpp"
#include "random/romu.hpp"
#include "random/sfc64.hpp"

#include <algorithm>
#include <random>
#include <type_traits>

#include <fmt/core.h>

namespace Operon::Random {
template<typename R, typename T>
T Uniform(R& random, T a, T b) 
{
    static_assert(std::is_arithmetic_v<T>, "T must be an arithmetic type.");
    if constexpr(std::is_integral_v<T>) {
        return std::uniform_int_distribution<T>(a,b)(random);
    } else if constexpr(std::is_floating_point_v<T>) {
        return std::uniform_real_distribution<T>(a,b)(random);
    }
}

template <typename R, typename InputIterator>
InputIterator Sample(R& random, InputIterator start, InputIterator end)
{
    auto dist = std::distance(start, end);
    if (dist <= 1) return start;
    std::advance(start, Uniform(random, decltype(dist){0}, dist-1));
    return start;
}

template <typename R, typename InputIterator>
InputIterator Sample(R& random, InputIterator start, InputIterator end,
    std::add_pointer_t<bool(typename InputIterator::value_type const&)> condition)
{
    auto n = std::count_if(start, end, condition);

    if (n == 0)
        return end; // no element satisfies the condition

    auto m = Uniform(random, decltype(n){0}, n-1);
    InputIterator it;
    for (it = start; it < end; ++it) {
        if (condition(*it) && 0 == m--)
            break;
    }

    ENSURE(start <= it && it <= end);
    return it;
}

template <typename R, typename InputIterator, typename OutputIterator>
OutputIterator Sample(R& random, InputIterator start, InputIterator end, OutputIterator out, size_t n)
{
    EXPECT(start < end);
    return std::sample(start, end, out, n, random);
}
}

#endif
