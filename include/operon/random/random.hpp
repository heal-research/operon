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
template <typename R, typename T>
T UniformInt(R& random, T lo, T hi)
{
    static_assert(std::is_invocable_r<uint64_t, R>::value);
    static_assert(std::is_integral_v<T>);
    EXPECT(lo < hi);
    // in our version, the upper limit is exclusive
    return std::uniform_int_distribution<T>(lo, hi - 1)(random);
}

template <typename R, typename T>
T UniformReal(R& random, T lo, T hi)
{
    static_assert(std::is_invocable_r<uint64_t, R>::value);
    static_assert(std::is_floating_point_v<T>);
    EXPECT(lo < hi);
    return std::uniform_real_distribution<T>(lo, hi)(random);
}

template <typename R, typename InputIterator>
InputIterator Sample(R& random, InputIterator start, InputIterator end)
{
    EXPECT(start < end);
    std::advance(start, UniformInt(random, 0l, std::distance(start, end)));
    return start;
}

template <typename R, typename InputIterator>
InputIterator Sample(R& random, InputIterator start, InputIterator end,
    std::add_pointer_t<bool(typename InputIterator::value_type const&)> condition)
{
    EXPECT(start < end);
    auto n = std::count_if(start, end, condition);

    if (n == 0)
        return end; // no element satisfies the condition

    auto m = UniformInt(random, 0l, n);
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
