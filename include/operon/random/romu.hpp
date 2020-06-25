/* Romu generators - http://romu-random.org 
 *
 * This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef ROMU_HPP
#define ROMU_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace Operon {
namespace RandomGenerator {
    namespace detail {
        static inline constexpr uint64_t rotl(uint64_t x, unsigned k) noexcept
        {
            return (x << k) | (x >> (64U - k));
        }

        static inline constexpr uint64_t splitMix64(uint64_t& state) noexcept
        {
            uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15));
            z = (z ^ (z >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9);
            z = (z ^ (z >> 27U)) * UINT64_C(0x94d049bb133111eb);
            return z ^ (z >> 31U);
        }
    }

    class RomuTrio final {
    public:
        using result_type = uint64_t;

        static inline constexpr uint64_t(min)() { return result_type { 0 }; }
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); }

        RomuTrio(RomuTrio const&) = delete;
        RomuTrio& operator=(RomuTrio const&) = delete;

        explicit RomuTrio(uint64_t seed) noexcept
            : xState(detail::splitMix64(seed))
            , yState(detail::splitMix64(seed))
            , zState(detail::splitMix64(seed))
        {
            for (size_t i = 0; i < 10; ++i) {
                operator()();
            }
        }

        inline uint64_t operator()() noexcept
        {
            uint64_t xp = xState, yp = yState, zp = zState;
            xState = 15241094284759029579u * zp;
            yState = yp - xp;
            yState = detail::rotl(yState, 12);
            zState = zp - yp;
            zState = detail::rotl(zState, 44);
            return xp;
        }

    private:
        uint64_t xState;
        uint64_t yState;
        uint64_t zState;
    };

    class RomuDuo final {
    public:
        using result_type = uint64_t;

        static inline constexpr uint64_t(min)() { return result_type { 0 }; }
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); }

        RomuDuo(RomuDuo const&) = delete;
        RomuDuo& operator=(RomuDuo const&) = delete;

        explicit RomuDuo(uint64_t seed) noexcept
            : xState(detail::splitMix64(seed))
            , yState(detail::splitMix64(seed))
        {
            for (size_t i = 0; i < 10; ++i) {
                operator()();
            }
        }

        inline uint64_t operator()() noexcept
        {
            uint64_t xp = xState;
            xState = 15241094284759029579u * yState;
            yState = detail::rotl(yState, 36) + detail::rotl(yState, 15) - xp;
            return xp;
        }

    private:
        uint64_t xState;
        uint64_t yState;
    };

}
}

#endif
