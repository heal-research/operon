/* Simple Fast Counting Random
 *
 * The original algorithm (C) Chris Doty-Humphrey was released into the public domain
 * http://pracrand.sourceforge.net/RNG_engines.txt
 *
 * This implementation of Sfc64 was lifted from:
 * Nanobench: Microbenchmark framework for C++11/14/17/20 
 * https://github.com/martinus/nanobench
 *
 * Licensed under the MIT License <http://opensource.org/licenses/MIT>.
 * SPDX-License-Identifier: MIT
 * Copyright (c) 2019 Martin Ankerl <http://martin.ankerl.com>
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */

#ifndef SFC64_HPP
#define SFC64_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace Operon {
namespace RandomGenerator {
    class Sfc64 final {
    public:
        using result_type = uint64_t;

        static inline constexpr uint64_t(min)() { return result_type { 0 }; }
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); }

        Sfc64()
            : Sfc64(UINT64_C(0xd3b45fd780a1b6a3))
        {
        }

        explicit Sfc64(uint64_t seed) noexcept
            : mA(seed)
            , mB(seed)
            , mC(seed)
            , mCounter(1)
        {
            for (size_t i = 0; i < 12; ++i) {
                operator()();
            }
        }

        inline uint64_t operator()() noexcept
        {
            uint64_t tmp = mA + mB + mCounter++;
            mA = mB ^ (mB >> 11U);
            mB = mC + (mC << 3U);
            mC = rotl(mC, 24U) + tmp;
            return tmp;
        }
        // random double in range [0, 1(
        inline double uniform01() noexcept
        {
            union {
                uint64_t i;
                double d;
            } x {};
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
            x.i = (UINT64_C(0x3ff) << 52U) | (operator()() >> 12U);
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access)
            return x.d - 1.0;
        }

    private:
        static constexpr uint64_t rotl(uint64_t x, unsigned k) noexcept
        {
            return (x << k) | (x >> (64U - k));
        }

        uint64_t mA;
        uint64_t mB;
        uint64_t mC;
        uint64_t mCounter;
    };
} // namespace Random
} // namespace Operon

#endif

