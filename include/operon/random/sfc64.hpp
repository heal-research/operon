/* Simple Fast Counting Random
 *
 * The original algorithm (C) Chris Doty-Humphrey was released into the public domain
 * http://pracrand.sourceforge.net/RNG_engines.txt
 *
 */

#ifndef SFC64_HPP
#define SFC64_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace Operon {
namespace Random {
    class Sfc64 final {
    public:
        using result_type = uint64_t;

        static inline constexpr uint64_t(min)() { return result_type { 0 }; }
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); }

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

        // disallow copying (to prevent misuse)
        Sfc64(Sfc64 const&) = delete;
        Sfc64& operator=(Sfc64 const&) = delete;

        // allow moving
        Sfc64(Sfc64&&) noexcept = default;
        Sfc64& operator=(Sfc64&&) noexcept = default;

        ~Sfc64() noexcept = default;

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

