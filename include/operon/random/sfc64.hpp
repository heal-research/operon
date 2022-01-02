/* Simple Fast Counting Random // NOLINT
 * // NOLINT
 * The original algorithm (C) Chris Doty-Humphrey was released into the public domain // NOLINT
 * http://pracrand.sourceforge.net/RNG_engines.txt // NOLINT
 * // NOLINT
 */ // NOLINT
 // NOLINT
#ifndef OPERON_RANDOM_SFC64_HPP // NOLINT
#define OPERON_RANDOM_SFC64_HPP // NOLINT
 // NOLINT
#include <cstdint> // NOLINT
#include <limits> // NOLINT
 // NOLINT
namespace Operon { // NOLINT
namespace Random { // NOLINT
    class Sfc64 final { // NOLINT
    public: // NOLINT
        using result_type = uint64_t; // NOLINT
 // NOLINT
        static inline constexpr uint64_t(min)() { return result_type { 0 }; } // NOLINT
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); } // NOLINT
 // NOLINT
        explicit Sfc64(uint64_t seed) noexcept // NOLINT
            : mA(seed) // NOLINT
            , mB(seed) // NOLINT
            , mC(seed) // NOLINT
            , mCounter(1) // NOLINT
        { // NOLINT
            for (size_t i = 0; i < 12; ++i) { // NOLINT
                operator()(); // NOLINT
            } // NOLINT
        } // NOLINT
 // NOLINT
        inline uint64_t operator()() noexcept // NOLINT
        { // NOLINT
            uint64_t tmp = mA + mB + mCounter++; // NOLINT
            mA = mB ^ (mB >> 11U); // NOLINT
            mB = mC + (mC << 3U); // NOLINT
            mC = rotl(mC, 24U) + tmp; // NOLINT
            return tmp; // NOLINT
        } // NOLINT
 // NOLINT
        // random double in range [0, 1( // NOLINT
        inline double uniform01() noexcept // NOLINT
        { // NOLINT
            union { // NOLINT
                uint64_t i; // NOLINT
                double d; // NOLINT
            } x {}; // NOLINT
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access) // NOLINT
            x.i = (UINT64_C(0x3ff) << 52U) | (operator()() >> 12U); // NOLINT
            // NOLINTNEXTLINE(cppcoreguidelines-pro-type-union-access) // NOLINT
            return x.d - 1.0; // NOLINT
        } // NOLINT
 // NOLINT
        // disallow copying (to prevent misuse) // NOLINT
        Sfc64(Sfc64 const&) = delete; // NOLINT
        Sfc64& operator=(Sfc64 const&) = delete; // NOLINT
 // NOLINT
        // allow moving // NOLINT
        Sfc64(Sfc64&&) noexcept = default; // NOLINT
        Sfc64& operator=(Sfc64&&) noexcept = default; // NOLINT
 // NOLINT
        ~Sfc64() noexcept = default; // NOLINT
 // NOLINT
    private: // NOLINT
        static constexpr uint64_t rotl(uint64_t x, unsigned k) noexcept // NOLINT
        { // NOLINT
            return (x << k) | (x >> (64U - k)); // NOLINT
        } // NOLINT
 // NOLINT
        uint64_t mA; // NOLINT
        uint64_t mB; // NOLINT
        uint64_t mC; // NOLINT
        uint64_t mCounter; // NOLINT
    }; // NOLINT
} // namespace Random // NOLINT
} // namespace Operon // NOLINT
 // NOLINT
#endif // NOLINT
 // NOLINT

