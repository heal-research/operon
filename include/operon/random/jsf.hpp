// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_RANDOM_JSF_HPP
#define OPERON_RANDOM_JSF_HPP
 // NOLINT
#include <cstddef>
#include <cstdint>
#include <limits>

// implementation of Bob Jenkins' small prng https://burtleburtle.net/bob/rand/smallprng.html
// the name JSF (Jenkins Small Fast) was coined by Doty-Humphrey when he included it in PractRand
// a more detailed analysis at http://www.pcg-random.org/posts/bob-jenkins-small-prng-passes-practrand.html
namespace Operon {
namespace Random {
    namespace detail {
        using _rand32_underlying = uint32_t; // NOLINT
        using _rand64_underlying = uint64_t; // NOLINT

        template <size_t N> // NOLINT
        struct rand_type { // NOLINT
            using type = void; // NOLINT
        }; // NOLINT
        template <> // NOLINT
        struct rand_type<32> { // NOLINT
            using type = _rand32_underlying; // NOLINT
        }; // NOLINT
        template <> // NOLINT
        struct rand_type<64> { // NOLINT
            using type = _rand64_underlying; // NOLINT
        }; // NOLINT
    } // NOLINT
 // NOLINT
    // public // NOLINT
    template <size_t N> // NOLINT
    using rand_t = typename detail::rand_type<N>::type; // NOLINT
    using rand32_t = rand_t<32>; // NOLINT
    using rand64_t = rand_t<64>; // NOLINT
 // NOLINT
    // bitwise circular left shift // NOLINT
    template <size_t N> // NOLINT
    static inline rand_t<N> rotl(rand_t<N> x, rand_t<N> k) noexcept { return (x << k) | (x >> (N - k)); } // NOLINT
 // NOLINT
    template <size_t N> // NOLINT
    class JsfRand final { // NOLINT
    private: // NOLINT
        rand_t<N> a, b, c, d; // NOLINT
 // NOLINT
        // 2-rotate version for 32-bit with the amounts (27, 7) // NOLINT
        inline rand32_t prng32() noexcept // NOLINT
        { // NOLINT
            rand32_t e = a - rotl<N>(b, 27); // NOLINT
            a = b ^ rotl<N>(c, 17); // NOLINT
            b = c + d; // NOLINT
            c = d + e; // NOLINT
            d = e + a; // NOLINT
            return d; // NOLINT
        } // NOLINT
 // NOLINT
        // 3-rotate version for 64-bit with the amounts (7, 13, 37) yielding 18.4 bits of avalanche after 5 rounds // NOLINT
        inline rand64_t prng64() noexcept // NOLINT
        { // NOLINT
            rand64_t e = a - rotl<N>(b, 7); // NOLINT
            a = b ^ rotl<N>(c, 13); // NOLINT
            b = c + rotl<N>(d, 37); // NOLINT
            c = d + e; // NOLINT
            d = e + a; // NOLINT
            return d; // NOLINT
        } // NOLINT
 // NOLINT
    public: // NOLINT
        using result_type = rand_t<N>; // NOLINT
        static constexpr result_type min() { return result_type { 0 }; } // NOLINT
        static constexpr result_type max() { return std::numeric_limits<result_type>::max(); } // NOLINT
 // NOLINT
        explicit JsfRand(result_type seed = 0xdeadbeef) noexcept // NOLINT
            : a { 0xf1ea5eed } // NOLINT
            , b { seed } // NOLINT
            , c { seed } // NOLINT
            , d { seed } // NOLINT
        { // NOLINT
            static_assert(N == 32 || N == 64, "Invalid template parameter. Valid values are 32 and 64 for 32-bit and 64-bit output."); // NOLINT
            for (size_t i = 0; i < 20; ++i) { // NOLINT
                (*this)(); // NOLINT
            } // NOLINT
        } // NOLINT
 // NOLINT
        // disallow copying (to prevent misuse) // NOLINT
        JsfRand(JsfRand const&) = delete; // NOLINT
        JsfRand& operator=(JsfRand const&) = delete; // NOLINT
 // NOLINT
        // allow moving // NOLINT
        JsfRand(JsfRand&&) noexcept = default; // NOLINT
        JsfRand& operator=(JsfRand&&) noexcept = default; // NOLINT
 // NOLINT
        ~JsfRand() noexcept = default; // NOLINT
 // NOLINT
        inline rand_t<N> operator()() noexcept // NOLINT
        { // NOLINT
            if constexpr (N == 32) // NOLINT
                return prng32(); // NOLINT
            return prng64(); // NOLINT
        } // NOLINT
    }; // NOLINT
 // NOLINT
    using Jsf32 = JsfRand<32>; // NOLINT
    using Jsf64 = JsfRand<64>; // NOLINT
} // NOLINT
} // NOLINT
 // NOLINT
#endif // NOLINT

