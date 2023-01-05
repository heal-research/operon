// SPDX-License-Identifier: MIT // NOLINT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research // NOLINT
 // NOLINT
#ifndef OPERON_RANDOM_WYRAND_HPP // NOLINT
#define OPERON_RANDOM_WYRAND_HPP // NOLINT
 // NOLINT
#include <cstdint> // NOLINT
#include <limits> // NOLINT
#include <memory> // NOLINT
 // NOLINT
namespace Operon { // NOLINT
namespace Random { // NOLINT
    namespace detail { // NOLINT
        static inline uint64_t wyrand_stateless(uint64_t *s) { // NOLINT
            *s += UINT64_C(0xa0761d6478bd642f); // NOLINT
            __uint128_t t = (__uint128_t)*s * (*s ^ UINT64_C(0xe7037ed1a0b428db)); // NOLINT
            return static_cast<uint64_t>((t >> 64) ^ t); // NOLINT
        } // NOLINT
    } // NOLINT
 // NOLINT
    class Wyrand final { // NOLINT
    public: // NOLINT
        using result_type = uint64_t; // NOLINT
 // NOLINT
        static inline constexpr uint64_t(min)() { return result_type { 0 }; } // NOLINT
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); } // NOLINT
 // NOLINT
        explicit Wyrand(uint64_t seed) noexcept // NOLINT
            : state({seed}) {} // NOLINT
 // NOLINT
        // forbid copy and copy-assignment // NOLINT
        Wyrand(Wyrand const&) = delete; // NOLINT
        Wyrand& operator=(Wyrand const&) = delete; // NOLINT
 // NOLINT
        Wyrand(Wyrand&&) noexcept = default; // NOLINT
        Wyrand& operator=(Wyrand&&) noexcept = default; // NOLINT
 // NOLINT
        ~Wyrand() noexcept = default; // NOLINT
 // NOLINT
        inline uint64_t operator()() noexcept { // NOLINT
            return detail::wyrand_stateless(&state.x); // NOLINT
        } // NOLINT
 // NOLINT
    private: // NOLINT
        struct state { // NOLINT
            uint64_t x; // NOLINT
        } state; // NOLINT
    }; // NOLINT
 // NOLINT
} // namespace Random // NOLINT
} // namespace Operon // NOLINT
 // NOLINT
#endif // NOLINT
 // NOLINT

