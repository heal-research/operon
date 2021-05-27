// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_RANDOM_WYRAND_HPP
#define OPERON_RANDOM_WYRAND_HPP

#include <cstddef>
#include <cstdint>
#include <limits>
#include <memory>

namespace Operon {
namespace Random {
    namespace detail {
        static inline uint64_t wyrand_stateless(uint64_t *s) {
            *s += UINT64_C(0xa0761d6478bd642f);
            __uint128_t t = (__uint128_t)*s * (*s ^ UINT64_C(0xe7037ed1a0b428db));
            return static_cast<uint64_t>((t >> 64) ^ t);
        }
    }

    class Wyrand final {
    public:
        using result_type = uint64_t;

        static inline constexpr uint64_t(min)() { return result_type { 0 }; }
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); }

        explicit Wyrand(uint64_t seed) noexcept
            : state({seed}) {}

        // forbid copy and copy-assignment
        Wyrand(Wyrand const&) = delete;
        Wyrand& operator=(Wyrand const&) = delete;

        Wyrand(Wyrand&&) noexcept = default;
        Wyrand& operator=(Wyrand&&) noexcept = default;

        ~Wyrand() noexcept = default;

        inline uint64_t operator()() noexcept {
            return detail::wyrand_stateless(&state.x);
        }

    private:
        struct state {
            uint64_t x;
        } state;
    };

} // namespace Random
} // namespace Operon

#endif

