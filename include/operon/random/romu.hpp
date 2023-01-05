// SPDX-License-Identifier: MIT // NOLINT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research // NOLINT
 // NOLINT
#ifndef OPERON_RANDOM_ROMU_HPP // NOLINT
#define OPERON_RANDOM_ROMU_HPP // NOLINT
 // NOLINT
#include <cstddef> // NOLINT
#include <cstdint> // NOLINT
#include <limits> // NOLINT
 // NOLINT
namespace Operon { // NOLINT
namespace Random { // NOLINT
    namespace detail { // NOLINT
        static inline constexpr uint64_t rotl(uint64_t x, unsigned k) noexcept // NOLINT
        { // NOLINT
            return (x << k) | (x >> (64U - k)); // NOLINT
        } // NOLINT
 // NOLINT
        static inline constexpr uint64_t splitMix64(uint64_t& state) noexcept // NOLINT
        { // NOLINT
            uint64_t z = (state += UINT64_C(0x9e3779b97f4a7c15)); // NOLINT
            z = (z ^ (z >> 30U)) * UINT64_C(0xbf58476d1ce4e5b9); // NOLINT
            z = (z ^ (z >> 27U)) * UINT64_C(0x94d049bb133111eb); // NOLINT
            return z ^ (z >> 31U); // NOLINT
        } // NOLINT
    } // NOLINT
 // NOLINT
    class RomuTrio final { // NOLINT
    public: // NOLINT
        using result_type = uint64_t; // NOLINT
 // NOLINT
        static inline constexpr uint64_t(min)() { return result_type { 0 }; } // NOLINT
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); } // NOLINT
 // NOLINT
        explicit RomuTrio(uint64_t seed) noexcept // NOLINT
            : state{ detail::splitMix64(seed), detail::splitMix64(seed), detail::splitMix64(seed) } // NOLINT
        { // NOLINT
            for (size_t i = 0; i < 10; ++i) { // NOLINT
                operator()(); // NOLINT
            } // NOLINT
        } // NOLINT
 // NOLINT
        // disallow copying (to prevent misuse) // NOLINT
        RomuTrio(RomuTrio const&) = delete; // NOLINT
        RomuTrio& operator=(RomuTrio const&) = delete; // NOLINT
 // NOLINT
        // allow moving // NOLINT
        RomuTrio(RomuTrio&&) noexcept = default; // NOLINT
        RomuTrio& operator=(RomuTrio&&) noexcept = default; // NOLINT
 // NOLINT
        ~RomuTrio() noexcept = default; // NOLINT
 // NOLINT
        inline uint64_t operator()() noexcept // NOLINT
        { // NOLINT
            uint64_t xp = state.x, yp = state.y, zp = state.z; // NOLINT
            state.x = 15241094284759029579u * zp; // NOLINT
            state.y = yp - xp; // NOLINT
            state.y = detail::rotl(state.y, 12); // NOLINT
            state.z = zp - yp; // NOLINT
            state.z = detail::rotl(state.z, 44); // NOLINT
            return xp; // NOLINT
        } // NOLINT
 // NOLINT
 // NOLINT
    private: // NOLINT
        struct state { // NOLINT
            uint64_t x; // NOLINT
            uint64_t y; // NOLINT
            uint64_t z; // NOLINT
        } state; // NOLINT
    }; // NOLINT
 // NOLINT
    class RomuDuo final { // NOLINT
    public: // NOLINT
        using result_type = uint64_t; // NOLINT
 // NOLINT
        static inline constexpr uint64_t(min)() { return result_type { 0 }; } // NOLINT
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); } // NOLINT
 // NOLINT
        explicit RomuDuo(uint64_t seed) noexcept // NOLINT
            : state{ detail::splitMix64(seed), detail::splitMix64(seed) } // NOLINT
        { // NOLINT
            for (size_t i = 0; i < 10; ++i) { // NOLINT
                operator()(); // NOLINT
            } // NOLINT
        } // NOLINT
 // NOLINT
        // disallow copying (to prevent misuse) // NOLINT
        RomuDuo(RomuDuo const&) = delete; // NOLINT
        RomuDuo& operator=(RomuDuo const&) = delete; // NOLINT
 // NOLINT
        // allow moving // NOLINT
        RomuDuo(RomuDuo&&) noexcept = default; // NOLINT
        RomuDuo& operator=(RomuDuo&&) noexcept = default; // NOLINT
 // NOLINT
        ~RomuDuo() noexcept = default; // NOLINT
 // NOLINT
        inline uint64_t operator()() noexcept // NOLINT
        { // NOLINT
            uint64_t xp = state.x; // NOLINT
            state.x = 15241094284759029579u * state.y; // NOLINT
            state.y = detail::rotl(state.y, 36) + detail::rotl(state.y, 15) - xp; // NOLINT
            return xp; // NOLINT
        } // NOLINT
 // NOLINT
    private: // NOLINT
        struct state { // NOLINT
            uint64_t x; // NOLINT
            uint64_t y; // NOLINT
        } state; // NOLINT
    }; // NOLINT
 // NOLINT
} // namespace Random // NOLINT
} // namespace Operon // NOLINT
 // NOLINT
#endif // NOLINT

