// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef ROMU_HPP
#define ROMU_HPP

#include <cstddef>
#include <cstdint>
#include <limits>

namespace Operon {
namespace Random {
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

        explicit RomuTrio(uint64_t seed) noexcept
            : state{ detail::splitMix64(seed), detail::splitMix64(seed), detail::splitMix64(seed) }
        {
            for (size_t i = 0; i < 10; ++i) {
                operator()();
            }
        }

        // disallow copying (to prevent misuse)
        RomuTrio(RomuTrio const&) = delete;
        RomuTrio& operator=(RomuTrio const&) = delete;

        // allow moving
        RomuTrio(RomuTrio&&) noexcept = default;
        RomuTrio& operator=(RomuTrio&&) noexcept = default;

        ~RomuTrio() noexcept = default;

        inline uint64_t operator()() noexcept
        {
            uint64_t xp = state.x, yp = state.y, zp = state.z;
            state.x = 15241094284759029579u * zp;
            state.y = yp - xp;
            state.y = detail::rotl(state.y, 12);
            state.z = zp - yp;
            state.z = detail::rotl(state.z, 44);
            return xp;
        }


    private:
        struct state {
            uint64_t x;
            uint64_t y;
            uint64_t z;
        } state;
    };

    class RomuDuo final {
    public:
        using result_type = uint64_t;

        static inline constexpr uint64_t(min)() { return result_type { 0 }; }
        static inline constexpr uint64_t(max)() { return std::numeric_limits<result_type>::max(); }

        explicit RomuDuo(uint64_t seed) noexcept
            : state{ detail::splitMix64(seed), detail::splitMix64(seed) }
        {
            for (size_t i = 0; i < 10; ++i) {
                operator()();
            }
        }

        // disallow copying (to prevent misuse)
        RomuDuo(RomuDuo const&) = delete;
        RomuDuo& operator=(RomuDuo const&) = delete;

        // allow moving
        RomuDuo(RomuDuo&&) noexcept = default;
        RomuDuo& operator=(RomuDuo&&) noexcept = default;

        ~RomuDuo() noexcept = default;

        inline uint64_t operator()() noexcept
        {
            uint64_t xp = state.x;
            state.x = 15241094284759029579u * state.y;
            state.y = detail::rotl(state.y, 36) + detail::rotl(state.y, 15) - xp;
            return xp;
        }

    private:
        struct state {
            uint64_t x;
            uint64_t y;
        } state;
    };

}
}

#endif
