// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CONSTANTS_HPP
#define OPERON_CONSTANTS_HPP

#include <array>
#include <cmath>

namespace Operon {
    // hashing mode for tree nodes:
    // - Strict: hash both node label and coefficient (for leaf nodes)
    // - Relaxed: hash only the node label
    enum HashMode {
        Strict  = 0x1,
        Relaxed = 0x2
    };

    enum HashFunction { 
        XXHash,
        MetroHash,
        FNV1Hash,
    };

    // M_PI, M_PI_2, M_PI_4, M_1_PI, M_2_PI, M_2_SQRTPI, M_SQRT2, M_SQRT1_2, M_E, M_LOG2E, M_LOG10E, M_LN2, M_LN10
    struct Math {
        static double constexpr Pi       = 3.141592653589793115997963468544185161590576171875;
        static double constexpr Tau      = 2 * Pi;
        static double constexpr InvPi    = 0.318309886183790691216444201927515678107738494873046875;
        static double constexpr SqrtPi   = 1.772453850905515881919427556567825376987457275390625;
        static double constexpr E        = 2.718281828459045090795598298427648842334747314453125;
        static double constexpr Log2E    = 1.442695040888963387004650940070860087871551513671875;
        static double constexpr Log10E   = 0.43429448190325181666793241674895398318767547607421875;
        static double constexpr LogE2    = 0.69314718055994528622676398299518041312694549560546875;
        static double constexpr LogE10   = 2.30258509299404590109361379290930926799774169921875;
        static double constexpr Sqrt2    = 1.4142135623730951454746218587388284504413604736328125;
        static double constexpr InvSqrt2 = 0.707106781186547461715008466853760182857513427734375;

        static std::array<double, 11> constexpr Constants { Pi, Tau, InvPi, SqrtPi, E, Log2E, Log10E, LogE2, LogE10, Sqrt2, InvSqrt2 };
    };

    static constexpr auto Align64 = 64U;
} // namespace Operon

#endif
