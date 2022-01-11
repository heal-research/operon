// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_CONSTANTS_HPP
#define OPERON_CONSTANTS_HPP

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

    static constexpr auto Align64 = 64U;
} // namespace Operon

#endif
