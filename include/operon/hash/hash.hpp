// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_HASH_HPP
#define OPERON_HASH_HPP

#include "operon/core/constants.hpp"
#include "metrohash64.hpp"

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace Operon {

template <HashFunction F = HashFunction::XXHash>
struct Hasher {
    auto operator()(const uint8_t* key, size_t len) noexcept -> uint64_t
    {
        return XXH3_64bits(key, len);
    }
};

template<>
struct Hasher<HashFunction::MetroHash> {
    auto operator()(const uint8_t* key, size_t len) noexcept -> uint64_t
    {
        uint64_t h = 0;
        HashUtil::MetroHash64::Hash(key, len, reinterpret_cast<uint8_t*>(&h)); // NOLINT
        return h;
    }
};

template<>
struct Hasher<HashFunction::FNV1Hash> {
    auto operator()(const uint8_t* key, size_t len) noexcept -> uint64_t
    {
        uint64_t h = 14695981039346656037ull; // NOLINT

        for(size_t i = 0; i < len; ++i) {
            h ^= *(key+i);
            h *= 1099511628211ull; // NOLINT
        }
        return h;
    }
};
} // namespace Operon
#endif

