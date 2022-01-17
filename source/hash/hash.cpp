// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

#define XXH_INLINE_ALL
#include <xxhash.h>

#include "operon/hash/metrohash64.hpp"

namespace Operon {

auto Hasher::operator()(uint8_t const* key, size_t len) noexcept -> uint64_t
{
    if constexpr (Operon::HashFunc == HashFunction::XXHash) {
        return XXH3_64bits(key, len);
    } else if constexpr(Operon::HashFunc == HashFunction::MetroHash) {
        uint64_t h = 0;
        HashUtil::MetroHash64::Hash(key, len, reinterpret_cast<uint8_t*>(&h)); // NOLINT
        return h;
    } else if constexpr(Operon::HashFunc == HashFunction::FNV1Hash) {
        uint64_t h = 14695981039346656037ULL; // NOLINT
        for (size_t i = 0; i < len; ++i) {
            h ^= *(key + i);
            h *= 1099511628211ULL; // NOLINT
        }
        return h;
    }
    return 0; // unreachable
}
} // namespace Operon
