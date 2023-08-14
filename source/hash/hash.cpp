// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

#include "operon/hash/metrohash64.hpp"
#include <cstdint>

#define XXH_INLINE_ALL
#include <xxhash.h>

namespace Operon {

auto Hasher::operator()(uint8_t const* key, size_t len) const noexcept -> uint64_t
{
    if constexpr (Operon::HashFunc == HashFunction::XXHash) {
        return XXH64(key, len, 0);
    } else if constexpr(Operon::HashFunc == HashFunction::MetroHash) {
        uint64_t h = 0;
        HashUtil::MetroHash64::Hash(key, len, std::bit_cast<uint8_t*>(&h));
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

auto Hasher::operator()(std::string_view key) const noexcept -> uint64_t
{
    return (*this)(std::bit_cast<uint8_t const*>(key.data()), key.size());
}

auto Hasher::operator()(std::string const& key) const noexcept -> uint64_t
{
    return (*this)(std::string_view{key});
}

auto Hasher::operator()(char const* key) const noexcept -> uint64_t
{
    return (*this)(std::string_view{key});
}
} // namespace Operon
