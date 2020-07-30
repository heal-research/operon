/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef OPERON_HASH_HPP
#define OPERON_HASH_HPP

#include "core/constants.hpp"
#include "aquahash.h"
#include "metrohash64.hpp"

#define XXH_INLINE_ALL
#include "xxhash/xxhash.h"

#include "gsl/gsl"

namespace Operon {

template <HashFunction F = HashFunction::XXHash>
struct Hasher {
    uint64_t operator()(const uint8_t* key, size_t len) noexcept
    {
        return XXH3_64bits(key, len);
    }
};

template<>
struct Hasher<HashFunction::MetroHash> {
    uint64_t operator()(const uint8_t* key, size_t len) noexcept
    {
        uint64_t h;
        HashUtil::MetroHash64::Hash(key, len, reinterpret_cast<uint8_t*>(&h));
        return h;
    }
};

template<>
struct Hasher<HashFunction::AquaHash> {
    uint64_t operator()(const uint8_t* key, size_t len) noexcept
    {
        __m128i h = AquaHash::SmallKeyAlgorithm(key, len);
        return _mm_extract_epi64(h, 0) ^ _mm_extract_epi64(h, 1);
    }
};

template<>
struct Hasher<HashFunction::FNV1Hash> {
    uint64_t operator()(const uint8_t* key, size_t len) noexcept
    {
        uint64_t h = 14695981039346656037ull;

        for(size_t i = 0; i < len; ++i) {
            h ^= *(key+i);
            h *= 1099511628211ull;
        }
        return h;
    }
};
} // namespace
#endif

