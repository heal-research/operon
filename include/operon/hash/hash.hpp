#ifndef OPERON_HASH_HPP
#define OPERON_HASH_HPP

#include "core/constants.hpp"
#include "aquahash/aquahash.h"
#include "metrohash64.hpp"
#include "xxhash/xxhash.hpp"

#include "gsl/gsl"

namespace Operon {

template <HashFunction F = HashFunction::XXHash>
struct Hasher {
    uint64_t operator()(const uint8_t* key, size_t len) noexcept
    {
        return xxh::xxhash3<64>(key, len);
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
        return _mm_extract_epi64(h, 0);
    }
};

template<>
struct Hasher<HashFunction::FNV1> {
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

