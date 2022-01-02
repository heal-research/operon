// SPDX-License-Identifier: Apache-2.0
// SPDX-FileCopyrightText: Copyright 2015-2018 J. Andrew Rogers

#ifndef METROHASH_METROHASH_64_H
#define METROHASH_METROHASH_64_H

#include <cstdint>

namespace Operon::HashUtil {
// NOLINTBEGIN(*)
class MetroHash64
{
public:
    static const uint32_t bits = 64; // NOLINT

    // Constructor initializes the same as Initialize()
    explicit MetroHash64(uint64_t seed=0);
    
    // Initializes internal state for new hash with optional seed
    void Initialize(uint64_t seed=0);
    
    // Update the hash state with a string of bytes. If the length
    // is sufficiently long, the implementation switches to a bulk
    // hashing algorithm directly on the argument buffer for speed.
    void Update(const uint8_t* buffer, uint64_t length);
    
    // Constructs the final hash and writes it to the argument buffer.
    // After a hash is finalized, this instance must be Initialized()-ed
    // again or the behavior of Update() and Finalize() is undefined.
    void Finalize(uint8_t* hash);
    
    // A non-incremental function implementation. This can be significantly
    // faster than the incremental implementation for some usage patterns.
    static void Hash(const uint8_t * buffer, uint64_t length, uint8_t* hash, uint64_t seed=0);

private:
    static const uint64_t k0 = 0xD6D018F5;// NOLINT
    static const uint64_t k1 = 0xA2AA033B;// NOLINT
    static const uint64_t k2 = 0x62992FC1;// NOLINT
    static const uint64_t k3 = 0x30BC5B29;// NOLINT
    // NOLINT
    struct { uint64_t v[4]; } state;// NOLINT
    struct { uint8_t b[32]; } input;// NOLINT
    uint64_t bytes;// NOLINT
    uint64_t vseed;// NOLINT
};
// NOLINTEND(*)
} // namespace Operon::HashUtil
#endif // #ifndef METROHASH_METROHASH_64_H

