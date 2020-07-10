// metrohash64.cpp
//
// Copyright 2015-2018 J. Andrew Rogers
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "hash/metrohash64.hpp"

#include <cstdint>
#include <cstring>

namespace {
// rotate right idiom recognized by most compilers
inline static uint64_t rotate_right(uint64_t v, unsigned k)
{
    return (v >> k) | (v << (64 - k));
}

// unaligned reads, fast and safe on Nehalem and later microarchitectures
inline static uint64_t read_u64(const void* const ptr)
{
    return static_cast<uint64_t>(*reinterpret_cast<const uint64_t*>(ptr));
}

inline static uint64_t read_u32(const void* const ptr)
{
    return static_cast<uint64_t>(*reinterpret_cast<const uint32_t*>(ptr));
}

inline static uint64_t read_u16(const void* const ptr)
{
    return static_cast<uint64_t>(*reinterpret_cast<const uint16_t*>(ptr));
}

inline static uint64_t read_u8(const void* const ptr)
{
    return static_cast<uint64_t>(*reinterpret_cast<const uint8_t*>(ptr));
}

}

namespace Operon::HashUtil {

MetroHash64::MetroHash64(const uint64_t seed)
{
    Initialize(seed);
}

void MetroHash64::Initialize(const uint64_t seed)
{
    vseed = (static_cast<uint64_t>(seed) + k2) * k0;

    // initialize internal hash registers
    state.v[0] = vseed;
    state.v[1] = vseed;
    state.v[2] = vseed;
    state.v[3] = vseed;

    // initialize total length of input
    bytes = 0;
}

void MetroHash64::Update(const uint8_t* const buffer, const uint64_t length)
{
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buffer);
    const uint8_t* const end = ptr + length;

    // input buffer may be partially filled
    if (bytes % 32) {
        uint64_t fill = 32 - (bytes % 32);
        if (fill > length)
            fill = length;

        memcpy(input.b + (bytes % 32), ptr, static_cast<size_t>(fill));
        ptr += fill;
        bytes += fill;

        // input buffer is still partially filled
        if ((bytes % 32) != 0)
            return;

        // process full input buffer
        state.v[0] += read_u64(&input.b[0]) * k0;
        state.v[0] = rotate_right(state.v[0], 29) + state.v[2];
        state.v[1] += read_u64(&input.b[8]) * k1;
        state.v[1] = rotate_right(state.v[1], 29) + state.v[3];
        state.v[2] += read_u64(&input.b[16]) * k2;
        state.v[2] = rotate_right(state.v[2], 29) + state.v[0];
        state.v[3] += read_u64(&input.b[24]) * k3;
        state.v[3] = rotate_right(state.v[3], 29) + state.v[1];
    }

    // bulk update
    bytes += static_cast<uint64_t>(end - ptr);
    while (ptr <= (end - 32)) {
        // process directly from the source, bypassing the input buffer
        state.v[0] += read_u64(ptr) * k0;
        ptr += 8;
        state.v[0] = rotate_right(state.v[0], 29) + state.v[2];
        state.v[1] += read_u64(ptr) * k1;
        ptr += 8;
        state.v[1] = rotate_right(state.v[1], 29) + state.v[3];
        state.v[2] += read_u64(ptr) * k2;
        ptr += 8;
        state.v[2] = rotate_right(state.v[2], 29) + state.v[0];
        state.v[3] += read_u64(ptr) * k3;
        ptr += 8;
        state.v[3] = rotate_right(state.v[3], 29) + state.v[1];
    }

    // store remaining bytes in input buffer
    if (ptr < end)
        memcpy(input.b, ptr, static_cast<size_t>(end - ptr));
}

void MetroHash64::Finalize(uint8_t* const hash)
{
    // finalize bulk loop, if used
    if (bytes >= 32) {
        state.v[2] ^= rotate_right(((state.v[0] + state.v[3]) * k0) + state.v[1], 37) * k1;
        state.v[3] ^= rotate_right(((state.v[1] + state.v[2]) * k1) + state.v[0], 37) * k0;
        state.v[0] ^= rotate_right(((state.v[0] + state.v[2]) * k0) + state.v[3], 37) * k1;
        state.v[1] ^= rotate_right(((state.v[1] + state.v[3]) * k1) + state.v[2], 37) * k0;

        state.v[0] = vseed + (state.v[0] ^ state.v[1]);
    }

    // process any bytes remaining in the input buffer
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(input.b);
    const uint8_t* const end = ptr + (bytes % 32);

    if ((end - ptr) >= 16) {
        state.v[1] = state.v[0] + (read_u64(ptr) * k2);
        ptr += 8;
        state.v[1] = rotate_right(state.v[1], 29) * k3;
        state.v[2] = state.v[0] + (read_u64(ptr) * k2);
        ptr += 8;
        state.v[2] = rotate_right(state.v[2], 29) * k3;
        state.v[1] ^= rotate_right(state.v[1] * k0, 21) + state.v[2];
        state.v[2] ^= rotate_right(state.v[2] * k3, 21) + state.v[1];
        state.v[0] += state.v[2];
    }

    if ((end - ptr) >= 8) {
        state.v[0] += read_u64(ptr) * k3;
        ptr += 8;
        state.v[0] ^= rotate_right(state.v[0], 55) * k1;
    }

    if ((end - ptr) >= 4) {
        state.v[0] += read_u32(ptr) * k3;
        ptr += 4;
        state.v[0] ^= rotate_right(state.v[0], 26) * k1;
    }

    if ((end - ptr) >= 2) {
        state.v[0] += read_u16(ptr) * k3;
        ptr += 2;
        state.v[0] ^= rotate_right(state.v[0], 48) * k1;
    }

    if ((end - ptr) >= 1) {
        state.v[0] += read_u8(ptr) * k3;
        state.v[0] ^= rotate_right(state.v[0], 37) * k1;
    }

    state.v[0] ^= rotate_right(state.v[0], 28);
    state.v[0] *= k0;
    state.v[0] ^= rotate_right(state.v[0], 29);

    bytes = 0;

    // do any endian conversion here

    memcpy(hash, state.v, 8);
}

void MetroHash64::Hash(const uint8_t* buffer, const uint64_t length, uint8_t* const hash, const uint64_t seed)
{
    const uint8_t* ptr = reinterpret_cast<const uint8_t*>(buffer);
    const uint8_t* const end = ptr + length;

    uint64_t h = (static_cast<uint64_t>(seed) + k2) * k0;

    if (length >= 32) {
        uint64_t v[4];
        v[0] = h;
        v[1] = h;
        v[2] = h;
        v[3] = h;

        do {
            v[0] += read_u64(ptr) * k0;
            ptr += 8;
            v[0] = rotate_right(v[0], 29) + v[2];
            v[1] += read_u64(ptr) * k1;
            ptr += 8;
            v[1] = rotate_right(v[1], 29) + v[3];
            v[2] += read_u64(ptr) * k2;
            ptr += 8;
            v[2] = rotate_right(v[2], 29) + v[0];
            v[3] += read_u64(ptr) * k3;
            ptr += 8;
            v[3] = rotate_right(v[3], 29) + v[1];
        } while (ptr <= (end - 32));

        v[2] ^= rotate_right(((v[0] + v[3]) * k0) + v[1], 37) * k1;
        v[3] ^= rotate_right(((v[1] + v[2]) * k1) + v[0], 37) * k0;
        v[0] ^= rotate_right(((v[0] + v[2]) * k0) + v[3], 37) * k1;
        v[1] ^= rotate_right(((v[1] + v[3]) * k1) + v[2], 37) * k0;
        h += v[0] ^ v[1];
    }

    if ((end - ptr) >= 16) {
        uint64_t v0 = h + (read_u64(ptr) * k2);
        ptr += 8;
        v0 = rotate_right(v0, 29) * k3;
        uint64_t v1 = h + (read_u64(ptr) * k2);
        ptr += 8;
        v1 = rotate_right(v1, 29) * k3;
        v0 ^= rotate_right(v0 * k0, 21) + v1;
        v1 ^= rotate_right(v1 * k3, 21) + v0;
        h += v1;
    }

    if ((end - ptr) >= 8) {
        h += read_u64(ptr) * k3;
        ptr += 8;
        h ^= rotate_right(h, 55) * k1;
    }

    if ((end - ptr) >= 4) {
        h += read_u32(ptr) * k3;
        ptr += 4;
        h ^= rotate_right(h, 26) * k1;
    }

    if ((end - ptr) >= 2) {
        h += read_u16(ptr) * k3;
        ptr += 2;
        h ^= rotate_right(h, 48) * k1;
    }

    if ((end - ptr) >= 1) {
        h += read_u8(ptr) * k3;
        h ^= rotate_right(h, 37) * k1;
    }

    h ^= rotate_right(h, 28);
    h *= k0;
    h ^= rotate_right(h, 29);

    memcpy(hash, &h, 8);
}
} // namespace Operon::HashUtil

