// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

// SHA-256 as specified in FIPS 180-4. The block transform and the streaming structure follow the public-domain
// "crypto-algorithms" sha256.c by Brad Conte (https://github.com/B-Con/crypto-algorithms; the author places the
// code in the public domain, "algorithms are not subject to copyright"), rewritten in C++ with a 64-bit message
// length and fixed-width big-endian loads/stores. Verified against the NIST/FIPS test vectors and against
// sha256sum (see test/source/implementation/sha256.cpp).

#include "operon/hash/sha256.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cstring>
#include <fstream>
#include <stdexcept>
#include <vector>

namespace Operon {

namespace {
    constexpr std::array<std::uint32_t, 64> K { 0x428a2f98U, 0x71374491U, 0xb5c0fbcfU, 0xe9b5dba5U, 0x3956c25bU,
        0x59f111f1U, 0x923f82a4U, 0xab1c5ed5U, 0xd807aa98U, 0x12835b01U, 0x243185beU, 0x550c7dc3U, 0x72be5d74U,
        0x80deb1feU, 0x9bdc06a7U, 0xc19bf174U, 0xe49b69c1U, 0xefbe4786U, 0x0fc19dc6U, 0x240ca1ccU, 0x2de92c6fU,
        0x4a7484aaU, 0x5cb0a9dcU, 0x76f988daU, 0x983e5152U, 0xa831c66dU, 0xb00327c8U, 0xbf597fc7U, 0xc6e00bf3U,
        0xd5a79147U, 0x06ca6351U, 0x14292967U, 0x27b70a85U, 0x2e1b2138U, 0x4d2c6dfcU, 0x53380d13U, 0x650a7354U,
        0x766a0abbU, 0x81c2c92eU, 0x92722c85U, 0xa2bfe8a1U, 0xa81a664bU, 0xc24b8b70U, 0xc76c51a3U, 0xd192e819U,
        0xd6990624U, 0xf40e3585U, 0x106aa070U, 0x19a4c116U, 0x1e376c08U, 0x2748774cU, 0x34b0bcb5U, 0x391c0cb3U,
        0x4ed8aa4aU, 0x5b9cca4fU, 0x682e6ff3U, 0x748f82eeU, 0x78a5636fU, 0x84c87814U, 0x8cc70208U, 0x90befffaU,
        0xa4506cebU, 0xbef9a3f7U, 0xc67178f2U };

    constexpr auto Ch(std::uint32_t x, std::uint32_t y, std::uint32_t z) -> std::uint32_t { return (x & y) ^ (~x & z); }
    constexpr auto Maj(std::uint32_t x, std::uint32_t y, std::uint32_t z) -> std::uint32_t
    {
        return (x & y) ^ (x & z) ^ (y & z);
    }
    constexpr auto BigSigma0(std::uint32_t x) -> std::uint32_t
    {
        return std::rotr(x, 2) ^ std::rotr(x, 13) ^ std::rotr(x, 22);
    }
    constexpr auto BigSigma1(std::uint32_t x) -> std::uint32_t
    {
        return std::rotr(x, 6) ^ std::rotr(x, 11) ^ std::rotr(x, 25);
    }
    constexpr auto SmallSigma0(std::uint32_t x) -> std::uint32_t
    {
        return std::rotr(x, 7) ^ std::rotr(x, 18) ^ (x >> 3U);
    }
    constexpr auto SmallSigma1(std::uint32_t x) -> std::uint32_t
    {
        return std::rotr(x, 17) ^ std::rotr(x, 19) ^ (x >> 10U);
    }

    constexpr auto LoadBigEndian32(std::uint8_t const* p) -> std::uint32_t
    {
        return (static_cast<std::uint32_t>(p[0]) << 24U) | (static_cast<std::uint32_t>(p[1]) << 16U)
            | (static_cast<std::uint32_t>(p[2]) << 8U) | static_cast<std::uint32_t>(p[3]);
    }

    constexpr auto StoreBigEndian32(std::uint32_t v, std::uint8_t* p) -> void
    {
        p[0] = static_cast<std::uint8_t>(v >> 24U);
        p[1] = static_cast<std::uint8_t>(v >> 16U);
        p[2] = static_cast<std::uint8_t>(v >> 8U);
        p[3] = static_cast<std::uint8_t>(v);
    }
} // namespace

auto Sha256::Reset() noexcept -> void
{
    state_ = { 0x6a09e667U, 0xbb67ae85U, 0x3c6ef372U, 0xa54ff53aU, 0x510e527fU, 0x9b05688cU, 0x1f83d9abU, 0x5be0cd19U };
    byteCount_ = 0;
    buffered_ = 0;
}

auto Sha256::Transform(std::uint8_t const* block) noexcept -> void
{
    std::array<std::uint32_t, 64> w;
    for (std::size_t i = 0; i < 16; ++i) {
        w[i] = LoadBigEndian32(block + (4 * i));
    }
    for (std::size_t i = 16; i < 64; ++i) {
        w[i] = SmallSigma1(w[i - 2]) + w[i - 7] + SmallSigma0(w[i - 15]) + w[i - 16];
    }

    auto a = state_[0];
    auto b = state_[1];
    auto c = state_[2];
    auto d = state_[3];
    auto e = state_[4];
    auto f = state_[5];
    auto g = state_[6];
    auto h = state_[7];

    for (std::size_t i = 0; i < 64; ++i) {
        auto const t1 = h + BigSigma1(e) + Ch(e, f, g) + K[i] + w[i];
        auto const t2 = BigSigma0(a) + Maj(a, b, c);
        h = g;
        g = f;
        f = e;
        e = d + t1;
        d = c;
        c = b;
        b = a;
        a = t1 + t2;
    }

    state_[0] += a;
    state_[1] += b;
    state_[2] += c;
    state_[3] += d;
    state_[4] += e;
    state_[5] += f;
    state_[6] += g;
    state_[7] += h;
}

auto Sha256::Update(void const* data, std::size_t size) noexcept -> Sha256&
{
    auto const* bytes = static_cast<std::uint8_t const*>(data);
    byteCount_ += size;

    if (buffered_ > 0) {
        auto const take = std::min(size, BlockSize - buffered_);
        std::memcpy(buffer_.data() + buffered_, bytes, take);
        buffered_ += take;
        bytes += take;
        size -= take;
        if (buffered_ < BlockSize) {
            return *this;
        }
        Transform(buffer_.data());
        buffered_ = 0;
    }
    while (size >= BlockSize) {
        Transform(bytes);
        bytes += BlockSize;
        size -= BlockSize;
    }
    if (size > 0) {
        std::memcpy(buffer_.data(), bytes, size);
        buffered_ = size;
    }
    return *this;
}

auto Sha256::Finalize() noexcept -> Digest
{
    auto const bitCount = byteCount_ * 8U;

    // Padding: a single 1 bit, zeros up to 56 mod 64 bytes, then the message length in bits as a big-endian u64.
    buffer_[buffered_++] = 0x80U;
    if (buffered_ > BlockSize - 8) {
        std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_), buffer_.end(), std::uint8_t { 0 });
        Transform(buffer_.data());
        buffered_ = 0;
    }
    std::fill(buffer_.begin() + static_cast<std::ptrdiff_t>(buffered_), buffer_.end() - 8, std::uint8_t { 0 });
    for (std::size_t i = 0; i < 8; ++i) {
        buffer_[BlockSize - 1 - i] = static_cast<std::uint8_t>(bitCount >> (8U * i));
    }
    Transform(buffer_.data());

    Digest digest;
    for (std::size_t i = 0; i < state_.size(); ++i) {
        StoreBigEndian32(state_[i], digest.data() + (4 * i));
    }
    return digest;
}

auto Sha256::ToHex(Digest const& digest) -> std::string
{
    static constexpr std::string_view hex = "0123456789abcdef";
    std::string out;
    out.reserve(2 * DigestSize);
    for (auto const byte : digest) {
        out.push_back(hex[byte >> 4U]);
        out.push_back(hex[byte & 0x0FU]);
    }
    return out;
}

auto Sha256Hex(std::string_view bytes) -> std::string
{
    Sha256 hasher;
    hasher.Update(bytes);
    return Sha256::ToHex(hasher.Finalize());
}

auto Sha256FileHex(std::filesystem::path const& path) -> std::string
{
    std::ifstream in(path, std::ios::binary);
    if (!in) {
        throw std::runtime_error("sha256: cannot open " + path.string());
    }
    Sha256 hasher;
    std::vector<char> chunk(std::size_t { 1 } << 16U);
    while (in) {
        in.read(chunk.data(), static_cast<std::streamsize>(chunk.size()));
        hasher.Update(chunk.data(), static_cast<std::size_t>(in.gcount()));
    }
    if (!in.eof()) {
        throw std::runtime_error("sha256: cannot read " + path.string());
    }
    return Sha256::ToHex(hasher.Finalize());
}

} // namespace Operon
