// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_HASH_SHA256_HPP
#define OPERON_HASH_SHA256_HPP

#include <array>
#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <string>
#include <string_view>

#include "operon/operon_export.hpp"

namespace Operon {

// SHA-256 (FIPS 180-4). A small self-contained implementation (source/hash/sha256.cpp, derived from the
// public-domain crypto-algorithms sha256.c by Brad Conte) used for content identity -- dataset, model and
// generation-segment hashes in machine-readable run records -- not for anything security sensitive. It is
// unrelated to the non-cryptographic xxHash/Metrohash used for tree hashing.
//
// Streaming use:  Sha256 h; h.Update(a); h.Update(b); auto digest = h.Finalize();
// One-shot use:   Sha256Hex(bytes), Sha256FileHex(path).
class OPERON_EXPORT Sha256 {
public:
    static constexpr std::size_t DigestSize = 32;
    static constexpr std::size_t BlockSize = 64;
    using Digest = std::array<std::uint8_t, DigestSize>;

    Sha256() noexcept { Reset(); }

    // Starts a new message.
    auto Reset() noexcept -> void;

    // Appends `size` bytes; may be called any number of times with any split of the message.
    auto Update(void const* data, std::size_t size) noexcept -> Sha256&;
    auto Update(std::string_view bytes) noexcept -> Sha256& { return Update(bytes.data(), bytes.size()); }

    // Completes the message and returns its digest. The object must be Reset() before it is used again.
    [[nodiscard]] auto Finalize() noexcept -> Digest;

    // Lowercase hexadecimal rendering (64 characters) of a digest.
    [[nodiscard]] static auto ToHex(Digest const& digest) -> std::string;

private:
    auto Transform(std::uint8_t const* block) noexcept -> void;

    std::array<std::uint8_t, BlockSize> buffer_ {};
    std::array<std::uint32_t, 8> state_ {};
    std::uint64_t byteCount_ { 0 };
    std::size_t buffered_ { 0 };
};

// SHA-256 of `bytes` as 64 lowercase hexadecimal characters.
[[nodiscard]] OPERON_EXPORT auto Sha256Hex(std::string_view bytes) -> std::string;

// SHA-256 of the contents of the file at `path` (read in chunks, never held in memory as a whole).
// Throws std::runtime_error if the file cannot be opened or read.
[[nodiscard]] OPERON_EXPORT auto Sha256FileHex(std::filesystem::path const& path) -> std::string;

} // namespace Operon

#endif
