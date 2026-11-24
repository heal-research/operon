// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "operon/hash/sha256.hpp"

namespace Operon::Test {
namespace {
    // Digests of the FIPS 180-4 / NIST CAVP example messages (and the empty message).
    constexpr std::string_view EmptyDigest = "e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855";
    constexpr std::string_view AbcDigest = "ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad";
    constexpr std::string_view TwoBlockMessage = "abcdbcdecdefdefgefghfghighijhijkijkljklmklmnlmnomnopnopq";
    constexpr std::string_view TwoBlockDigest = "248d6a61d20638b8e5c026930c3e6039a33ce45964ff2167f6ecedd419db06c1";
    constexpr std::string_view MillionADigest = "cdc76e5c9914fb9281a1c7e284d73e67f1809a48a497200e046d39ccc7112cd0";
    constexpr std::string_view LongMessage = "abcdefghbcdefghicdefghijdefghijkefghijklfghijklmghijklmnhijklmnoijklmnopj"
                                             "klmnopqklmnopqrlmnopqrsmnopqrstnopqrstu";
    constexpr std::string_view LongDigest = "cf5b16a778af8380036ce59e7b0492370b249b11e8f07a51afac45037afee9d1";

    auto SystemSha256(std::filesystem::path const& path) -> std::string
    {
#if defined(_WIN32)
        return {};
#else
        auto const command = "sha256sum '" + path.string() + "' 2>/dev/null";
        std::unique_ptr<FILE, int (*)(FILE*)> pipe { popen(command.c_str(), "r"), pclose };
        if (!pipe) {
            return {};
        }
        std::array<char, 65> hex {};
        if (std::fread(hex.data(), 1, 64, pipe.get()) != 64) {
            return {};
        }
        return { hex.data(), 64 };
#endif
    }
} // namespace

TEST_CASE("SHA-256 known-answer vectors", "[sha256]")
{
    CHECK(Sha256Hex("") == EmptyDigest);
    CHECK(Sha256Hex("abc") == AbcDigest);
    CHECK(Sha256Hex(TwoBlockMessage) == TwoBlockDigest); // 56 bytes: the padding spills into a second block
    CHECK(Sha256Hex(LongMessage) == LongDigest); // 112 bytes, two full blocks plus padding
}

TEST_CASE("SHA-256 of one million 'a'", "[sha256]")
{
    std::string const message(1'000'000, 'a');
    CHECK(Sha256Hex(message) == MillionADigest);

    // The same message streamed through odd-sized chunks, one byte at a time at the start.
    Sha256 hasher;
    std::size_t offset = 0;
    for (std::size_t i = 0; i < 100; ++i) {
        hasher.Update(message.data() + offset, 1);
        ++offset;
    }
    constexpr std::array<std::size_t, 5> chunks { 7, 63, 64, 65, 4093 };
    for (std::size_t i = 0; offset < message.size(); ++i) {
        auto const n = std::min(chunks[i % chunks.size()], message.size() - offset);
        hasher.Update(message.data() + offset, n);
        offset += n;
    }
    CHECK(Sha256::ToHex(hasher.Finalize()) == MillionADigest);
}

TEST_CASE("SHA-256 padding boundaries", "[sha256]")
{
    // Messages of 55, 56, 63, 64 and 65 bytes straddle the single-block / two-block padding cases; compare the
    // one-shot result against streaming in every possible split and against known digests of the endpoints.
    constexpr std::array<std::pair<std::size_t, std::string_view>, 5> known {
        std::pair<std::size_t, std::string_view> {
            55, "9f4390f8d30c2dd92ec9f095b65e2b9ae9b0a925a5258e241c9f1e910f734318" },
        { 56, "b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a" },
        { 63, "7d3e74a05d7db15bce4ad9ec0658ea98e3f06eeecf16b4c6fff2da457ddc2f34" },
        { 64, "ffe054fe7ae0cb6dc65c3af9b61d5209f439851db43d0ba5997337df154668eb" },
        { 65, "635361c48bb9eab14198e76ea8ab7f1a41685d6ad62aa9146d301d4f17eb0ae0" },
    };
    for (auto const& [size, digest] : known) {
        INFO("size " << size);
        std::string const message(size, 'a');
        CHECK(Sha256Hex(message) == digest);
        for (std::size_t split = 0; split <= size; ++split) {
            Sha256 hasher;
            hasher.Update(message.data(), split);
            hasher.Update(message.data() + split, size - split);
            REQUIRE(Sha256::ToHex(hasher.Finalize()) == digest);
        }
    }
}

TEST_CASE("SHA-256 Reset starts a new message", "[sha256]")
{
    Sha256 hasher;
    hasher.Update("abc");
    static_cast<void>(hasher.Finalize());
    hasher.Reset();
    CHECK(Sha256::ToHex(hasher.Finalize()) == EmptyDigest);
}

TEST_CASE("SHA-256 of files matches sha256sum", "[sha256]")
{
    auto const dir = std::filesystem::temp_directory_path() / "operon-sha256-test";
    std::filesystem::create_directories(dir);

    SECTION("a small file with known content")
    {
        auto const path = dir / "abc.txt";
        {
            std::ofstream out(path, std::ios::binary);
            out << "abc";
        }
        CHECK(Sha256FileHex(path) == AbcDigest);
        std::filesystem::remove(path);
    }

    SECTION("a multi-chunk binary file with every byte value")
    {
        auto const path = dir / "random.bin";
        std::mt19937_64 rng { 42 };
        std::vector<char> bytes(3'000'001); // larger than the 64 KiB read chunk, not a multiple of the block size
        for (auto& b : bytes) {
            b = static_cast<char>(rng() & 0xFFU);
        }
        {
            std::ofstream out(path, std::ios::binary);
            out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));
        }
        auto const expected = SystemSha256(path);
        CHECK(Sha256FileHex(path) == Sha256Hex({ bytes.data(), bytes.size() }));
        if (expected.empty()) {
            WARN("sha256sum is not available: compared against the in-memory hash only");
        } else {
            CHECK(Sha256FileHex(path) == expected);
        }
        std::filesystem::remove(path);
    }

    SECTION("the files of the test data directory and the test binary itself")
    {
        auto const data = std::filesystem::path { OPERON_TEST_SOURCE_DIR } / "data";
        std::vector<std::filesystem::path> files;
        if (std::filesystem::is_directory(data)) {
            for (auto const& entry : std::filesystem::directory_iterator(data)) {
                if (entry.is_regular_file()) {
                    files.push_back(entry.path());
                }
            }
        }
#if defined(__linux__)
        files.emplace_back(std::filesystem::read_symlink("/proc/self/exe"));
#endif
        std::size_t compared = 0;
        for (auto const& file : files) {
            auto const expected = SystemSha256(file);
            if (expected.empty()) {
                continue;
            }
            INFO(file.string());
            CHECK(Sha256FileHex(file) == expected);
            ++compared;
        }
        if (compared == 0) {
            WARN("sha256sum is not available or there are no files to compare");
        }
    }

    SECTION("a missing file throws") { CHECK_THROWS_AS(Sha256FileHex(dir / "does-not-exist"), std::runtime_error); }
}

} // namespace Operon::Test
