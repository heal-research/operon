// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#if defined(_WIN32)
#include <io.h>
#else
#include <unistd.h>
#endif

#include <algorithm>
#include <array>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <span>
#include <stdexcept>
#include <string>
#include <string_view>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon::Test {
namespace {

    // Minimal variable hash used across all tree-building helpers.
    constexpr Operon::Hash VarHash = 1234567890ULL;

    auto MakeTree(Operon::RandomGenerator& rng) -> Operon::Tree
    {
        Operon::PrimitiveSet pset;
        pset.SetConfig(PrimitiveSet::Arithmetic);
        std::vector<Operon::Hash> vars { VarHash };
        Operon::BalancedTreeCreator creator { &pset, vars, /*bias=*/0.0, /*maxLength=*/20 };
        Operon::UniformTreeInitializer treeInit(&creator);
        treeInit.ParameterizeDistribution(5UL, 20UL);
        treeInit.SetMaxDepth(6);
        Operon::UniformCoefficientInitializer coeffInit;
        coeffInit.ParameterizeDistribution(Operon::Scalar { -5 }, Operon::Scalar { +5 });
        auto tree = treeInit(rng);
        coeffInit(rng, tree);
        return tree;
    }

    auto MakeIndividual(Operon::RandomGenerator& rng, std::size_t nObjectives = 2) -> Operon::Individual
    {
        Operon::Individual ind;
        ind.Genotype = MakeTree(rng);
        ind.Fitness.resize(nObjectives);
        std::uniform_real_distribution<Operon::Scalar> dist { -1, 1 };
        for (auto& f : ind.Fitness) {
            f = dist(rng);
        }
        ind.Rank = 0;
        ind.Distance = Operon::Scalar { 1 };
        return ind;
    }

    auto TreesEqual(Operon::Tree const& a, Operon::Tree const& b) -> bool
    {
        if (a.Length() != b.Length()) {
            return false;
        }
        for (std::size_t i = 0; i < a.Length(); ++i) {
            auto const& na = a[i];
            auto const& nb = b[i];
            if (na.Type != nb.Type) {
                return false;
            }
            if (na.HashValue != nb.HashValue) {
                return false;
            }
            if (na.IsEnabled != nb.IsEnabled) {
                return false;
            }
            if (na.Optimize != nb.Optimize) {
                return false;
            }
            if (na.RefTo != nb.RefTo) {
                return false;
            }
            if (std::abs(na.Value - nb.Value) > Operon::Scalar { 1e-6 }) {
                return false;
            }
        }
        return true;
    }

    // Recomputes the checksum in the envelope of a serialized checkpoint after its payload was edited, so that a test
    // of the structural validation (or of the BEVE decoder) is not stopped by the checksum.
    void Reseal(std::string& bytes)
    {
        REQUIRE(bytes.size() >= Operon::Serialization::CheckpointEnvelopeBytes);
        auto const payload = std::string_view { bytes }.substr(Operon::Serialization::CheckpointEnvelopeBytes);
        auto const sum = Operon::Hasher {}(reinterpret_cast<std::uint8_t const*>(payload.data()), payload.size());
        for (std::size_t i = 0; i < 8; ++i) {
            bytes[8 + i] = static_cast<char>((sum >> (8U * i)) & 0xFFU);
        }
    }

    auto IndividualsEqual(Operon::Individual const& a, Operon::Individual const& b) -> bool
    {
        if (!TreesEqual(a.Genotype, b.Genotype)) {
            return false;
        }
        if (a.Fitness.size() != b.Fitness.size()) {
            return false;
        }
        for (std::size_t i = 0; i < a.Fitness.size(); ++i) {
            if (std::abs(a.Fitness[i] - b.Fitness[i]) > Operon::Scalar { 1e-6 }) {
                return false;
            }
        }
        return a.Rank == b.Rank && std::abs(a.Distance - b.Distance) < Operon::Scalar { 1e-6 };
    }

} // namespace

TEST_CASE("Tree JSON round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    for (auto i = 0; i < 20; ++i) {
        auto original = MakeTree(rng);
        auto json = Operon::Serialization::ToJson(original);
        auto restored = Operon::Serialization::TreeFromJson(json);
        REQUIRE(restored);
        CHECK(restored->Validate());
        CHECK(TreesEqual(original, *restored));
    }
}

TEST_CASE("Individual JSON round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    for (auto i = 0; i < 20; ++i) {
        auto original = MakeIndividual(rng);
        auto json = Operon::Serialization::ToJson(original);
        auto restored = Operon::Serialization::IndividualFromJson(json);
        REQUIRE(restored);
        CHECK(restored->Genotype.Validate());
        CHECK(IndividualsEqual(original, *restored));
    }
}

TEST_CASE("Population span JSON produces array", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    Operon::Vector<Operon::Individual> front(10);
    for (auto& ind : front) {
        ind = MakeIndividual(rng);
    }

    auto json = Operon::Serialization::ToJson(std::span<Operon::Individual const>(front));
    CHECK(!json.empty());
    CHECK(json.front() == '[');

    for (auto const& original : front) {
        auto elemJson = Operon::Serialization::ToJson(original);
        auto restored = Operon::Serialization::IndividualFromJson(elemJson);
        REQUIRE(restored);
        CHECK(IndividualsEqual(original, *restored));
    }
}

TEST_CASE("Tree BEVE round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    for (auto i = 0; i < 20; ++i) {
        auto original = MakeTree(rng);
        auto beve = Operon::Serialization::ToBeve(original);
        auto restored = Operon::Serialization::TreeFromBeve(beve);
        REQUIRE(restored);
        CHECK(restored->Validate());
        CHECK(TreesEqual(original, *restored));
    }
}

TEST_CASE("Individual BEVE round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(42);
    for (auto i = 0; i < 20; ++i) {
        auto original = MakeIndividual(rng);
        auto beve = Operon::Serialization::ToBeve(original);
        auto restored = Operon::Serialization::IndividualFromBeve(beve);
        REQUIRE(restored);
        CHECK(restored->Genotype.Validate());
        CHECK(IndividualsEqual(original, *restored));
    }
}

TEST_CASE("Tree BEVE rejects wrong version", "[serialization]")
{
    Operon::RandomGenerator rng(3);
    auto beve = Operon::Serialization::ToBeve(MakeTree(rng));
    REQUIRE(!beve.empty());

    // TreeMagic = 0x4F505452 is emitted little-endian; the following u32 is
    // the format version. Corrupting it must reject the payload before any
    // tree nodes are materialized.
    constexpr std::array<char, 4> MagicLE { '\x52', '\x54', '\x50', '\x4F' };
    auto magicIt = std::search(beve.begin(), beve.end(), MagicLE.begin(), MagicLE.end());
    REQUIRE(magicIt != beve.end());
    constexpr std::array<char, 4> VersionLE { '\x01', '\x00', '\x00', '\x00' };
    auto versionIt = std::search(magicIt + 4, beve.end(), VersionLE.begin(), VersionLE.end());
    REQUIRE(versionIt != beve.end());
    *versionIt = '\x02';

    CHECK(!Operon::Serialization::TreeFromBeve(beve));
}

TEST_CASE("Tree BEVE rejects wrong magic", "[serialization]")
{
    Operon::RandomGenerator rng(3);
    auto beve = Operon::Serialization::ToBeve(MakeTree(rng));
    REQUIRE(!beve.empty());

    constexpr std::array<char, 4> MagicLE { '\x52', '\x54', '\x50', '\x4F' };
    auto magicIt = std::search(beve.begin(), beve.end(), MagicLE.begin(), MagicLE.end());
    REQUIRE(magicIt != beve.end());
    *magicIt ^= static_cast<char>(0xFF);

    CHECK(!Operon::Serialization::TreeFromBeve(beve));
}

TEST_CASE("Checkpoint BEVE round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(99);
    for (auto i = 0; i < 1000; ++i) {
        (void)rng();
    }

    Operon::Serialization::Checkpoint original;
    original.RngState = rng.state();
    original.Generation = 42;
    original.Population.resize(50);
    for (auto& ind : original.Population) {
        ind = MakeIndividual(rng);
    }
    original.WorkerRngStates.resize(8);
    for (auto& s : original.WorkerRngStates) {
        for (auto i = 0; i < 100; ++i) {
            (void)rng();
        }
        s = rng.state();
    }

    auto beve = Operon::Serialization::ToBeve(original);
    auto restored = Operon::Serialization::CheckpointFromBeve(beve);
    REQUIRE(restored);

    CHECK(restored->RngState == original.RngState);
    CHECK(restored->Generation == original.Generation);
    REQUIRE(restored->Population.size() == original.Population.size());
    for (std::size_t i = 0; i < original.Population.size(); ++i) {
        CHECK(IndividualsEqual(original.Population[i], restored->Population[i]));
    }
    REQUIRE(restored->WorkerRngStates.size() == original.WorkerRngStates.size());
    for (std::size_t i = 0; i < original.WorkerRngStates.size(); ++i) {
        CHECK(restored->WorkerRngStates[i] == original.WorkerRngStates[i]);
    }
}

TEST_CASE("Checkpoint file save/load round-trip", "[serialization]")
{
    Operon::RandomGenerator rng(7);
    for (auto i = 0; i < 500; ++i) {
        (void)rng();
    }

    Operon::Serialization::Checkpoint original;
    original.RngState = rng.state();
    original.Generation = 10;
    original.Population.resize(20);
    for (auto& ind : original.Population) {
        ind = MakeIndividual(rng);
    }
    original.WorkerRngStates.resize(4);
    for (auto& s : original.WorkerRngStates) {
        for (auto i = 0; i < 100; ++i) {
            (void)rng();
        }
        s = rng.state();
    }

    auto const path = (std::filesystem::temp_directory_path() / "operon_test_checkpoint.beve").string();
    Operon::Serialization::SaveCheckpoint(original, path);
    auto restored = Operon::Serialization::LoadCheckpoint(path);
    REQUIRE(restored);

    CHECK(restored->RngState == original.RngState);
    CHECK(restored->Generation == original.Generation);
    REQUIRE(restored->Population.size() == original.Population.size());
    for (std::size_t i = 0; i < original.Population.size(); ++i) {
        CHECK(IndividualsEqual(original.Population[i], restored->Population[i]));
    }
    REQUIRE(restored->WorkerRngStates.size() == original.WorkerRngStates.size());
    for (std::size_t i = 0; i < original.WorkerRngStates.size(); ++i) {
        CHECK(restored->WorkerRngStates[i] == original.WorkerRngStates[i]);
    }

    std::error_code cleanupEc;
    std::filesystem::remove(path, cleanupEc);
}

TEST_CASE("Checkpoint BEVE rejects wrong magic", "[serialization]")
{
    Operon::RandomGenerator rng(3);
    Operon::Serialization::Checkpoint cp;
    cp.RngState = rng.state();
    cp.Generation = 0;
    // Population size ≥ 2 ensures the BEVE payload is large enough for std::search
    // to locate the 4-byte magic pattern without hitting end-of-buffer prematurely.
    cp.Population.resize(2);
    for (auto& ind : cp.Population) {
        ind = MakeIndividual(rng);
    }

    auto beve = Operon::Serialization::ToBeve(cp);
    REQUIRE(!beve.empty());

    // Locate the magic value (0x4F50434B, LE: 4B 43 50 4F) in the BEVE
    // payload and flip one byte so FromProxy rejects it.
    constexpr std::array<char, 4> MagicLE { '\x4B', '\x43', '\x50', '\x4F' };
    auto it = std::search(beve.begin(), beve.end(), MagicLE.begin(), MagicLE.end());
    REQUIRE(it != beve.end()); // sanity: magic must be present in valid checkpoint
    *it ^= static_cast<char>(0xFF);
    CHECK(!Operon::Serialization::CheckpointFromBeve(beve)); // damaged payload: the checksum no longer matches
    Reseal(beve);
    CHECK(!Operon::Serialization::CheckpointFromBeve(beve)); // and with a matching checksum the payload magic is wrong

    // the magic of the envelope
    auto envelope = Operon::Serialization::ToBeve(cp);
    envelope[0] ^= static_cast<char>(0xFF);
    CHECK(!Operon::Serialization::CheckpointFromBeve(envelope));
}

TEST_CASE("Checkpoint BEVE rejects wrong version", "[serialization]")
{
    Operon::RandomGenerator rng(3);
    Operon::Serialization::Checkpoint cp;
    cp.RngState = rng.state();
    cp.Generation = 0; // keep 0 so \x01\x00\x00\x00 uniquely identifies version
    cp.Population.resize(2);
    for (auto& ind : cp.Population) {
        ind = MakeIndividual(rng);
    }

    auto beve = Operon::Serialization::ToBeve(cp);
    REQUIRE(!beve.empty());

    // Magic is at some position; the current CheckpointVersion (LE) follows
    // it. Search starting after the magic bytes so we land on the version
    // value.
    constexpr std::array<char, 4> MagicLE { '\x4B', '\x43', '\x50', '\x4F' };
    auto magicIt = std::search(beve.begin(), beve.end(), MagicLE.begin(), MagicLE.end());
    REQUIRE(magicIt != beve.end());

    // Generation=0 avoids a false match; assumes BEVE writes fields in
    // declaration order with no other occurrence of the version bytes
    // between magic and version.
    constexpr auto CurrentVersion
        = std::to_array<char>({ '\x03', '\x00', '\x00', '\x00' }); // must track CheckpointVersion in serialization.cpp
    auto versionIt = std::search(magicIt + 4, beve.end(), CurrentVersion.begin(), CurrentVersion.end());
    REQUIRE(versionIt != beve.end());

    // a newer version
    auto newer = beve;
    newer[static_cast<std::size_t>(versionIt - beve.begin())] = static_cast<char>(CurrentVersion[0] + 1);
    CHECK(!Operon::Serialization::CheckpointFromBeve(newer));

    // the previous version, which had no transposition cache, is no longer read
    auto older = beve;
    older[static_cast<std::size_t>(versionIt - beve.begin())] = static_cast<char>(CurrentVersion[0] - 1);
    CHECK(!Operon::Serialization::CheckpointFromBeve(older));

    // the version of the envelope: reported as such, not as a corrupt file
    auto envelopeNewer = beve;
    envelopeNewer[4] = static_cast<char>(Operon::Serialization::CheckpointEnvelopeBytes); // any other version
    CHECK(!Operon::Serialization::CheckpointFromBeve(envelopeNewer));

    // with a matching checksum the payload version is still checked
    Reseal(newer);
    Reseal(older);
    CHECK(!Operon::Serialization::CheckpointFromBeve(newer));
    CHECK(!Operon::Serialization::CheckpointFromBeve(older));

    // the unmodified payload is accepted
    CHECK(Operon::Serialization::CheckpointFromBeve(beve));
}

namespace {
    auto MakeCacheCheckpoint(std::size_t entries) -> Operon::Serialization::CacheCheckpoint
    {
        Operon::Serialization::CacheCheckpoint cache;
        cache.TableFingerprint = 0x0123456789ABCDEFULL;
        cache.Clock = 17;
        cache.MaxAge = 5;
        cache.Visibility = 1;
        for (std::size_t i = 0; i < entries; ++i) {
            cache.Keys.push_back(1000 + 7 * i);
            cache.InsertGenerations.push_back(static_cast<std::uint32_t>(i % 18));
            auto const length = static_cast<std::uint32_t>(1 + i % 3);
            cache.Lengths.push_back(length);
            for (std::uint32_t j = 0; j < length; ++j) {
                cache.Values.push_back(static_cast<Operon::Scalar>(i) + static_cast<Operon::Scalar>(j) / 8);
            }
        }
        return cache;
    }
} // namespace

TEST_CASE("Checkpoint BEVE carries the transposition cache bit for bit", "[serialization]")
{
    Operon::RandomGenerator rng(11);
    Operon::Serialization::Checkpoint original;
    original.RngState = rng.state();
    original.Generation = 17;
    original.Population.resize(3);
    for (auto& ind : original.Population) {
        ind = MakeIndividual(rng);
    }
    original.Cache = MakeCacheCheckpoint(100);
    original.Cache->Values[3] = std::numeric_limits<Operon::Scalar>::infinity();
    original.Cache->Values[4] = -Operon::Scalar { 0 };

    auto restored = Operon::Serialization::CheckpointFromBeve(Operon::Serialization::ToBeve(original));
    REQUIRE(restored);
    REQUIRE(restored->Cache.has_value());
    auto const& a = *original.Cache;
    auto const& b = *restored->Cache;
    CHECK(b.TableFingerprint == a.TableFingerprint);
    CHECK(b.Clock == a.Clock);
    CHECK(b.MaxAge == a.MaxAge);
    CHECK(b.Visibility == a.Visibility);
    CHECK(b.Keys == a.Keys);
    CHECK(b.InsertGenerations == a.InsertGenerations);
    CHECK(b.Lengths == a.Lengths);
    REQUIRE(b.Values.size() == a.Values.size());
    CHECK(std::memcmp(b.Values.data(), a.Values.data(), a.Values.size() * sizeof(Operon::Scalar)) == 0);

    // a checkpoint of a run without a cache stays without one
    original.Cache.reset();
    restored = Operon::Serialization::CheckpointFromBeve(Operon::Serialization::ToBeve(original));
    REQUIRE(restored);
    CHECK_FALSE(restored->Cache.has_value());
}

namespace {
    // Decoding a damaged checkpoint prints the reason to stderr; the tests below decode thousands of damaged ones.
    class QuietStderr {
    public:
        QuietStderr()
        {
            std::fflush(stderr);
            saved_ = DupFd(2);
            if (auto* sink = std::fopen(NullDevice(), "w")) {
                Dup2Fd(FileNo(sink), 2);
                std::fclose(sink);
            }
        }
        ~QuietStderr()
        {
            std::fflush(stderr);
            if (saved_ >= 0) {
                Dup2Fd(saved_, 2);
                CloseFd(saved_);
            }
        }
        QuietStderr(QuietStderr const&) = delete;
        auto operator=(QuietStderr const&) -> QuietStderr& = delete;

    private:
#if defined(_WIN32)
        static auto NullDevice() -> char const* { return "NUL"; }
        static auto DupFd(int fd) -> int { return _dup(fd); }
        static auto Dup2Fd(int from, int to) -> void { _dup2(from, to); }
        static auto FileNo(std::FILE* f) -> int { return _fileno(f); }
        static auto CloseFd(int fd) -> void { _close(fd); }
#else
        static auto NullDevice() -> char const* { return "/dev/null"; }
        static auto DupFd(int fd) -> int { return ::dup(fd); }
        static auto Dup2Fd(int from, int to) -> void { ::dup2(from, to); }
        static auto FileNo(std::FILE* f) -> int { return ::fileno(f); }
        static auto CloseFd(int fd) -> void { ::close(fd); }
#endif
        int saved_ { -1 };
    };

    // A small but complete checkpoint: population, worker states, cache, run fingerprint.
    auto MakeSmallCheckpoint() -> Operon::Serialization::Checkpoint
    {
        Operon::RandomGenerator rng(21);
        Operon::Serialization::Checkpoint cp;
        cp.RngState = rng.state();
        cp.Generation = 3;
        cp.Population.resize(2);
        for (auto& ind : cp.Population) {
            ind = MakeIndividual(rng);
        }
        cp.WorkerRngStates.resize(2);
        for (auto& state : cp.WorkerRngStates) {
            (void)rng();
            state = rng.state();
        }
        cp.Cache = MakeCacheCheckpoint(4);
        cp.Run = Operon::Serialization::RunFingerprint { { "dataset.content", "option.objective" }, { 11, 22 } };
        return cp;
    }
} // namespace

TEST_CASE("Checkpoint BEVE rejects inconsistent cache arrays", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    auto const base = MakeSmallCheckpoint();
    auto const rejects = [&](auto&& mutate) {
        auto cp = base;
        mutate(*cp.Cache);
        return !Operon::Serialization::CheckpointFromBeve(Operon::Serialization::ToBeve(cp)).has_value();
    };
    CHECK_FALSE(rejects([](auto&) {})); // the unmodified checkpoint is accepted
    CHECK(rejects([](auto& c) { c.Values.pop_back(); })); // fewer values than the lengths add up to
    CHECK(rejects([](auto& c) { c.Values.push_back(1); })); // more
    CHECK(rejects([](auto& c) { c.Keys.pop_back(); }));
    CHECK(rejects([](auto& c) { c.InsertGenerations.pop_back(); }));
    CHECK(rejects([](auto& c) { c.Lengths[1] = 0; })); // a zero-length entry
    CHECK(rejects([](auto& c) { c.Lengths[1] = 0xFFFFFFFFU; })); // a length that is not there
    CHECK(rejects([](auto& c) { std::swap(c.Keys[0], c.Keys[1]); })); // keys not ascending
    CHECK(rejects([](auto& c) { c.Keys[1] = c.Keys[0]; })); // duplicate key
    CHECK(rejects([](auto& c) { c.Visibility = 7; }));

    auto cp = base;
    cp.Run->Hashes.pop_back(); // fingerprint arrays of different lengths
    CHECK_FALSE(Operon::Serialization::CheckpointFromBeve(Operon::Serialization::ToBeve(cp)).has_value());
}

TEST_CASE("ToZobristState checks its input instead of indexing out of bounds", "[serialization][checkpoint]")
{
    auto const base = MakeCacheCheckpoint(5);
    CHECK_NOTHROW(Operon::Serialization::ToZobristState(base));
    auto check = [&](auto&& mutate) {
        auto cache = base;
        mutate(cache);
        CHECK_THROWS_AS(Operon::Serialization::ToZobristState(cache), std::invalid_argument);
    };
    check([](auto& c) { c.Lengths.pop_back(); }); // Keys longer than Lengths: would index past the end
    check([](auto& c) { c.InsertGenerations.pop_back(); });
    check([](auto& c) { c.Values.pop_back(); });
    check([](auto& c) {
        c.Lengths[0] = 0xFFFFFFFFU;
        c.Lengths[1] = 0xFFFFFFFFU;
    }); // sum far beyond Values
    check([](auto& c) { c.Lengths[2] = 0; });
    check([](auto& c) { c.Keys[3] = c.Keys[2]; });
    check([](auto& c) { c.Visibility = 2; });
}

TEST_CASE("Checkpoint BEVE rejects trees that are not well-formed postfix expressions", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    auto cp = MakeSmallCheckpoint();
    auto const bad = [&](auto&& mutate) {
        auto copy = cp;
        auto nodes = copy.Population[0].Genotype.Nodes();
        mutate(nodes);
        copy.Population[0].Genotype = Operon::Tree { std::move(nodes) };
        return !Operon::Serialization::CheckpointFromBeve(Operon::Serialization::ToBeve(copy)).has_value();
    };
    CHECK_FALSE(bad([](auto&) {}));
    CHECK(bad([](auto& nodes) { nodes.back().Arity = 200; })); // more arguments than nodes
    CHECK(bad([](auto& nodes) { nodes.front().Arity = 1; })); // the first node has nothing to consume
    CHECK(bad([](auto& nodes) { nodes.push_back(nodes.front()); })); // two expressions
    CHECK(bad([](auto& nodes) {
        nodes.back().Type = Operon::NodeType::Ref;
        nodes.back().Arity = 0;
        nodes.back().RefTo = 9999;
    }));
}

// Truncation at every length, and any single byte changed to another value, is rejected (envelope checksum), never a
// crash, hang or huge allocation.
TEST_CASE("Checkpoint BEVE decoding rejects truncation and corruption of every byte", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    auto const bytes = Operon::Serialization::ToBeve(MakeSmallCheckpoint());
    REQUIRE(Operon::Serialization::CheckpointFromBeve(bytes).has_value());
    REQUIRE(bytes.size() < 20000);

    for (std::size_t length = 0; length < bytes.size(); ++length) {
        INFO("prefix length " << length);
        CHECK_FALSE(Operon::Serialization::CheckpointFromBeve(std::string_view { bytes.data(), length }).has_value());
    }

    std::size_t accepted = 0;
    for (std::size_t i = 0; i < bytes.size(); ++i) {
        for (auto const value : { '\x00', '\x01', '\x7F', '\xFF', '\xFE' }) {
            if (bytes[i] == value) {
                continue; // not a change
            }
            auto damaged = bytes;
            damaged[i] = value;
            accepted += Operon::Serialization::CheckpointFromBeve(damaged).has_value() ? 1 : 0;
        }
    }
    CHECK(accepted == 0); // the checksum notices every change of a byte
}

namespace {
    // What a tree of an accepted checkpoint must satisfy, written independently of the reader: empty string if it does.
    auto CheckpointTreeProblem(Operon::Tree const& tree) -> std::string
    {
        auto const& nodes = tree.Nodes();
        if (nodes.empty()) {
            return "empty genotype";
        }
        if (nodes.size() > 65535) {
            return "too many nodes";
        }
        for (std::size_t i = 0; i < nodes.size(); ++i) {
            auto const& n = nodes[i];
            auto const type = static_cast<unsigned>(n.Type);
            if (type > static_cast<unsigned>(Operon::NodeType::Function)) {
                return "node type out of range";
            }
            if (n.IsRef() && n.RefTo >= i) {
                return "Ref does not point backwards";
            }
            if (!n.IsFunction() && n.Arity != 0) {
                return "terminal with arguments";
            }
            if (n.IsFunction()) {
                if (n.Arity == 0) {
                    return "function without arguments";
                }
                if (n.HashValue < Operon::BuiltinOpCount) {
                    auto const op = static_cast<Operon::BuiltinOp>(n.HashValue);
                    if (op > Operon::BuiltinOp::Fmax && op <= Operon::BuiltinOp::Powabs && n.Arity != 2) {
                        return "binary function with another arity";
                    }
                    if (op > Operon::BuiltinOp::Powabs && n.Arity != 1) {
                        return "unary function with another arity";
                    }
                }
            }
        }
        if (auto const valid = tree.Validate(); !valid) {
            return "Tree::Validate failed";
        }
        return {};
    }

    auto CheckpointProblem(Operon::Serialization::Checkpoint const& cp) -> std::string
    {
        for (auto const& ind : cp.Population) {
            if (auto problem = CheckpointTreeProblem(ind.Genotype); !problem.empty()) {
                return problem;
            }
            if (ind.Fitness.size() > 64) {
                return "too many objectives";
            }
        }
        if (cp.Population.size() > (1U << 20U)
            || cp.WorkerRngStates.size() > Operon::Serialization::MaxCheckpointWorkerStates) {
            return "too many individuals or workers";
        }
        return {};
    }

    // A small checkpoint whose first individual has the given nodes (in the order given, nothing recomputed).
    auto EncodeWithNodes(std::vector<Operon::Node> nodes) -> std::string
    {
        auto cp = MakeSmallCheckpoint();
        cp.Population[0].Genotype = Operon::Tree { Operon::Vector<Operon::Node>(nodes.begin(), nodes.end()) };
        return Operon::Serialization::ToBeve(cp);
    }

    auto Const() -> Operon::Node { return Operon::Node::Constant(1.5); }

    auto Variable() -> Operon::Node { return Operon::Node(Operon::NodeType::Variable, VarHash); }

    auto Fn(Operon::BuiltinOp op, std::uint16_t arity) -> Operon::Node
    {
        return Operon::Node::Function(static_cast<Operon::Hash>(op), arity);
    }

    auto RefTo(std::uint16_t target) -> Operon::Node { return Operon::Node::Ref(target); }

    auto WithArity(Operon::Node node, std::uint16_t arity) -> Operon::Node
    {
        node.Arity = arity;
        return node;
    }

    auto Decodes(std::string_view bytes) -> bool
    {
        return Operon::Serialization::CheckpointFromBeve(bytes).has_value();
    }

    // The bytes of MakeSmallCheckpoint() with the number of individuals the file declares replaced (the count of the
    // two individuals is a one-byte compressed integer; it becomes a four-byte one) and `padding` zero bytes appended.
    auto DeclarePopulation(std::string bytes, std::uint64_t count, std::size_t padding = 0) -> std::string
    {
        auto const key = bytes.find("population");
        REQUIRE(key != std::string::npos);
        auto const countPos = key + std::string_view { "population" }.size() + 1; // after the array tag
        REQUIRE((static_cast<unsigned char>(bytes[countPos]) & 3U) == 0U);
        REQUIRE((static_cast<unsigned char>(bytes[countPos]) >> 2U) == 2U);
        REQUIRE(count < (std::uint64_t { 1 } << 30U));
        auto const encoded = static_cast<std::uint32_t>((count << 2U) | 2U);
        std::string field(4, '\0');
        for (std::size_t i = 0; i < 4; ++i) {
            field[i] = static_cast<char>((encoded >> (8U * i)) & 0xFFU);
        }
        bytes.replace(countPos, 1, field);
        bytes.resize(bytes.size() + padding, '\0');
        Reseal(bytes);
        return bytes;
    }

#if defined(__linux__)
    // Peak resident set size of this process in KiB since it was last reset, if /proc says so.
    auto PeakRssKiB() -> std::optional<std::uint64_t>
    {
        std::ifstream status("/proc/self/status");
        for (std::string line; std::getline(status, line);) {
            if (line.starts_with("VmHWM:")) {
                return std::stoull(line.substr(6));
            }
        }
        return std::nullopt;
    }

    // Sets the peak resident set size to the current one (Linux 4.0+: writing 5 to clear_refs).
    auto ResetPeakRss() -> bool
    {
        std::ofstream out("/proc/self/clear_refs");
        out << "5";
        out.flush();
        return static_cast<bool>(out);
    }
#endif
} // namespace

TEST_CASE("Checkpoint BEVE rejects malformed trees", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    // the trees below are accepted when they are well-formed
    CHECK(Decodes(EncodeWithNodes({ Const(), Const(), Fn(Operon::BuiltinOp::Add, 2) })));
    CHECK(Decodes(EncodeWithNodes({ Const(), Const(), Const(), Fn(Operon::BuiltinOp::Add, 3) }))); // n-ary
    CHECK(Decodes(EncodeWithNodes({ Const(), Fn(Operon::BuiltinOp::Sub, 1) }))); // negation
    CHECK(Decodes(EncodeWithNodes({ Const(), RefTo(0), Fn(Operon::BuiltinOp::Mul, 2) }))); // a shared subtree
    CHECK(Decodes(EncodeWithNodes({ Variable(), Fn(Operon::BuiltinOp::Sin, 1) })));
    CHECK(Decodes(EncodeWithNodes({ Variable(), Const(), Fn(Operon::BuiltinOp::Pow, 2) })));

    // a Ref that points to itself, and one that points forward (to the root)
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), RefTo(1), Fn(Operon::BuiltinOp::Add, 2) })));
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), RefTo(2), Fn(Operon::BuiltinOp::Add, 2) })));
    // terminals with arguments (these are well-formed postfix expressions: each consumes the nodes before it)
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), Const(), WithArity(Const(), 2) })));
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), WithArity(Variable(), 1) })));
    // a Ref with an argument
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), WithArity(RefTo(0), 1) })));
    // binary functions with one argument, a unary one with two, a function without arguments
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), Fn(Operon::BuiltinOp::Pow, 1) })));
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), Fn(Operon::BuiltinOp::Aq, 1) })));
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), Const(), Const(), Fn(Operon::BuiltinOp::Powabs, 3) })));
    CHECK_FALSE(Decodes(EncodeWithNodes({ Const(), Const(), Fn(Operon::BuiltinOp::Sin, 2) })));
    CHECK_FALSE(Decodes(EncodeWithNodes({ Fn(Operon::BuiltinOp::Add, 0) })));
    // a node type that is not one of the four
    for (auto const type : { 4U, 7U, 255U }) {
        auto n = Const();
        n.Type = static_cast<Operon::NodeType>(type);
        CHECK_FALSE(Decodes(EncodeWithNodes({ n })));
    }
    // an empty genotype among the parents
    {
        auto cp = MakeSmallCheckpoint();
        cp.Population[1].Genotype = Operon::Tree {};
        CHECK_FALSE(Decodes(Operon::Serialization::ToBeve(cp)));
    }
}

TEST_CASE(
    "Checkpoint BEVE limits the nodes of a tree to what the 16-bit metadata can hold", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    auto const wide = [](std::size_t children) {
        std::vector<Operon::Node> nodes(children, Const());
        nodes.push_back(Fn(Operon::BuiltinOp::Add, static_cast<std::uint16_t>(children)));
        return nodes;
    };
    CHECK(Decodes(EncodeWithNodes(wide(65534)))); // 65535 nodes
    CHECK_FALSE(Decodes(EncodeWithNodes(wide(65535)))); // 65536
}

TEST_CASE("Checkpoint BEVE limits the other arrays", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    // worker random states
    {
        auto cp = MakeSmallCheckpoint();
        cp.WorkerRngStates.assign(Operon::Serialization::MaxCheckpointWorkerStates, cp.RngState);
        CHECK(Decodes(Operon::Serialization::ToBeve(cp)));
        cp.WorkerRngStates.push_back(cp.RngState);
        CHECK_FALSE(Decodes(Operon::Serialization::ToBeve(cp)));
    }
    // objectives of an individual
    {
        auto cp = MakeSmallCheckpoint();
        cp.Population[1].Fitness.assign(64, Operon::Scalar { 1 });
        CHECK(Decodes(Operon::Serialization::ToBeve(cp)));
        cp.Population[1].Fitness.push_back(Operon::Scalar { 1 });
        CHECK_FALSE(Decodes(Operon::Serialization::ToBeve(cp)));
    }
    // fitness values of a cache entry
    {
        auto cp = MakeSmallCheckpoint();
        cp.Cache = Operon::Serialization::CacheCheckpoint {};
        cp.Cache->Keys = { 1 };
        cp.Cache->InsertGenerations = { 0 };
        cp.Cache->Lengths = { 64 };
        cp.Cache->Values.assign(64, Operon::Scalar { 1 });
        CHECK(Decodes(Operon::Serialization::ToBeve(cp)));
        cp.Cache->Lengths = { 65 };
        cp.Cache->Values.push_back(Operon::Scalar { 1 });
        CHECK_FALSE(Decodes(Operon::Serialization::ToBeve(cp)));
    }
    // groups of the run fingerprint
    {
        auto cp = MakeSmallCheckpoint();
        cp.Run = Operon::Serialization::RunFingerprint {};
        for (std::size_t i = 0; i < 4096; ++i) {
            cp.Run->Groups.push_back("group." + std::to_string(i));
            cp.Run->Hashes.push_back(i);
        }
        CHECK(Decodes(Operon::Serialization::ToBeve(cp)));
        cp.Run->Groups.push_back("one.more");
        cp.Run->Hashes.push_back(0);
        CHECK_FALSE(Decodes(Operon::Serialization::ToBeve(cp)));
    }
}

TEST_CASE("Checkpoint BEVE rejects huge declared counts without allocating for them", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    auto const small = Operon::Serialization::ToBeve(MakeSmallCheckpoint());
    REQUIRE(Decodes(small));
    REQUIRE(Decodes(DeclarePopulation(small, 2))); // the helper itself leaves the file as it was

    // more individuals than the limit although the file has the bytes to back the count
    CHECK_FALSE(Decodes(DeclarePopulation(small, (1U << 20U) + 1, (1U << 20U) + 4096)));
    // 2^26 declared in a file of that many bytes (the review's hostile file: 64 MiB, 4 GiB with the old bound)
    auto const hostile = DeclarePopulation(small, std::uint64_t { 1 } << 26U, std::size_t { 1 } << 26U);
#if defined(__linux__)
    auto const reset = ResetPeakRss();
#endif
    CHECK_FALSE(Decodes(hostile));
#if defined(__linux__)
    if (reset) {
        auto const peak = PeakRssKiB();
        REQUIRE(peak.has_value());
        // the 64 MiB buffer and the test binary itself (the peak depends on what the earlier tests left behind, so
        // the bound is loose); with the old bound the decoder allocated 4 GiB for the proxies
        CHECK(*peak < 1024U * 1024U);
    }
#endif
}

// A deterministic pseudo-random damage of a valid checkpoint: a few hundred single- and multi-byte flips (and byte
// replacements) are all rejected by the envelope checksum. With the checksum recomputed (damage of the payload that
// the checksum cannot see, the way a crafted file looks) they must still neither crash nor hang, and whatever is
// accepted must hold only valid trees.
TEST_CASE("Checkpoint BEVE decoding of flipped bits is rejected, or valid when resealed", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    auto const bytes = Operon::Serialization::ToBeve(MakeSmallCheckpoint());
    REQUIRE(Decodes(bytes));
    Operon::RandomGenerator rng(0xF1F0U);
    std::size_t accepted = 0;
    std::size_t acceptedResealed = 0;
    constexpr std::size_t Rounds = 600;
    for (std::size_t round = 0; round < Rounds; ++round) {
        auto damaged = bytes;
        auto const flips = round < Rounds / 2 ? 1U : 2U + static_cast<std::size_t>(rng() % 3U); // 1, then 2 to 4
        for (std::size_t k = 0; k < flips; ++k) {
            auto const at = static_cast<std::size_t>(rng() % damaged.size());
            auto const how = rng() % 3U;
            if (how == 0) {
                damaged[at] = static_cast<char>(static_cast<unsigned char>(damaged[at]) ^ (1U << (rng() % 8U)));
            } else if (how == 1) {
                damaged[at] = static_cast<char>(rng() & 0xFFU);
            } else {
                damaged[at] = static_cast<char>(static_cast<unsigned char>(damaged[at]) ^ 0xFFU);
            }
        }
        if (damaged == bytes) {
            continue; // a replacement by the same byte changes nothing
        }
        if (Operon::Serialization::CheckpointFromBeve(damaged)) {
            ++accepted;
        }
        if (damaged.size() >= Operon::Serialization::CheckpointEnvelopeBytes) {
            Reseal(damaged);
            if (auto const cp = Operon::Serialization::CheckpointFromBeve(damaged)) {
                ++acceptedResealed;
                CHECK(CheckpointProblem(*cp).empty());
            }
        }
    }
    CHECK(accepted == 0);
    CHECK(acceptedResealed < Rounds); // not everything is noticed: value bytes can change freely
    CHECK(acceptedResealed > 0);
}

static_assert(Operon::Serialization::MaxCheckpointPopulation == (1U << 20U));
static_assert(Operon::Serialization::MaxCheckpointWorkerStates == (std::size_t { 1 } << 20U));
static_assert(Operon::Serialization::MaxCheckpointTreeNodes == 65535);
static_assert(Operon::Serialization::MaxCheckpointObjectives == 64);
static_assert(Operon::Serialization::MaxCheckpointCacheEntries == (1U << 26U));
static_assert(Operon::Serialization::MaxCheckpointFingerprintEntries == 4096);

TEST_CASE("SaveCheckpoint refuses a file the reader would refuse", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    auto const path = std::filesystem::temp_directory_path() / "operon_test_save_limits.beve";
    auto const tmp = std::filesystem::path { path.string() + ".tmp" };
    std::error_code ec;
    std::filesystem::remove(path, ec);

    auto const cp = MakeSmallCheckpoint();
    auto const size = Operon::Serialization::ToBeve(cp).size();
    REQUIRE(size > 0);

    // the file may have exactly the number of bytes the limit allows, and not one more
    REQUIRE(Operon::Serialization::SaveCheckpoint(cp, path.string(), size));
    auto const loaded = Operon::Serialization::LoadCheckpoint(path.string());
    REQUIRE(loaded.has_value());
    CHECK(loaded->Generation == cp.Generation);

    // a refused checkpoint leaves the file that is there (and no temporary file) alone
    auto newer = cp;
    newer.Generation = 99;
    CHECK_FALSE(Operon::Serialization::SaveCheckpoint(newer, path.string(), size - 1));
    CHECK_FALSE(std::filesystem::exists(tmp));
    auto const kept = Operon::Serialization::LoadCheckpoint(path.string());
    REQUIRE(kept.has_value());
    CHECK(kept->Generation == cp.Generation);

    // the limit cannot be raised above what the reader accepts
    CHECK(Operon::Serialization::SaveCheckpoint(
        newer, path.string(), std::numeric_limits<std::uint64_t>::max())); // the file is small: accepted
    // counts the reader would refuse are refused at the writer, whatever the size
    auto tooManyWorkers = cp;
    tooManyWorkers.WorkerRngStates.assign(Operon::Serialization::MaxCheckpointWorkerStates + 1, cp.RngState);
    CHECK_FALSE(Operon::Serialization::SaveCheckpoint(tooManyWorkers, path.string()));
    auto tooManyObjectives = cp;
    tooManyObjectives.Population[0].Fitness.assign(65, Operon::Scalar { 1 });
    CHECK_FALSE(Operon::Serialization::SaveCheckpoint(tooManyObjectives, path.string()));
    auto tooLong = cp;
    {
        std::vector<Operon::Node> nodes(65536, Const());
        tooLong.Population[0].Genotype = Operon::Tree { Operon::Vector<Operon::Node>(nodes.begin(), nodes.end()) };
    }
    CHECK_FALSE(Operon::Serialization::SaveCheckpoint(tooLong, path.string()));
    // a fingerprint group name the reader would refuse (its limit is the same constant)
    auto longName = cp;
    longName.Run = Operon::Serialization::RunFingerprint {
        { std::string(Operon::Serialization::MaxCheckpointGroupNameBytes + 1, 'g') }, { 1 }
    };
    CHECK_FALSE(Operon::Serialization::SaveCheckpoint(longName, path.string()));
    longName.Run->Groups[0].pop_back(); // exactly the limit: written and read back
    CHECK(Operon::Serialization::SaveCheckpoint(longName, path.string()));
    CHECK(Operon::Serialization::LoadCheckpoint(path.string()).has_value());
    CHECK(Operon::Serialization::SaveCheckpoint(newer, path.string())); // restore the file the next checks expect
    CHECK_FALSE(std::filesystem::exists(tmp));
    auto const still = Operon::Serialization::LoadCheckpoint(path.string());
    REQUIRE(still.has_value());
    CHECK(still->Generation == 99);

    // and into a directory that does not exist: an error, not an exception
    CHECK_FALSE(Operon::Serialization::SaveCheckpoint(
        cp, (std::filesystem::temp_directory_path() / "operon_no_such_dir" / "cp.beve").string()));
    std::filesystem::remove(path, ec);
}

TEST_CASE("Tree and Individual decoders reject trees that would crash the library", "[serialization]")
{
    QuietStderr const quiet;
    auto const forward
        = Operon::Tree { Operon::Vector<Operon::Node> { Const(), RefTo(2), Fn(Operon::BuiltinOp::Add, 2) } };
    Operon::Individual individual;
    individual.Genotype = forward;
    CHECK_FALSE(Operon::Serialization::TreeFromBeve(Operon::Serialization::ToBeve(forward)).has_value());
    CHECK_FALSE(Operon::Serialization::IndividualFromBeve(Operon::Serialization::ToBeve(individual)).has_value());
    CHECK_FALSE(Operon::Serialization::TreeFromJson(Operon::Serialization::ToJson(forward)).has_value());
    CHECK_FALSE(Operon::Serialization::IndividualFromJson(Operon::Serialization::ToJson(individual)).has_value());

    // and a valid tree still goes through every one of them
    Operon::RandomGenerator rng(5);
    auto const tree = MakeTree(rng);
    CHECK(Operon::Serialization::TreeFromBeve(Operon::Serialization::ToBeve(tree)).has_value());
    CHECK(Operon::Serialization::TreeFromJson(Operon::Serialization::ToJson(tree)).has_value());
}

TEST_CASE("Checkpoint files above the size limit are refused without being read", "[serialization][checkpoint]")
{
    QuietStderr const quiet;
    auto const path = std::filesystem::temp_directory_path() / "operon_test_huge_checkpoint.beve";
    {
        std::ofstream out(path, std::ios::binary);
        out << "x";
    }
    std::error_code ec;
    std::filesystem::resize_file(path, Operon::Serialization::MaxCheckpointFileBytes + 1, ec);
    if (!ec) { // sparse on every file system that matters; skipped where the file cannot be created
        CHECK_FALSE(Operon::Serialization::LoadCheckpoint(path.string()).has_value());
    }
    std::filesystem::remove(path, ec);
    CHECK_FALSE(Operon::Serialization::LoadCheckpoint(
        (std::filesystem::temp_directory_path() / "operon_no_such_checkpoint").string())
            .has_value());
}

TEST_CASE("Checkpoint BEVE preserves WorkerRngStates for CLI validation", "[serialization]")
{
    // The size-mismatch check lives in the CLI layer (ResumeFromCheckpoint),
    // not in the serialization layer.  Verify the payload survives the BEVE
    // round-trip intact so the CLI check can fire on the correct values.
    Operon::RandomGenerator rng(5);
    Operon::Serialization::Checkpoint cp;
    cp.RngState = rng.state();
    cp.Generation = 2;
    cp.Population.resize(4);
    for (auto& ind : cp.Population) {
        ind = MakeIndividual(rng);
    }
    cp.WorkerRngStates.resize(3); // intentionally mismatched (3 vs popSize 4)
    for (auto& s : cp.WorkerRngStates) {
        s = rng.state();
    }

    auto beve = Operon::Serialization::ToBeve(cp);
    auto restored = Operon::Serialization::CheckpointFromBeve(beve);
    REQUIRE(restored);
    CHECK(restored->WorkerRngStates.size() == 3);
    CHECK(restored->Population.size() == 4);
}

TEST_CASE("TreeFromJson returns empty tree for empty node list", "[serialization]")
{
    // Previously: UB crash via UpdateNodes() touching nodes_.back() on empty vector.
    auto tree = Operon::Serialization::TreeFromJson(R"({"nodes":[]})");
    REQUIRE(tree);
    CHECK(tree->Length() == 0);
}

TEST_CASE("TreeFromBeve returns empty tree for empty node list", "[serialization]")
{
    // A default-constructed Tree serializes with an empty node array; deserializing
    // it must return a zero-length tree, not crash via UpdateNodes() on nodes_.back().
    Operon::Tree emptyTree;
    auto beve = Operon::Serialization::ToBeve(emptyTree);
    auto restored = Operon::Serialization::TreeFromBeve(beve);
    REQUIRE(restored);
    CHECK(restored->Length() == 0);
}

#ifndef _WIN32
TEST_CASE("SaveCheckpoint preserves destination on non-file rename failure", "[serialization]")
{
    // Place a directory at the checkpoint path so rename(tmp, path) fails with
    // EISDIR — not the "dest is an existing file" case.  The directory must
    // survive: SaveCheckpoint must not remove it and must not crash.
    auto const blockPath = std::filesystem::temp_directory_path() / "operon_test_rename_fail_block";
    std::filesystem::create_directories(blockPath);

    Operon::RandomGenerator rng(77);
    Operon::Serialization::Checkpoint cp;
    cp.RngState = rng.state();
    cp.Generation = 1;
    cp.Population.resize(2);
    for (auto& ind : cp.Population) {
        ind = MakeIndividual(rng);
    }

    Operon::Serialization::SaveCheckpoint(cp, blockPath.string()); // must not crash or delete the dir

    CHECK(std::filesystem::is_directory(blockPath)); // directory still intact

    std::error_code ec;
    std::filesystem::remove_all(blockPath, ec);
    // clean up the .tmp file if it was left behind
    std::filesystem::remove(blockPath.string() + ".tmp", ec);
}
#endif

TEST_CASE("SaveCheckpoint is crash-safe: second save overwrites first", "[serialization]")
{
    Operon::RandomGenerator rng(11);

    auto makeCp = [&](std::size_t gen) {
        Operon::Serialization::Checkpoint cp;
        cp.RngState = rng.state();
        cp.Generation = gen;
        cp.Population.resize(4);
        for (auto& ind : cp.Population) {
            ind = MakeIndividual(rng);
        }
        return cp;
    };

    auto const path = (std::filesystem::temp_directory_path() / "operon_test_overwrite.beve").string();

    auto first = makeCp(1);
    auto second = makeCp(2);

    Operon::Serialization::SaveCheckpoint(first, path);
    Operon::Serialization::SaveCheckpoint(second, path);

    auto restored = Operon::Serialization::LoadCheckpoint(path);
    REQUIRE(restored);
    CHECK(restored->Generation == 2);

    std::error_code ec;
    std::filesystem::remove(path, ec);
}

} // namespace Operon::Test
