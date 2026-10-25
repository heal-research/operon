// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_SERIALIZATION_HPP
#define OPERON_SERIALIZATION_HPP

#include <array>
#include <cstdint>
#include <optional>
#include <span>
#include <string>
#include <string_view>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/operon_export.hpp"

namespace Operon {
struct ZobristState;
}

namespace Operon::Serialization {

// ---- JSON ----

OPERON_EXPORT auto ToJson(Tree const& tree) -> std::string;
OPERON_EXPORT auto ToJson(Individual const& individual) -> std::string;
OPERON_EXPORT auto ToJson(std::span<Individual const> front) -> std::string;

OPERON_EXPORT auto TreeFromJson(std::string_view json) -> std::optional<Tree>;
OPERON_EXPORT auto IndividualFromJson(std::string_view json) -> std::optional<Individual>;

// ---- BEVE (binary) ----
//
// Tree and Individual payloads are self-identifying and versioned. Readers reject
// unversioned or incompatible data instead of attempting a potentially corrupt
// decode. Use this format for persisted models and enumeration corpora.

OPERON_EXPORT auto ToBeve(Tree const& tree) -> std::string;
OPERON_EXPORT auto ToBeve(Individual const& individual) -> std::string;
OPERON_EXPORT auto ToBeve(std::span<Individual const> front) -> std::string;

OPERON_EXPORT auto TreeFromBeve(std::string_view data) -> std::optional<Tree>;
OPERON_EXPORT auto IndividualFromBeve(std::string_view data) -> std::optional<Individual>;

// ---- Checkpoint (algorithm save / resume) ----

// The transposition cache of a run (Zobrist::Export), stored as parallel arrays so that the binary encoding is
// compact: entry i has the key Keys[i], was inserted in generation InsertGenerations[i] and holds the Lengths[i]
// fitness values that follow the values of entries 0..i-1 in Values. Keys are in ascending order.
struct OPERON_EXPORT CacheCheckpoint {
    uint64_t TableFingerprint { 0 };
    uint32_t Clock { 0 };
    uint64_t MaxAge { 0 };
    uint8_t Visibility { 1 }; // CacheVisibility: 0 = immediate, 1 = committed
    std::vector<uint64_t> Keys;
    std::vector<uint32_t> InsertGenerations;
    std::vector<uint32_t> Lengths;
    std::vector<Operon::Scalar> Values;
};

OPERON_EXPORT auto ToCacheCheckpoint(ZobristState const& state) -> CacheCheckpoint;
// Throws std::invalid_argument if the arrays are inconsistent (different lengths, a fitness length of zero or a sum of
// lengths that is not the number of values, keys that are not strictly ascending, an unknown visibility, a maximum age
// that is not representable); the checks use checked arithmetic and never index out of bounds.
OPERON_EXPORT auto ToZobristState(CacheCheckpoint const& cache) -> ZobristState;

// What the checkpointed run computed with, as a set of named groups of options (dataset, evaluation, search, ...), each
// reduced to a hash. The writer decides the groups; a reader compares them group by group before it trusts anything
// the checkpoint carries (fitness values, cache entries). Groups and Hashes have the same length.
struct OPERON_EXPORT RunFingerprint {
    std::vector<std::string> Groups;
    std::vector<uint64_t> Hashes;
};

// Everything needed to continue a run as if it had not been interrupted: the master and worker random states, the
// generation counter, the parent population and, when the run has a transposition cache, the cache (an entry's
// presence or absence changes what the next generations compute). The file format is versioned; a checkpoint of
// another version is rejected with an error message instead of being read.
struct OPERON_EXPORT Checkpoint {
    std::array<uint64_t, 4> RngState {};
    uint64_t Generation { 0 };
    Operon::Vector<Individual> Population;
    std::vector<std::array<uint64_t, 4>> WorkerRngStates;
    std::optional<CacheCheckpoint> Cache; // absent: the run had no transposition cache
    std::optional<RunFingerprint> Run; // absent: written by a program that does not record one
};

// Limits applied when a checkpoint is read, so that a corrupt or hostile file is rejected before it is allocated:
// the file may not be larger than MaxCheckpointFileBytes, and the declared number of elements of every array has its
// own bound below and cannot exceed the bytes that are left in the file. A checkpoint written by this program within
// these limits is always readable; SaveCheckpoint refuses to write a checkpoint the reader would refuse for its size or
// one of these counts.
inline constexpr std::uint64_t MaxCheckpointFileBytes = std::uint64_t { 1 } << 30U;
// individuals in the (parent) population
inline constexpr std::size_t MaxCheckpointPopulation = std::size_t { 1 } << 20U;
// Random states of the offspring-generation workers. The command-line programs keep one per population slot or pool
// slot (max of --population-size and --pool-size), not one per thread, so the bound is the one of the population.
// SaveCheckpoint refuses (as it does for any checkpoint the reader would refuse) instead of writing a file that could
// not be resumed.
inline constexpr std::size_t MaxCheckpointWorkerStates = std::size_t { 1 } << 20U;
// nodes of a tree: Length, Parent and RefTo of a Node are 16-bit
inline constexpr std::size_t MaxCheckpointTreeNodes = 65535;
// fitness values of an individual and of a cache entry
inline constexpr std::size_t MaxCheckpointObjectives = 64;
// entries of the transposition cache (keys, insertion generations, lengths); the fitness values of all entries are
// bounded by this times MaxCheckpointObjectives
inline constexpr std::size_t MaxCheckpointCacheEntries = std::size_t { 1 } << 26U;
// groups of the run fingerprint
inline constexpr std::size_t MaxCheckpointFingerprintEntries = 4096;
// bytes of a run fingerprint group name
inline constexpr std::size_t MaxCheckpointGroupNameBytes = 1024;
// A serialized checkpoint is an envelope followed by the BEVE payload: magic (4 bytes), format version (4 bytes) and
// the checksum of the payload (8 bytes, Operon::Hasher over the payload bytes), all little endian. The reader checks
// magic, version and checksum before it decodes anything and reports a mismatch as a corrupt checkpoint.
inline constexpr std::size_t CheckpointEnvelopeBytes = 16;

// The encoded checkpoint, or an empty string (after printing the reason to stderr) if it cannot be encoded. The
// reader's limits are not applied here (SaveCheckpoint applies them before it writes).
OPERON_EXPORT auto ToBeve(Checkpoint const&) -> std::string;
// Returns nullopt, after printing the reason to stderr (never the data), for anything that is not a valid checkpoint of
// this format version: truncated or corrupt data, an older or newer version, an array above its limit, arrays that are
// inconsistent with each other, or a tree that is not valid. Every tree of the population is checked before it is
// accepted, without a primitive set: at least one node, at most MaxCheckpointTreeNodes nodes, a valid node type, a
// postfix expression with exactly one root, Ref nodes pointing to an earlier node, terminals (Constant, Variable, Ref)
// without arguments, functions with at least one argument and the fixed arity of a binary (Aq, Pow, Powabs) or unary
// built-in function, and the derived node metadata that Tree::Validate checks. What needs the dataset or the primitive
// set (Variable hashes that exist, registered functions, the arity range of user functions and of the n-ary built-in
// ones, the maximum length of the problem) is the caller's to check. Never throws and never indexes out of bounds on
// malformed input; an allocation failure gives nullopt too.
OPERON_EXPORT auto CheckpointFromBeve(std::string_view data) -> std::optional<Checkpoint>;

// Returns true on success; false (error printed to stderr, an existing file at `path` left as it was) if the checkpoint
// is larger than the reader's limit (`maxFileBytes`, itself capped at MaxCheckpointFileBytes), exceeds one of the count
// limits above, or the write or rename failed.
OPERON_EXPORT auto SaveCheckpoint(
    Checkpoint const&, std::string_view path, std::uint64_t maxFileBytes = MaxCheckpointFileBytes) -> bool;
OPERON_EXPORT auto LoadCheckpoint(std::string_view path) -> std::optional<Checkpoint>;

} // namespace Operon::Serialization

#endif
