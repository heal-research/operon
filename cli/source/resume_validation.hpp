// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_RESUME_VALIDATION_HPP
#define OPERON_CLI_RESUME_VALIDATION_HPP

#include <cstddef>
#include <cstdint>
#include <string>
#include <string_view>
#include <vector>

#include "operon/core/constraint.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/types.hpp"
#include "operon/core/version.hpp"

// The checks that `--resume` applies to a checkpoint before anything of it is used, and the hashes that make up the
// run fingerprint. They are free of option parsing so that they can be tested directly; ResumeFromCheckpoint (util.hpp)
// calls them. Every check throws std::runtime_error with a message that names the checkpoint and what is wrong.
namespace Operon {

// Name of the fingerprint entry that holds BuildIdentityHash().
inline constexpr std::string_view BuildIdentityGroup = "build.identity";

// The hash used by the run fingerprint (not a cryptographic hash, and it depends on the byte representation of what it
// is given: fingerprints are only comparable between runs on the same architecture).
auto HashBytes(void const* data, std::size_t size) -> std::uint64_t;
auto HashText(std::string_view text) -> std::uint64_t;
auto CombineHashes(std::uint64_t seed, std::uint64_t value) -> std::uint64_t;

// Identity of the build for the fingerprint entry "build.identity": the Operon revision (BuildInfo::Revision, "unknown"
// when the build had no git revision, in which case only the other components can tell builds apart), the scalar
// precision and the architecture and x86-64 ISA level (the levels that enable FMA and wider vectors change the
// rounding of floating point results). The compiler and its flags are not part of it.
auto BuildIdentityHash(BuildInfo const& info) -> std::uint64_t;

// The parsed content of --shape-constraints-config, for the fingerprint entry "shape.constraints" (the path of the file
// is not part of the fingerprint, its content is): the domains (independent of their order in the file) and the
// constraints in order. nullptr (no constraints configured) hashes to a value no constraint set has.
auto ShapeConstraintsHash(ShapeConstraintSet const* constraints) -> std::uint64_t;

// The entries of `current` that differ from `saved`, or are missing in either, as the user sees them ("--objective" for
// the entry "option.objective").
auto FingerprintDifferences(Serialization::RunFingerprint const& saved, Serialization::RunFingerprint const& current)
    -> std::vector<std::string>;

// Empty if the fingerprints agree. Otherwise the error message for the user: if the build identity differs, that the
// checkpoint was written by another build (and nothing else, the other entries are meaningless then); else the list of
// differing entries.
auto FingerprintMismatchMessage(std::string const& path, Serialization::RunFingerprint const& saved,
    Serialization::RunFingerprint const& current) -> std::string;

// A finite generation limit (--generations) below the generation of the checkpoint cannot be continued: the run would
// have to end before it starts. An unlimited run has limit SIZE_MAX and is never rejected.
auto CheckGenerationLimit(std::string const& path, std::uint64_t checkpointGeneration, std::size_t generationLimit)
    -> void;

// The checkpoint's cache and the run's cache must both exist or both not: a cache-bearing checkpoint resumed without
// --transposition-cache loses the entries the checkpointed run would have hit, and a cache-less one resumed with it
// starts a cache the checkpointed run never had; either run would not continue the checkpointed one.
auto CheckCachePresence(std::string const& path, bool checkpointHasCache, bool runHasCache) -> void;

// Every parent of the checkpoint against what this run can evaluate: no empty genotype, at most `maxLength` nodes,
// every Variable node an input of `problem`, every Function node a registered and enabled primitive of its primitive
// set. (The structure of the trees themselves is validated by the checkpoint reader.)
auto CheckPopulation(std::string const& path, Operon::Vector<Individual> const& population, Problem const& problem,
    std::size_t maxLength) -> void;

} // namespace Operon

#endif
