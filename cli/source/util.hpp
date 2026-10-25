// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_UTIL_HPP
#define OPERON_CLI_UTIL_HPP

#include "operon/algorithms/ga_base.hpp"
#include "operon/core/constraint.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/range.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/types.hpp"
#include <chrono>
#include <cstddef>
#include <cxxopts.hpp>
#include <string>
#include <utility>
#include <vector>

namespace Operon {
constexpr int optionsWidth = 200;
auto ParseRange(std::string const& str) -> std::pair<size_t, size_t>;
auto Split(const std::string& s, char delimiter) -> std::vector<std::string>;
auto FormatBytes(size_t bytes) -> std::string;
auto FormatDuration(std::chrono::duration<double> d) -> std::string;
auto ParsePrimitiveSetConfig(const std::string& options) -> PrimitiveSetConfig;
auto PrintPrimitives(PrimitiveSetConfig config) -> void;

auto InitOptions(std::string const& name, std::string const& desc, int width = optionsWidth) -> cxxopts::Options;
auto ParseOptions(cxxopts::Options&& opts, int argc, char** argv) -> cxxopts::ParseResult;

// Value of a budget option (--generations, --evaluations): an integer or `unlimited` (>= 2^62 also means unlimited),
// returned as SIZE_MAX when unlimited. ParseOptions has already validated the text, so this cannot fail.
auto GetLimit(cxxopts::ParseResult const& result, std::string const& name) -> size_t;

// Set trainingRange and testRange from CLI options, inferring defaults from dataset if not provided.
auto SetupRanges(cxxopts::ParseResult const& result, Dataset const& dataset, Range& trainingRange, Range& testRange)
    -> void;

// Look up the target variable by name.
// Throws std::runtime_error if the named variable does not exist in the dataset.
auto ResolveTarget(Dataset const& dataset, std::string const& targetName) -> Variable;

// Return input variable hashes from CLI options, excluding targetHash.
// Throws std::runtime_error if a named variable does not exist in the dataset.
auto BuildInputs(cxxopts::ParseResult const& result, Dataset const& dataset, Hash targetHash) -> std::vector<Hash>;

// The options of the program as ParseOptions saw them: for every registered option its effective value (the last
// explicit value, else the default, else "<unset>"), in registration order. Valid after ParseOptions.
auto NormalizedOptions() -> std::vector<std::pair<std::string, std::string>> const&;

// Fingerprint of everything that determines what the algorithm computes and how the run continues: the build identity
// (Operon revision, scalar precision, architecture and ISA level: see BuildIdentityHash), the dataset contents and the
// resolved training/test ranges, target and inputs, the parsed content of --shape-constraints-config (pass nullptr if
// none is configured; the path is not part of it), the resolved algorithm configuration and objective count, and every
// option of NormalizedOptions() except the ones that may legitimately change on resume (see
// RunFingerprintExcludedOptions). The programs compute it once, after the dataset has been prepared (--shuffle,
// --standardize), and pass it to ResumeFromCheckpoint and MaybeSaveCheckpoint. It is stored in every checkpoint and
// compared by ResumeFromCheckpoint before anything else of the checkpoint is used. Entries are named "build.identity",
// "dataset.content", "shape.constraints", "config.<field>", "option.<name>", ... The hashes use the native byte
// representation of the data, so a fingerprint is only comparable on the architecture that wrote it.
auto MakeRunFingerprint(GeneticAlgorithmBase const& algo, ShapeConstraintSet const* shapeConstraints)
    -> Serialization::RunFingerprint;

// Options that are NOT part of the run fingerprint: they may change on resume (--threads, --generations,
// --evaluations, --timelimit), only affect what is reported or written (--checkpoint-*, --pareto-front,
// --model-selection, --mdl-likelihood, --probes-config, --report-json, --debug), name the data by path (--dataset,
// --shape-constraints-config, resolved to their content instead; --train/--test/--target/--inputs, resolved likewise),
// or are checked separately against the cache stored in the checkpoint (--transposition-cache, --cache-visibility,
// --cache-max-age, --seed: the seed only matters through the cache's random table).
auto RunFingerprintExcludedOptions() -> std::vector<std::string> const&;

// Restore algorithm state from the checkpoint file given with --resume. Returns true if a checkpoint was applied
// (the caller passes warmStart=true to Run()), false if --resume was not given or names a file that does not exist
// (a warning is printed and the run starts fresh: the "resume if present" use). Every other problem is an error:
// throws std::runtime_error for a checkpoint that cannot be read, is corrupt or of another format version, whose
// population size or worker count does not match, whose run fingerprint differs from `fingerprint` (another build, or
// other settings, naming the differing entries), whose generation is above a finite --generations, that has a
// transposition cache while this run has none or the reverse, whose cache cannot be continued by this run's cache, or
// whose parents do not fit this run (a variable that is not an input of the problem, a function that is not an
// enabled primitive, a tree longer than --maxlength). All of this is checked before any state of `algo` is replaced.
// On success the algorithm object is marked fitted (IsFitted()), which is what lets the warm Run() continue from the
// restored population.
auto ResumeFromCheckpoint(GeneticAlgorithmBase& algo, RandomGenerator& rng, cxxopts::ParseResult const& result,
    Serialization::RunFingerprint const& fingerprint) -> bool;

// Save a checkpoint (carrying `fingerprint`) if --checkpoint-interval is set and the current generation is due. Pass
// force=true to save unconditionally (e.g. at end of run).
// No-op when interval == 0 or when Generation() == 0 (initial evaluation).
auto MaybeSaveCheckpoint(GeneticAlgorithmBase const& algo, RandomGenerator const& rng,
    cxxopts::ParseResult const& result, Serialization::RunFingerprint const& fingerprint, bool force = false) -> void;
} // namespace Operon
#endif
