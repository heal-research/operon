// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_UTIL_HPP
#define OPERON_CLI_UTIL_HPP

#include "operon/algorithms/ga_base.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/range.hpp"
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

// Set trainingRange and testRange from CLI options, inferring defaults from dataset if not provided.
auto SetupRanges(cxxopts::ParseResult const& result, Dataset const& dataset,
                 Range& trainingRange, Range& testRange) -> void;

// Return input variable hashes from CLI options, excluding targetHash.
// Throws std::runtime_error if a named variable does not exist in the dataset.
auto BuildInputs(cxxopts::ParseResult const& result, Dataset const& dataset,
                 Hash targetHash) -> std::vector<Hash>;

// Restore algorithm state from a checkpoint file specified via --resume.
// Returns true if a checkpoint was applied (caller should pass warmStart=true to Run()).
// Throws std::runtime_error if the checkpoint population size mismatches --population-size.
auto ResumeFromCheckpoint(GeneticAlgorithmBase& algo, RandomGenerator& rng,
                          cxxopts::ParseResult const& result) -> bool;

// Save a checkpoint if --checkpoint-interval is set and the current generation is due.
// Pass force=true to save unconditionally (e.g. at end of run).
// No-op when interval == 0 or when Generation() == 0 (initial evaluation).
auto MaybeSaveCheckpoint(GeneticAlgorithmBase const& algo, RandomGenerator const& rng,
                         cxxopts::ParseResult const& result, bool force = false) -> void;
} // namespace Operon
#endif
