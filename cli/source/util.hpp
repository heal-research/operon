// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CLI_UTIL_HPP
#define OPERON_CLI_UTIL_HPP

#include "operon/core/dataset.hpp"
#include "operon/core/node.hpp"
#include "operon/core/range.hpp"
#include "operon/core/types.hpp"
#include <chrono>
#include <cstddef>
#include <cxxopts.hpp>
#include <fmt/core.h>
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
} // namespace Operon
#endif
