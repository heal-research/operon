// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CLI_UTIL_HPP
#define OPERON_CLI_UTIL_HPP

#include <charconv>
#include <chrono>
#include <fmt/core.h>
#include <sstream>
#include <string>

#include <cxxopts.hpp>

#include "operon/core/node.hpp"

namespace Operon {

constexpr int optionsWidth = 200;

auto ParseRange(std::string const& str) -> std::pair<size_t, size_t>;
auto Split(const std::string& s, char delimiter) -> std::vector<std::string>;
auto FormatBytes(size_t bytes) -> std::string;
auto FormatDuration(std::chrono::duration<double> d) -> std::string;
auto ParsePrimitiveSetConfig(const std::string& options) -> NodeType;
auto PrintPrimitives(PrimitiveSetConfig config) -> void;
auto PrintStats(std::vector<std::tuple<std::string, double, std::string>> const& stats, bool printHeader = true) -> void;

auto InitOptions(std::string const& name, std::string const& desc, int width = optionsWidth) -> cxxopts::Options;
auto ParseOptions(cxxopts::Options&& opts, int argc, char** argv) -> cxxopts::ParseResult;

} // namespace Operon
#endif
