// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_CLI_UTIL_HPP
#define OPERON_CLI_UTIL_HPP

#include <charconv>
#include <chrono>
#include <fmt/core.h>
#include <sstream>
#include <string>

#include "operon/core/node.hpp"

namespace Operon {

auto ParseRange(std::string const& str) -> std::pair<size_t, size_t>;
auto Split(const std::string& s, char delimiter) -> std::vector<std::string>;
auto FormatBytes(size_t bytes) -> std::string;
auto FormatDuration(std::chrono::duration<double> d) -> std::string;
auto ParsePrimitiveSetConfig(const std::string& options) -> NodeType;
auto PrintPrimitives(PrimitiveSetConfig config) -> void;

} // namespace Operon

#endif
