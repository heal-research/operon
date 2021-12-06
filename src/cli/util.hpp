// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef CLI_UTIL_HPP
#define CLI_UTIL_HPP

#include <charconv>
#include <chrono>
#include <fmt/core.h>
#include <sstream>
#include <string>

#include "core/dataset.hpp"
#include "core/pset.hpp"

namespace Operon {
// parse a range specified as stard:end
Range ParseRange(const std::string& range)
{
    auto pos = static_cast<long>(range.find_first_of(':'));
    auto rangeFirst = std::string(range.begin(), range.begin() + pos);
    auto rangeLast = std::string(range.begin() + pos + 1, range.end());
    size_t begin, end;
    if (auto [p, ec] = std::from_chars(rangeFirst.data(), rangeFirst.data() + rangeFirst.size(), begin); ec != std::errc()) {
        throw std::runtime_error(fmt::format("Could not parse training range from argument \"{}\"", range));
    }
    if (auto [p, ec] = std::from_chars(rangeLast.data(), rangeLast.data() + rangeLast.size(), end); ec != std::errc()) {
        throw std::runtime_error(fmt::format("Could not parse training range from argument \"{}\"", range));
    }
    return { begin, end };
}

// parses a double value
auto ParseDouble(const std::string& str)
{
    char* end;
    double val = std::strtod(str.data(), &end);
    return std::make_pair(val, std::isspace(*end) || end == str.data() + str.size());
}

// splits a string into substrings separated by delimiter
std::vector<std::string> Split(const std::string& s, char delimiter)
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// formats a duration as dd:hh:mm:ss.ms
std::string FormatDuration(std::chrono::duration<double> d)
{
    auto h = std::chrono::duration_cast<std::chrono::hours>(d);
    auto m = std::chrono::duration_cast<std::chrono::minutes>(d - h);
    auto s = std::chrono::duration_cast<std::chrono::seconds>(d - h - m);
    auto l = std::chrono::duration_cast<std::chrono::milliseconds>(d - h - m - s);
    return fmt::format("{:#02d}:{:#02d}:{:#02d}.{:#03d}", h.count(), m.count(), s.count(), l.count());
}

std::string FormatBytes(size_t bytes)
{
    constexpr char sizes[] = " KMGT";
    auto p = static_cast<size_t>(std::floor(std::log2(bytes) / std::log2(1024)));
    return fmt::format("{:.2f} {}b", (double)bytes / std::pow(1024, p), sizes[p]);
}

static const std::unordered_map<std::string, NodeType> primitives {
    { "add",      NodeType::Add },
    { "mul",      NodeType::Mul },
    { "sub",      NodeType::Sub },
    { "div",      NodeType::Div },
    { "aq",       NodeType::Aq },
    { "fmax",     NodeType::Fmax },
    { "fmin",     NodeType::Fmin },
    { "pow",      NodeType::Pow },
    { "abs",      NodeType::Abs },
    { "acos",     NodeType::Acos },
    { "asin",     NodeType::Asin },
    { "atan",     NodeType::Atan },
    { "cbrt",     NodeType::Cbrt },
    { "ceil",     NodeType::Ceil },
    { "cos",      NodeType::Cos },
    { "cosh",     NodeType::Cosh },
    { "erf",      NodeType::Erf },
    { "erfc",     NodeType::Erfc },
    { "exp",      NodeType::Exp },
    { "floor",    NodeType::Floor },
    { "log",      NodeType::Log },
    { "log1p",    NodeType::Log1p },
    { "sin",      NodeType::Sin },
    { "sinh",     NodeType::Sinh },
    { "sqrt",     NodeType::Sqrt },
    { "square",   NodeType::Square },
    { "tan",      NodeType::Tan },
    { "tanh",     NodeType::Tanh },
    { "dyn",      NodeType::Dynamic },
    { "constant", NodeType::Constant },
    { "variable", NodeType::Variable }
};

PrimitiveSetConfig ParsePrimitiveSetConfig(const std::string& options)
{
    PrimitiveSetConfig config = static_cast<PrimitiveSetConfig>(0);
    for (auto& s : Split(options, ',')) {
        if (auto it = primitives.find(s); it != primitives.end()) {
            config |= it->second;
        } else {
            fmt::print("Unrecognized symbol {}\n", s);
            std::exit(1);
        }
    }
    return config;
}
}

#endif
