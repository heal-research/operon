#ifndef CLI_UTIL_HPP
#define CLI_UTIL_HPP

#include <charconv>
#include <chrono>
#include <fmt/core.h>
#include <sstream>
#include <string>

#include "core/dataset.hpp"
#include "core/grammar.hpp"

namespace Operon {
// parse a range specified as stard:end
Range ParseRange(const std::string& range)
{
    auto pos = range.find_first_of(':');
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
};

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

GrammarConfig ParseGrammarConfig(const std::string& options)
{
    GrammarConfig config = static_cast<GrammarConfig>(0);
    for (auto& s : Split(options, ',')) {
        if (s == "add") {
            config |= GrammarConfig::Add;
        } else if (s == "sub") {
            config |= GrammarConfig::Sub;
        } else if (s == "mul") {
            config |= GrammarConfig::Mul;
        } else if (s == "div") {
            config |= GrammarConfig::Div;
        } else if (s == "exp") {
            config |= GrammarConfig::Exp;
        } else if (s == "log") {
            config |= GrammarConfig::Log;
        } else if (s == "sin") {
            config |= GrammarConfig::Sin;
        } else if (s == "cos") {
            config |= GrammarConfig::Cos;
        } else if (s == "tan") {
            config |= GrammarConfig::Tan;
        } else if (s == "sqrt") {
            config |= GrammarConfig::Sqrt;
        } else if (s == "cbrt") {
            config |= GrammarConfig::Cbrt;
        } else if (s == "square") {
            config |= GrammarConfig::Square;
        } else {
            fmt::print("Unrecognized symbol {}\n", s);
            std::exit(1);
        }
    }
    return config;
}
}

#endif
