// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "util.hpp"

#include <memory>
#include <scn/scn.h>

#include "operon/operators/selector.hpp"
#include "operon/core/node.hpp"
#include "operon/core/pset.hpp"

using Operon::NodeType;

namespace Operon {

static const std::unordered_map<std::string, NodeType> Primitives {
    { "add",      NodeType::Add },
    { "mul",      NodeType::Mul },
    { "sub",      NodeType::Sub },
    { "div",      NodeType::Div },
    { "aq",       NodeType::Aq },
    { "pow",      NodeType::Pow },
    { "cbrt",     NodeType::Cbrt },
    { "cos",      NodeType::Cos },
    { "exp",      NodeType::Exp },
    { "log",      NodeType::Log },
    { "sin",      NodeType::Sin },
    { "sqrt",     NodeType::Sqrt },
    { "square",   NodeType::Square },
    { "tan",      NodeType::Tan },
    { "tanh",     NodeType::Tanh },
    { "dyn",      NodeType::Dynamic },
    { "constant", NodeType::Constant },
    { "variable", NodeType::Variable }
};

auto Split(const std::string& s, char delimiter) -> std::vector<std::string>
{
    std::vector<std::string> tokens;
    std::string token;
    std::istringstream tokenStream(s);
    while (std::getline(tokenStream, token, delimiter)) {
        tokens.push_back(token);
    }
    return tokens;
}

// splits a string into substrings separated by delimiter
// formats a duration as dd:hh:mm:ss.ms
auto FormatDuration(std::chrono::duration<double> d) -> std::string
{
    auto h = std::chrono::duration_cast<std::chrono::hours>(d);
    auto m = std::chrono::duration_cast<std::chrono::minutes>(d - h);
    auto s = std::chrono::duration_cast<std::chrono::seconds>(d - h - m);
    auto l = std::chrono::duration_cast<std::chrono::milliseconds>(d - h - m - s);
    return fmt::format("{:#02d}:{:#02d}:{:#02d}.{:#03d}", h.count(), m.count(), s.count(), l.count());
}

auto FormatBytes(size_t bytes) -> std::string
{
    constexpr std::array<char, 6> sizes{" KMGT"};
    constexpr size_t base{1024};
    auto p = static_cast<size_t>(std::floor(std::log2(bytes) / std::log2(base)));
    return fmt::format("{:.2f} {}b", static_cast<double>(bytes) / std::pow(base, p), sizes.at(p));
}

auto ParseRange(std::string const& str) -> std::pair<size_t, size_t>
{
    size_t a{0};
    size_t b{0};
    scn::scan(str, "{}:{}", a, b);
    return std::make_pair(a, b);
}

auto ParsePrimitiveSetConfig(const std::string& options) -> PrimitiveSetConfig
{
    auto config = static_cast<PrimitiveSetConfig>(0);
    for (auto& s : Split(options, ',')) {
        if (auto it = Primitives.find(s); it != Primitives.end()) {
            config |= it->second;
        } else {
            throw std::runtime_error(fmt::format("Unrecognized symbol {}\n", s));
        }
    }
    return config;
}

auto PrintPrimitives(NodeType config) -> void
{
    PrimitiveSet tmpSet;
    tmpSet.SetConfig(config);
    fmt::print("Built-in primitives:\n");
    fmt::print("{:<8}\t{:<50}\t{:>7}\t\t{:>9}\n", "Symbol", "Description", "Enabled", "Frequency");
    for (size_t i = 0; i < Operon::NodeTypes::Count; ++i) {
        auto type = static_cast<NodeType>(1U << i);
        auto hash = Node(type).HashValue;
        auto enabled = tmpSet.Contains(hash) && tmpSet.IsEnabled(hash);
        auto freq = enabled ? tmpSet.Frequency(hash) : 0U;
        Node node(type);
        fmt::print("{:<8}\t{:<50}\t{:>7}\t\t{:>9}\n", node.Name(), node.Desc(), enabled, freq != 0U ? std::to_string(freq) : "-");
    }
}


} // namespace Operon
