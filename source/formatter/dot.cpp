// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <fmt/format.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"

namespace Operon {

auto DotFormatter::Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision) -> std::string
{
    std::string result;
    result += "strict digraph {\n";
    result += "\trankdir=BT\n";

    auto formatLeaf = [&](auto const& s) {
        if (s.IsConstant()) {
            auto formatString = fmt::format(fmt::runtime("{{:.{}f}}"), decimalPrecision);
            return fmt::format(fmt::runtime(formatString), s.Value);
        }
        if (s.IsVariable()) {
            auto formatString = fmt::format(fmt::runtime("({{:.{}f}} * {{}})"), decimalPrecision);
            if (auto it = variableNames.find(s.HashValue); it != variableNames.end()) {
                return fmt::format(fmt::runtime(formatString), s.Value, it->second);
            }
            throw std::runtime_error(fmt::format("A key with hash value {} could not be found in the variable map.\n", s.HashValue));
        }
        throw std::runtime_error("node is not a leaf (constant or variable)");
    };

    auto format = [&](auto const& s) {
        if (s.IsLeaf()) { return formatLeaf(s); }
        return s.Name();
    };

    for (auto i = 0UL; i < tree.Length(); ++i) {
        auto label = format(tree[i]);
        fmt::format_to(std::back_inserter(result), "\t{} [label=\"{}\"]\n", i, label);

        if (tree[i].IsLeaf()) { continue; }

        for (auto j : tree.Indices(i)) {
            fmt::format_to(std::back_inserter(result), "\t{} -> {}\n", j, i);
        }
    }

    result += "}\n";
    return result;
}

auto DotFormatter::Format(Tree const& tree, Dataset const& dataset, int decimalPrecision) -> std::string
{
    Operon::Map<Operon::Hash, std::string> variableNames;
    for (auto const& var : dataset.GetVariables()) {
        variableNames.insert({ var.Hash, var.Name });
    }
    return DotFormatter::Format(tree, variableNames, decimalPrecision);
}

} // namespace Operon
