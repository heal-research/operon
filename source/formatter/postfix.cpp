// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"

namespace Operon {

auto PostfixFormatter::FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, size_t i, fmt::memory_buffer& current, int decimalPrecision) -> void
{
    auto const& s = tree[i];

    switch(s.Type) {
        case NodeType::Constant: {
            auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}})" : "{{:.{}f}}", decimalPrecision);
            fmt::format_to(std::back_inserter(current), formatString, s.Value);
            break;
        }
        case NodeType::Variable: {
            auto formatString = fmt::format(s.Value < 0 ? "(({{:.{}f}}) * {{}})" : "({{:.{}f}} * {{}})", decimalPrecision);
            if (auto it = variableNames.find(s.HashValue); it != variableNames.end()) {
                fmt::format_to(std::back_inserter(current), formatString, s.Value, it->second);
            } else {
                throw std::runtime_error(fmt::format("A variable with hash value {} could not be found in the dataset.\n", s.HashValue));
            }
        }
        default: {
            fmt::format_to(current, s.Name());
        }
    }
}

auto PostfixFormatter::Format(Tree const& tree, Dataset const& dataset, int decimalPrecision) -> std::string
{
    std::unordered_map<Operon::Hash, std::string> variableNames;
    for (auto const& var : dataset.Variables()) {
        variableNames.insert({ var.Hash, var.Name });
    }
    return Format(tree, variableNames, decimalPrecision);
}

auto PostfixFormatter::Format(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, int decimalPrecision) -> std::string
{
    fmt::memory_buffer result;
    for (auto i = 0UL; i < tree.Length(); ++i) {
        if (i == tree[i].Parent - tree[tree[i].Parent].Length) {
            fmt::format_to(result, "(");
        }
        FormatNode(tree, variableNames, i, result, decimalPrecision);
        if (!tree[i].IsLeaf()) {
            fmt::format_to(result, ")");
        }
        fmt::format_to(result, " ");
    }
    return { result.begin(), result.end() };
}
} // namespace Operon
