// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <fmt/format.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"

namespace Operon {

auto PostfixFormatter::FormatNode(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, size_t i, std::string& current, int decimalPrecision) -> void
{
    auto const& s = tree[i];

    switch(s.Type) {
        case NodeType::Constant: {
            auto formatString = fmt::format(fmt::runtime(s.Value < 0 ? "({{:.{}f}})" : "{{:.{}f}}"), decimalPrecision);
            fmt::format_to(std::back_inserter(current), fmt::runtime(formatString), s.Value);
            break;
        }
        case NodeType::Variable: {
            auto formatString = fmt::format(fmt::runtime(s.Value < 0 ? "(({{:.{}f}}) * {{}})" : "({{:.{}f}} * {{}})"), decimalPrecision);
            if (auto it = variableNames.find(s.HashValue); it != variableNames.end()) {
                fmt::format_to(std::back_inserter(current), fmt::runtime(formatString), s.Value, it->second);
            } else {
                throw std::runtime_error(fmt::format("A variable with hash value {} could not be found in the dataset.\n", s.HashValue));
            }
            break;
        }
        default: {
            fmt::format_to(std::back_inserter(current), fmt::runtime(s.Name()));
        }
    }
}

auto PostfixFormatter::Format(Tree const& tree, Dataset const& dataset, int decimalPrecision) -> std::string
{
    Operon::Map<Operon::Hash, std::string> variableNames;
    for (auto const& var : dataset.GetVariables()) {
        variableNames.insert({ var.Hash, var.Name });
    }
    return Format(tree, variableNames, decimalPrecision);
}

auto PostfixFormatter::Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision) -> std::string
{
    std::string result;
    for (auto i = 0UL; i < tree.Length(); ++i) {
        if (static_cast<int>(i) == tree[i].Parent - tree[tree[i].Parent].Length) {
            fmt::format_to(std::back_inserter(result), "(");
        }
        FormatNode(tree, variableNames, i, result, decimalPrecision);
        if (!tree[i].IsLeaf()) {
            fmt::format_to(std::back_inserter(result), ")");
        }
        fmt::format_to(std::back_inserter(result), " ");
    }
    return { result.begin(), result.end() };
}
} // namespace Operon
