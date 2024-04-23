// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <fmt/format.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"

namespace Operon {

auto TreeFormatter::FormatNode(Tree const& tree, Operon::Map<Operon::Hash, std::string> variableNames, size_t i, std::string& current, std::string indent, bool isLast, bool initialMarker, int decimalPrecision) -> void
{
    std::string const last{"└── "};
    std::string const notLast{"├── "};

    current += indent;

    if (initialMarker) {
        current += isLast ? last : notLast;
    }

    const auto& s = tree[i];
    if (s.IsConstant()) {
        auto formatString = fmt::format("{{:.{}f}}", decimalPrecision);
        fmt::format_to(std::back_inserter(current), fmt::runtime(formatString), s.Value);
    } else if (s.IsVariable()) {
        auto formatString = fmt::format(fmt::runtime(s.Value < 0 ? "({{:.{}f}}) * {{}}" : "{{:.{}f}} * {{}}"), decimalPrecision);

        if (auto it = variableNames.find(s.HashValue); it != variableNames.end()) {
            fmt::format_to(std::back_inserter(current), fmt::runtime(formatString), s.Value, it->second);
        } else {
            throw std::runtime_error(fmt::format("A variable with hash value {} could not be found in the dataset.\n", s.HashValue));
        }
    } else {
        fmt::format_to(std::back_inserter(current), "{}", s.Name());
    }
    fmt::format_to(std::back_inserter(current), " D:{} L:{} N:{}\n", s.Depth, s.Level, s.Length + 1);

    if (s.IsLeaf()) {
        return;
    }

    if (i != tree.Length() - 1) {
        indent += isLast ? "    " : "│   ";
    }

    size_t count = 0;
    for (auto j : tree.Indices(i)) {
        FormatNode(tree, variableNames, j, current, indent, ++count == s.Arity, /*initialMarker=*/true, decimalPrecision);
    }
}

auto TreeFormatter::Format(Tree const& tree, Dataset const& dataset, int decimalPrecision) -> std::string
{
    Operon::Map<Operon::Hash, std::string> variableNames;
    for (auto const& var : dataset.GetVariables()) {
        variableNames.insert({ var.Hash, var.Name });
    }

    std::string result;
    FormatNode(tree, variableNames, tree.Length() - 1, result, "", /*isLast=*/true, /*initialMarker=*/false, decimalPrecision);
    return result;
}

auto TreeFormatter::Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision) -> std::string
{
    std::string result;
    FormatNode(tree, variableNames, tree.Length() - 1, result, "", /*isLast=*/true, /*initialMarker=*/false, decimalPrecision);
    return result;
}

} // namespace Operon
