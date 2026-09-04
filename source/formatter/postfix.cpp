// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

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
        case NodeType::Ref: {
            // Ref has no tokens of its own -- replay the referenced
            // subtree's token range (it occupies the contiguous flat
            // range [RefTo-Length, RefTo], same as any other subtree)
            // instead of emitting a bare, non-evaluable "ref" token.
            auto const& t = tree[s.RefTo];
            for (auto j = s.RefTo - t.Length; j <= s.RefTo; ++j) {
                if (static_cast<int>(j) == tree[j].Parent - tree[tree[j].Parent].Length) {
                    fmt::format_to(std::back_inserter(current), "(");
                }
                FormatNode(tree, variableNames, j, current, decimalPrecision);
                if (!tree[j].IsLeaf()) {
                    fmt::format_to(std::back_inserter(current), ")");
                }
                if (j != s.RefTo) {
                    fmt::format_to(std::back_inserter(current), " ");
                }
            }
            break;
        }
        default: {
            // s.Name() is data, not a format string -- fmt::runtime(s.Name())
            // previously reparsed the function's own display name as a spec,
            // throwing fmt::format_error for any registered name containing
            // '{'/'}' (braces are otherwise legal in a registered name; only
            // empty names and reserved hashes are rejected).
            fmt::format_to(std::back_inserter(current), "{}", s.Name());
            if (s.Value != Operon::Scalar{1}) {
                // Unlike the leaf/variable case above (a single "(w * name)"
                // token, not real RPN), a function node already has its own
                // argument tokens preceding it in the stream, so its weight
                // is represented as genuine trailing RPN: "... name w *"
                // pushes the weight and multiplies it onto the function's
                // result, exactly mirroring what the interpreter evaluates.
                // Omitting this made postfix describe a different model
                // than the one actually evaluated (weights are real,
                // evaluated state -- see Tree::AdjustedLength()).
                auto formatString = fmt::format(fmt::runtime(s.Value < 0 ? " ({{:.{}f}}) *" : " {{:.{}f}} *"), decimalPrecision);
                fmt::format_to(std::back_inserter(current), fmt::runtime(formatString), s.Value);
            }
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
