// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/format.hpp"
#include "operon/core/dataset.hpp"

namespace Operon {

void TreeFormatter::FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> variableNames, size_t i, std::string& current, std::string indent, bool isLast, bool initialMarker, int decimalPrecision)
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
        fmt::format_to(std::back_inserter(current), formatString, s.Value);
    } else if (s.IsVariable()) {
        auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}}) * {{}}" : "{{:.{}f}} * {{}}", decimalPrecision);

        if (auto it = variableNames.find(s.CalculatedHashValue); it != variableNames.end()) {
            fmt::format_to(std::back_inserter(current), formatString, s.Value, it->second);
        } else {
            throw std::runtime_error(fmt::format("A variable with hash value {} could not be found in the dataset.\n", s.CalculatedHashValue));
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
    for (auto it = tree.Children(i); it.HasNext(); ++it) {
        FormatNode(tree, variableNames, it.Index(), current, indent, ++count == s.Arity, /*initialMarker=*/true, decimalPrecision);
    }
}

auto TreeFormatter::Format(Tree const& tree, Dataset const& dataset, int decimalPrecision) -> std::string
{
    std::unordered_map<Operon::Hash, std::string> variableNames;
    for (auto const& var : dataset.Variables()) {
        variableNames.insert({ var.Hash, var.Name });
    }

    std::string result;
    FormatNode(tree, variableNames, tree.Length() - 1, result, "", /*isLast=*/true, /*initialMarker=*/false, decimalPrecision);
    return result;
}

auto TreeFormatter::Format(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, int decimalPrecision) -> std::string
{
    std::string result;
    FormatNode(tree, variableNames, tree.Length() - 1, result, "", /*isLast=*/true, /*initialMarker=*/false, decimalPrecision);
    return result;
}

void InfixFormatter::FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, size_t i, fmt::memory_buffer& current, int decimalPrecision)
{
    const auto& s = tree[i];
    if (s.IsConstant()) {
        auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}})" : "{{:.{}f}}", decimalPrecision);
        fmt::format_to(current, formatString, s.Value);
    } else if (s.IsVariable()) {
        auto formatString = fmt::format(s.Value < 0 ? "(({{:.{}f}}) * {{}})" : "({{:.{}f}} * {{}})", decimalPrecision);
        if (auto it = variableNames.find(s.CalculatedHashValue); it != variableNames.end()) {
            fmt::format_to(current, formatString, s.Value, it->second);
        } else {
            throw std::runtime_error(fmt::format("A variable with hash value {} could not be found in the dataset.\n", s.CalculatedHashValue));
        }
    } else {
        if (s.Type < NodeType::Abs) // add, sub, mul, div, aq, fmax, fmin, pow
        {
            fmt::format_to(current, "(");
            if (s.Arity == 1) {
                if (s.Type == NodeType::Sub) {
                    // subtraction with a single argument is a negation -x
                    fmt::format_to(current, "-");
                } else if (s.Type == NodeType::Div) {
                    // division with a single argument is an inversion 1/x
                    fmt::format_to(current, "1 / ");
                }
            } else if (s.Type == NodeType::Pow) {
                // format pow(a,b) as a^b
                auto j = i - 1;
                auto k = j - tree[j].Length - 1;
                FormatNode(tree, variableNames, j, current, decimalPrecision);
                fmt::format_to(current, " ^ ");
                FormatNode(tree, variableNames, k, current, decimalPrecision);
            } else if (s.Type == NodeType::Aq) {
                // format aq(a,b) as a / (1 + b^2)
                auto j = i - 1;
                auto k = j - tree[j].Length - 1;
                FormatNode(tree, variableNames, j, current, decimalPrecision);
                fmt::format_to(current, " / (sqrt(1 + ");
                FormatNode(tree, variableNames, k, current, decimalPrecision);
                fmt::format_to(current, " ^ 2))");
            } else {
                size_t count = 0;
                for (auto it = tree.Children(i); it.HasNext(); ++it) {
                    FormatNode(tree, variableNames, it.Index(), current, decimalPrecision);
                    if (++count < s.Arity) {
                        fmt::format_to(current, " {} ", s.Name());
                    }
                }
            }
            fmt::format_to(current, ")");
        } else { // unary operators abs, asin, ... log, exp, sin, etc.
            if (s.Type == NodeType::Square) {
                // format square(a) as a ^ 2
                fmt::format_to(current, "(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(current, " ^ 2)");
            } else if (s.Type == NodeType::Logabs) {
                // format logabs(a) as log(abs(a))
                fmt::format_to(current, "log(abs(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(current, "))");
            } else if (s.Type == NodeType::Log1p) {
                // format log1p(a) as log(a+1)
                fmt::format_to(current, "log(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(current, "+1)");
            } else if (s.Type == NodeType::Sqrtabs) {
                // format sqrtabs(a) as sqrt(abs(a))
                fmt::format_to(current, "sqrt(abs(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(current, "))");
            } else {
                fmt::format_to(current, "{}", s.Name());
                fmt::format_to(current, "(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(current, ")");
            }
        }
    }
}

auto InfixFormatter::Format(Tree const& tree, Dataset const& dataset, int decimalPrecision) -> std::string
{
    std::unordered_map<Operon::Hash, std::string> variableNames;
    for (auto const& var : dataset.Variables()) {
        variableNames.insert({ var.Hash, var.Name });
    }
    fmt::memory_buffer result;
    FormatNode(tree, variableNames, tree.Length() - 1, result, decimalPrecision);
    return { result.begin(), result.end() };
}

auto InfixFormatter::Format(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, int decimalPrecision) -> std::string
{
    fmt::memory_buffer result;
    FormatNode(tree, variableNames, tree.Length() - 1, result, decimalPrecision);
    return { result.begin(), result.end() };
}

} // namespace Operon

