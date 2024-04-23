// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <fmt/format.h>

#include "operon/core/dataset.hpp"
#include "operon/formatter/formatter.hpp"

namespace Operon {

auto InfixFormatter::FormatNode(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, size_t i, std::string& current, int decimalPrecision) -> void
{
    const auto& s = tree[i];
    if (s.IsConstant()) {
        auto formatString = fmt::format(fmt::runtime(s.Value < 0 ? "({{:.{}f}})" : "{{:.{}f}}"), decimalPrecision);
        fmt::format_to(std::back_inserter(current), fmt::runtime(formatString), s.Value);
    } else if (s.IsVariable()) {
        auto formatString = fmt::format(fmt::runtime(s.Value < 0 ? "(({{:.{}f}}) * {{}})" : "({{:.{}f}} * {{}})"), decimalPrecision);
        if (auto it = variableNames.find(s.HashValue); it != variableNames.end()) {
            fmt::format_to(std::back_inserter(current), fmt::runtime(formatString), s.Value, it->second);
        } else {
            throw std::runtime_error(fmt::format("A key with hash value {} could not be found in the variable map.\n", s.HashValue));
        }
    } else {
        if (s.Value != 1) {
            fmt::format_to(std::back_inserter(current), "(");
            auto formatString = fmt::format(fmt::runtime(s.Value < 0 ? "({{:.{}f}})" : "{{:.{}f}}"), decimalPrecision);
            fmt::format_to(std::back_inserter(current), fmt::runtime(formatString), s.Value);
            fmt::format_to(std::back_inserter(current), " * ");
        }
        if (s.Type < NodeType::Abs) // add, sub, mul, div, aq, fmax, fmin, pow
        {
            fmt::format_to(std::back_inserter(current), "(");
            if (s.Arity == 1) {
                if (s.Type == NodeType::Sub) {
                    // subtraction with a single argument is a negation -x
                    fmt::format_to(std::back_inserter(current), "-");
                } else if (s.Type == NodeType::Div) {
                    // division with a single argument is an inversion 1/x
                    fmt::format_to(std::back_inserter(current), "1 / ");
                }
                FormatNode(tree, variableNames, i-1, current, decimalPrecision);
            } else if (s.Type == NodeType::Pow) {
                // format pow(a,b) as a^b
                auto j = i - 1;
                auto k = j - tree[j].Length - 1;
                FormatNode(tree, variableNames, j, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), " ^ ");
                FormatNode(tree, variableNames, k, current, decimalPrecision);
            } else if (s.Type == NodeType::Aq) {
                // format aq(a,b) as a / (1 + b^2)
                auto j = i - 1;
                auto k = j - tree[j].Length - 1;
                FormatNode(tree, variableNames, j, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), " / (sqrt(1 + ");
                FormatNode(tree, variableNames, k, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), " ^ 2))");
            } else if (s.Type == NodeType::Fmin) {
                auto j = i - 1;
                auto k = j - tree[j].Length - 1;
                fmt::format_to(std::back_inserter(current), "min(");
                FormatNode(tree, variableNames, j, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), ", ");
                FormatNode(tree, variableNames, k, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), ")");
            } else if (s.Type == NodeType::Fmax) {
                auto j = i - 1;
                auto k = j - tree[j].Length - 1;
                fmt::format_to(std::back_inserter(current), "max(");
                FormatNode(tree, variableNames, j, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), ", ");
                FormatNode(tree, variableNames, k, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), ")");
            } else {
                size_t count = 0;
                for (auto j : tree.Indices(i)) {
                    FormatNode(tree, variableNames, j, current, decimalPrecision);
                    if (++count < s.Arity) {
                        fmt::format_to(std::back_inserter(current), " {} ", s.Name());
                    }
                }
            }
            fmt::format_to(std::back_inserter(current), ")");
        } else { // unary operators abs, asin, ... log, exp, sin, etc.
            if (s.Type == NodeType::Square) {
                // format square(a) as a ^ 2
                fmt::format_to(std::back_inserter(current), "(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), " ^ 2)");
            } else if (s.Type == NodeType::Logabs) {
                // format logabs(a) as log(abs(a))
                fmt::format_to(std::back_inserter(current), "log(abs(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), "))");
            } else if (s.Type == NodeType::Log1p) {
                // format log1p(a) as log(a+1)
                fmt::format_to(std::back_inserter(current), "log(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), "+1)");
            } else if (s.Type == NodeType::Sqrtabs) {
                // format sqrtabs(a) as sqrt(abs(a))
                fmt::format_to(std::back_inserter(current), "sqrt(abs(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), "))");
            } else {
                fmt::format_to(std::back_inserter(current), "{}", s.Name());
                fmt::format_to(std::back_inserter(current), "(");
                FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                fmt::format_to(std::back_inserter(current), ")");
            }
        }
        if (s.Value != 1) {
            fmt::format_to(std::back_inserter(current), ")");
        }
    }
}



auto InfixFormatter::Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision) -> std::string
{
    if (tree.Nodes().empty()) { return std::string{}; }
    std::string result;
    FormatNode(tree, variableNames, tree.Length() - 1, result, decimalPrecision);
    return { result.begin(), result.end() };
}

auto InfixFormatter::Format(Tree const& tree, Dataset const& dataset, int decimalPrecision) -> std::string
{
    Operon::Map<Operon::Hash, std::string> variableNames;
    for (auto const& var : dataset.GetVariables()) {
        variableNames.insert({ var.Hash, var.Name });
    }
    return Format(tree, variableNames, decimalPrecision);
}

} // namespace Operon