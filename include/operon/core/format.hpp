// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef FORMATTERS_HPP
#define FORMATTERS_HPP

#include "dataset.hpp"
#include "tree.hpp"

namespace Operon {
class TreeFormatter {
    static void FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> variableNames, size_t i, std::string& current, std::string indent, bool isLast, bool initialMarker, int decimalPrecision)
    {
        current += indent;

        if (initialMarker) {
            current += isLast ? "└── " : "├── ";
        }

        auto& s = tree[i];
        if (s.IsConstant()) {
            auto formatString = fmt::format("{{:.{}f}}", decimalPrecision);
            fmt::format_to(std::back_inserter(current), formatString, s.Value);
        } else if (s.IsVariable()) {
            auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}}) * {{}}" : "{{:.{}f}} * {{}}", decimalPrecision);

            if(auto it = variableNames.find(s.CalculatedHashValue); it != variableNames.end()) {
                fmt::format_to(std::back_inserter(current), formatString, s.Value, it->second);
            } else {
                throw std::runtime_error(fmt::format("A variable with hash value {} could not be found in the dataset.\n", s.CalculatedHashValue));
            }
        } else {
            fmt::format_to(std::back_inserter(current), "{}", s.Name());
        }
        fmt::format_to(std::back_inserter(current), " D:{} L:{} N:{}\n", s.Depth, s.Level, s.Length+1);

        if (s.IsLeaf()) {
            return;
        }

        if (i != tree.Length() - 1) {
            indent += isLast ? "    " : "│   ";
        }

        for (auto it = tree.Children(i); it.HasNext(); ++it) {
            FormatNode(tree, variableNames, it.Index(), current, indent, it.Count() + 1 == s.Arity, true, decimalPrecision);
        }
    }

public:
    static std::string Format(Tree const& tree, Dataset const& dataset, int decimalPrecision = 2)
    {
        std::unordered_map<Operon::Hash, std::string> variableNames;
        for (auto const& var : dataset.Variables()) {
            variableNames.insert({ var.Hash, var.Name });
        }

        std::string result;
        FormatNode(tree, variableNames, tree.Length() - 1, result, "", true, false, decimalPrecision);
        return result;
    }

    static std::string Format(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, int decimalPrecision = 2)
    {
        std::string result;
        FormatNode(tree, variableNames, tree.Length() - 1, result, "", true, false, decimalPrecision);
        return result;
    }
};

class InfixFormatter {
    static void FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, size_t i, fmt::memory_buffer & current, int decimalPrecision)
    {
        auto& s = tree[i];
        if (s.IsConstant()) {
            auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}})" : "{{:.{}f}}", decimalPrecision);
            fmt::format_to(current, formatString, s.Value);
        } else if (s.IsVariable()) {
            auto formatString = fmt::format(s.Value < 0 ? "(({{:.{}f}}) * {{}})" : "({{:.{}f}} * {{}})", decimalPrecision);
            if(auto it = variableNames.find(s.CalculatedHashValue); it != variableNames.end()) {
                fmt::format_to(current, formatString, s.Value, it->second);
            } else {
                throw std::runtime_error(fmt::format("A variable with hash value {} could not be found in the dataset.\n", s.CalculatedHashValue));
            }
        } else {
            if (s.Type < NodeType::Log) // add, sub, mul, div, aq, pow
            {
                if (s.Arity == 1) {
                    if (s.Type == NodeType::Sub) {
                        // subtraction with a single argument is a negation -x
                        fmt::format_to(current, "-");
                    } else if (s.Type == NodeType::Div) {
                        // division with a single argument is an inversion 1/x
                        fmt::format_to(current, "1 / ");
                    }
                }
                fmt::format_to(current, "(");
                if (s.Type == NodeType::Pow) {
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
                    for (auto it = tree.Children(i); it.HasNext(); ++it) {
                        FormatNode(tree, variableNames, it.Index(), current, decimalPrecision);
                        if (it.Count() + 1 < s.Arity) {
                            fmt::format_to(current, " {} ", s.Name());
                        }
                    }
                }
                fmt::format_to(current, ")");
            } else { // unary operators log, exp, sin, etc.
                if (s.IsSquare()) {
                    // format square(a)  as a ^ 2
                    fmt::format_to(current, "(");
                    FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                    fmt::format_to(current, " ^ 2)");
                } else {
                    fmt::format_to(current, "{}", s.Name());
                    fmt::format_to(current, "(");
                    FormatNode(tree, variableNames, i - 1, current, decimalPrecision);
                    fmt::format_to(current, ")");
                }
            }
        }
    }

public:
    static std::string Format(Tree const& tree, Dataset const& dataset, int decimalPrecision = 2)
    {
        std::unordered_map<Operon::Hash, std::string> variableNames;
        for (auto const& var : dataset.Variables()) {
            variableNames.insert({ var.Hash, var.Name });
        }
        fmt::memory_buffer result;
        FormatNode(tree, variableNames, tree.Length() - 1, result, decimalPrecision);
        return std::string(result.begin(), result.end());
    }

    static std::string Format(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, int decimalPrecision = 2)
    {
        fmt::memory_buffer result;
        FormatNode(tree, variableNames, tree.Length() - 1, result, decimalPrecision);
        return std::string(result.begin(), result.end());
    }
};
}

#endif
