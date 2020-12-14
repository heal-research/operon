/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef FORMATTERS_HPP
#define FORMATTERS_HPP

#include "dataset.hpp"
#include "tree.hpp"

namespace Operon {
class TreeFormatter {
    static void FormatNode(const Tree& tree, const Dataset& dataset, size_t i, std::string& current, std::string indent, bool isLast, bool initialMarker, int decimalPrecision)
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
            if (auto res = dataset.GetVariable(s.CalculatedHashValue); res.has_value()) {
                fmt::format_to(std::back_inserter(current), formatString, s.Value, res.value().Name);
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
            FormatNode(tree, dataset, it.Index(), current, indent, it.Count() + 1 == s.Arity, true, decimalPrecision);
        }
    }

public:
    static std::string Format(const Tree& tree, const Dataset& dataset, int decimalPrecision = 2)
    {
        std::string result;
        FormatNode(tree, dataset, tree.Length() - 1, result, "", true, false, decimalPrecision);
        return result;
    }
};

class InfixFormatter {
    static void FormatNode(const Tree& tree, const Dataset& dataset, size_t i, std::string& current, int decimalPrecision)
    {
        auto& s = tree[i];
        if (s.IsConstant()) {
            auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}})" : "{{:.{}f}}", decimalPrecision);
            fmt::format_to(std::back_inserter(current), formatString, s.Value);
        } else if (s.IsVariable()) {
            auto formatString = fmt::format(s.Value < 0 ? "(({{:.{}f}}) * {{}})" : "({{:.{}f}} * {{}})", decimalPrecision);
            if (auto res = dataset.GetVariable(s.CalculatedHashValue); res.has_value()) {
                fmt::format_to(std::back_inserter(current), formatString, s.Value, res.value().Name);
            } else {
                throw std::runtime_error(fmt::format("A variable with hash value {} could not be found in the dataset.\n", s.CalculatedHashValue));
            }
        } else {
            if (s.Type < NodeType::Log) // add, sub, mul, div
            {
                fmt::format_to(std::back_inserter(current), "(");
                for (auto it = tree.Children(i); it.HasNext(); ++it) {
                    FormatNode(tree, dataset, it.Index(), current, decimalPrecision);
                    if (it.Count() + 1 < s.Arity) {
                        fmt::format_to(std::back_inserter(current), " {} ", s.Name());
                    }
                }
                fmt::format_to(std::back_inserter(current), ")");
            } else // unary operators log, exp, sin, etc.
            {
                fmt::format_to(std::back_inserter(current), "{}", s.Name());
                fmt::format_to(std::back_inserter(current), "(");
                if (tree[i - 1].IsLeaf()) {
                    // surround a single leaf argument with parantheses
                    fmt::format_to(std::back_inserter(current), "(");
                    FormatNode(tree, dataset, i - 1, current, decimalPrecision);
                    fmt::format_to(std::back_inserter(current), ")");
                } else {
                    FormatNode(tree, dataset, i - 1, current, decimalPrecision);
                }
                fmt::format_to(std::back_inserter(current), ")");
            }
        }
    }

public:
    static std::string Format(const Tree& tree, const Dataset& dataset, int decimalPrecision = 2)
    {
        std::string result;
        FormatNode(tree, dataset, tree.Length() - 1, result, decimalPrecision);
        return result;
    }
};
}

#endif
