/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
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
            auto formatString = fmt::format("{{:.{}f}}\n", decimalPrecision);
            fmt::format_to(std::back_inserter(current), formatString, s.Value);
        } else if (s.IsVariable()) {
            auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}}) * {{}}\n" : "{{:.{}f}} * {{}}\n", decimalPrecision);
            fmt::format_to(std::back_inserter(current), formatString, s.Value, dataset.GetName(s.CalculatedHashValue));
        } else {
            fmt::format_to(std::back_inserter(current), "{} {} {}\n", s.Name(), s.Depth, s.Length+1);
        }

        if (s.IsLeaf()) {
            return;
        }

        if (i != tree.Length() - 1) {
            indent += isLast ? "    " : "|   ";
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
            auto formatString = fmt::format(s.Value < 0 ? "({{:.{}f}}) * {{}}" : "{{:.{}f}} * {{}}", decimalPrecision);
            fmt::format_to(std::back_inserter(current), formatString, s.Value, dataset.GetName(s.CalculatedHashValue));
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
