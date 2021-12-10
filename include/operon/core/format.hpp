// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef FORMATTERS_HPP
#define FORMATTERS_HPP

#include "dataset.hpp"
#include "tree.hpp"

namespace Operon {
class TreeFormatter {
    static void FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> variableNames, size_t i, std::string& current, std::string indent, bool isLast, bool initialMarker, int decimalPrecision);

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
    static void FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, size_t i, fmt::memory_buffer& current, int decimalPrecision);

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
