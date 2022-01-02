// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_FORMAT_HPP
#define OPERON_FORMAT_HPP

#include "tree.hpp"

namespace Operon {

class Dataset;

class OPERON_EXPORT TreeFormatter {
    static void FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> variableNames, size_t i, std::string& current, std::string indent, bool isLast, bool initialMarker, int decimalPrecision);

public:
    static auto Format(Tree const& tree, Dataset const& dataset, int decimalPrecision = 2) -> std::string;

    static auto Format(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, int decimalPrecision = 2) -> std::string;
};

class OPERON_EXPORT InfixFormatter {
    static void FormatNode(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, size_t i, fmt::memory_buffer& current, int decimalPrecision);

public:
    static auto Format(Tree const& tree, Dataset const& dataset, int decimalPrecision = 2) -> std::string;

    static auto Format(Tree const& tree, std::unordered_map<Operon::Hash, std::string> const& variableNames, int decimalPrecision = 2) -> std::string;
};
} // namespace Operon

#endif

