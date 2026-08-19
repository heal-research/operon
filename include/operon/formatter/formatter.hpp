// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_FORMAT_HPP
#define OPERON_FORMAT_HPP

#include <unordered_map>

#include "operon/core/tree.hpp"

namespace Operon {

class Dataset;

class OPERON_EXPORT TreeFormatter {
    static auto FormatNode(Tree const& tree, Operon::Map<Operon::Hash, std::string> variableNames, size_t i, std::string& current, std::string indent, bool isLast, bool initialMarker, int decimalPrecision) -> void;

public:
    static auto Format(Tree const& tree, Dataset const& dataset, int decimalPrecision = 2) -> std::string;
    static auto Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision = 2) -> std::string;
};

class OPERON_EXPORT InfixFormatter {
    static auto FormatNode(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, size_t i, std::string& current, int decimalPrecision) -> void;

public:
    // decimalPrecision is significant digits (general/%g-style formatting,
    // fixed or scientific chosen automatically), not decimal places -- pass
    // std::numeric_limits<Operon::Scalar>::max_digits10 for output that must
    // round-trip exactly back through InfixParser (e.g. any model printed
    // for a caller to re-parse). A smaller value trades round-trip fidelity
    // for brevity; coefficients below the chosen precision's threshold can
    // print as 0 and silently change the model's behavior when re-parsed.
    static auto Format(Tree const& tree, Dataset const& dataset, int decimalPrecision = 2) -> std::string;
    static auto Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision = 2) -> std::string;
};

class OPERON_EXPORT PostfixFormatter {
    static auto FormatNode(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, size_t i, std::string& current, int decimalPrecision) -> void;

public:
    static auto Format(Tree const& tree, Dataset const& dataset, int decimalPrecision = 2) -> std::string;
    static auto Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision = 2) -> std::string;
};

struct OPERON_EXPORT DotFormatter {
    static auto Format(Tree const& tree, Dataset const& dataset, int decimalPrecision = 2) -> std::string;
    static auto Format(Tree const& tree, Operon::Map<Operon::Hash, std::string> const& variableNames, int decimalPrecision = 2) -> std::string;
};
} // namespace Operon

#endif
