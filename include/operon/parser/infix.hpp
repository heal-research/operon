// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_PARSER_HPP
#define OPERON_PARSER_HPP

#include <infix-parser/parser.hpp>
#include <tl/expected.hpp>
#include <span>
#include <string>
#include <string_view>
#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon {
struct InfixParseError {
    std::string Message;
};

struct InfixParseOptions {
    bool Reduce{false};
    // Fold constant * variable products into Operon variable weights.
    // Disabled by default; function and compound-subtree weights remain explicit.
    bool FoldVariableWeights{false};
};

struct OPERON_EXPORT InfixParser {
    static auto TryParse(std::string_view infix, InfixParseOptions options = {}) -> tl::expected<Tree, InfixParseError>;
    static auto TryParse(std::string_view infix, Dataset const& dataset, InfixParseOptions options = {}) -> tl::expected<Tree, InfixParseError>;
    static auto Parse(std::string_view infix, InfixParseOptions options = {}) -> Tree;
    static auto Parse(std::string_view infix, Dataset const& dataset, InfixParseOptions options = {}) -> Tree;
    static auto ParseFunctionBody(std::string_view infix, std::span<std::string const> params,
                                  InfixParseOptions options = {}) -> Tree;
};
} // namespace Operon

#endif
