// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_PARSER_HPP
#define OPERON_PARSER_HPP

#include <infix-parser/parser.hpp>
#include <span>
#include <string>
#include <string_view>
#include "operon/core/dataset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon {

struct OPERON_EXPORT InfixParser {
    static auto Parse(std::string_view infix, bool reduce = false) -> Tree;
    static auto Parse(std::string_view infix, Dataset const& dataset, bool reduce = false) -> Tree;

    // Parses a composed-function body: bare identifiers matching `params`
    // become formal-parameter leaves (see Node.hpp's ParamHash), any other
    // bare identifier throws (no dataset to resolve it against), and every
    // parameter must be referenced at least once. v1 only recognizes
    // built-in function names in the body (the fixed lexy grammar) — no
    // recursive composition. `params.size()` must not exceed
    // kMaxComposedFunctionArity.
    static auto ParseFunctionBody(std::string_view infix, std::span<std::string const> params) -> Tree;
};
} // namespace Operon

#endif
