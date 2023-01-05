// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_PARSER_HPP
#define OPERON_PARSER_HPP

#include <pratt-parser/token.hpp>
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/hash.hpp"

namespace Operon {

namespace detail {
    using TokenKind = pratt::token_kind;
    using Token = pratt::token<Operon::Vector<Node>>;
    using TokenMap = Operon::Map<std::string, detail::Token, Operon::Hasher, std::equal_to<>>;
    using VariableMap = Operon::Map<std::string, Operon::Hash>;
} // namespace detail

struct OPERON_EXPORT InfixParser {
    using Token = detail::Token;
    using TokenKind = detail::TokenKind;

static auto Parse(std::string const& infix, detail::VariableMap const& vars, detail::TokenMap const& tokens, bool reduce = false) -> Tree;
static auto Parse(std::string const& infix, detail::VariableMap const& vars, bool reduce = false) -> Tree;
};
} // namespace Operon

#endif
