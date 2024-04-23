// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <ankerl/unordered_dense.h>
#include <cstddef>
#include <fmt/core.h>
#include <functional>
#include <iterator>
#include <pratt-parser/parser.hpp>
#include <stdexcept>
#include <string>
#include <vector>

#include "operon/hash/hash.hpp"
#include "operon/parser/infix.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon {
namespace detail {
    struct Conv {
        auto operator()(double val) const noexcept -> Token::value_t
        {
            Node node(NodeType::Constant);
            node.Value = static_cast<Operon::Scalar>(val);
            return Token::value_t { node };
        }
    };

    struct Nud {
        using token_t = Token; // NOLINT
        using value_t = Token::value_t; // NOLINT

        template <typename Parser>
        auto operator()(Parser& parser, token_t const& tok, token_t const& left) const -> value_t
        {
            if (tok.kind() == TokenKind::constant) {
                return left.value();
            }

            if (tok.kind() == TokenKind::variable) {
                auto hash = parser.template get_desc<Operon::Hash>(left.name());
                if (!hash.has_value()) {
                    throw std::invalid_argument(fmt::format("unknown variable name {}\n", left.name()));
                }
                return value_t { Node(NodeType::Variable, hash.value()) };
            }

            auto bp = tok.precedence(); // binding power
            if (tok.kind() == TokenKind::lparen) {
                return parser.parse_bp(bp, TokenKind::rparen).value();
            }
            auto result = std::move(parser.parse_bp(bp, TokenKind::eof).value());

            if (tok.kind() == TokenKind::dynamic) {
                Node n(static_cast<NodeType>(tok.opcode()));
                if (n.Is<NodeType::Asin, NodeType::Acos, NodeType::Atan,
                         NodeType::Sin,  NodeType::Cos,  NodeType::Tan,
                         NodeType::Sinh, NodeType::Cosh, NodeType::Tanh,
                         NodeType::Sub,  NodeType::Exp,  NodeType::Log,
                         NodeType::Sqrt, NodeType::Cbrt, NodeType::Square,
                         NodeType::Abs,  NodeType::Ceil, NodeType::Floor>()) {
                    result.push_back(n);
                    result.back().Arity = 1;
                } else {
                    throw std::runtime_error(fmt::format("nud: unsupported token {}\n", tok.name()));
                }
            }

            return result;
        }
    };

    struct Led {
        using token_t = Token;          // NOLINT
        using value_t = Token::value_t; // NOLINT

        template <typename Parser>
        auto operator()(Parser& /*unused*/, Token const& tok, token_t const& left, token_t& right) const -> value_t
        {
            auto const& lhs = left.value();
            auto& rhs = right.value();

            rhs.reserve(lhs.size() + rhs.size() + 1);
            std::copy(lhs.begin(), lhs.end(), std::back_inserter(rhs));

            ENSURE(tok.kind() == TokenKind::dynamic);
            Node n(static_cast<NodeType>(tok.opcode()));
            if (n.Is<NodeType::Add, NodeType::Sub, NodeType::Mul, NodeType::Div, NodeType::Aq, NodeType::Pow, NodeType::Fmin, NodeType::Fmax>()) {
                rhs.push_back(n);
            } else {
                throw std::runtime_error(fmt::format("led: unsupported token ", tok.name()));
            }

            return rhs;
        }
    };

    static auto DefaultTokens() {
        return Operon::Map<std::string, Token, Operon::Hasher, std::equal_to<>>{
            // NOLINTBEGIN
            { "+", Token(TokenKind::dynamic, "add", static_cast<size_t>(NodeType::Add), 10, pratt::associativity::left) },
            { "-", Token(TokenKind::dynamic, "sub", static_cast<size_t>(NodeType::Sub), 10, pratt::associativity::left) },
            { "*", Token(TokenKind::dynamic, "mul", static_cast<size_t>(NodeType::Mul), 20, pratt::associativity::left) },
            { "/", Token(TokenKind::dynamic, "div", static_cast<size_t>(NodeType::Div), 20, pratt::associativity::left) },
            { "^", Token(TokenKind::dynamic, "pow", static_cast<size_t>(NodeType::Pow), 30, pratt::associativity::right) },
            { "pow", Token(TokenKind::dynamic, "pow", static_cast<size_t>(NodeType::Pow), 30, pratt::associativity::right) },
            { "min", Token(TokenKind::dynamic, "min", static_cast<size_t>(NodeType::Fmin), 30, pratt::associativity::left) },
            { "max", Token(TokenKind::dynamic, "max", static_cast<size_t>(NodeType::Fmax), 30, pratt::associativity::left) },
            { "abs", Token(TokenKind::dynamic, "abs", static_cast<size_t>(NodeType::Abs), 30, pratt::associativity::none) },
            { "ceil", Token(TokenKind::dynamic, "ceil", static_cast<size_t>(NodeType::Ceil), 30, pratt::associativity::none) },
            { "floor", Token(TokenKind::dynamic, "floor", static_cast<size_t>(NodeType::Floor), 30, pratt::associativity::none) },
            { "cbrt", Token(TokenKind::dynamic, "cbrt", static_cast<size_t>(NodeType::Cbrt), 30, pratt::associativity::none) },
            { "acos", Token(TokenKind::dynamic, "acos", static_cast<size_t>(NodeType::Acos), 30, pratt::associativity::none) },
            { "cos", Token(TokenKind::dynamic, "cos", static_cast<size_t>(NodeType::Cos), 30, pratt::associativity::none) },
            { "cosh", Token(TokenKind::dynamic, "cosh", static_cast<size_t>(NodeType::Cosh), 30, pratt::associativity::none) },
            { "exp", Token(TokenKind::dynamic, "exp", static_cast<size_t>(NodeType::Exp), 30, pratt::associativity::none) },
            { "log", Token(TokenKind::dynamic, "log", static_cast<size_t>(NodeType::Log), 30, pratt::associativity::none) },
            { "asin", Token(TokenKind::dynamic, "asin", static_cast<size_t>(NodeType::Asin), 30, pratt::associativity::none) },
            { "sin", Token(TokenKind::dynamic, "sin", static_cast<size_t>(NodeType::Sin), 30, pratt::associativity::none) },
            { "sinh", Token(TokenKind::dynamic, "sinh", static_cast<size_t>(NodeType::Sinh), 30, pratt::associativity::none) },
            { "sqrt", Token(TokenKind::dynamic, "sqrt", static_cast<size_t>(NodeType::Sqrt), 30, pratt::associativity::none) },
            { "square", Token(TokenKind::dynamic, "square", static_cast<size_t>(NodeType::Square), 30, pratt::associativity::right) },
            { "atan", Token(TokenKind::dynamic, "atan", static_cast<size_t>(NodeType::Atan), 30, pratt::associativity::none) },
            { "tan", Token(TokenKind::dynamic, "tan", static_cast<size_t>(NodeType::Tan), 30, pratt::associativity::none) },
            { "tanh", Token(TokenKind::dynamic, "tanh", static_cast<size_t>(NodeType::Tanh), 30, pratt::associativity::none) },
            { "(", Token(pratt::token_kind::lparen, "(", 0UL /* don't care */, 0, pratt::associativity::none) },
            { ")", Token(pratt::token_kind::rparen, ")", 0UL /* don't care */, 0, pratt::associativity::none) },
            { "eof", Token(pratt::token_kind::eof, "eof", 0UL /* don't care */, 0, pratt::associativity::none) }
             // NOLINTEND
        };
    }

    using Parser = pratt::parser<Nud, Led, Conv, TokenMap, VariableMap>;
} // namespace detail

auto InfixParser::Parse(std::string const& infix, detail::VariableMap const& vars, detail::TokenMap const& toks, bool reduce) -> Tree
{
    auto nodes = detail::Parser(infix, toks, vars).parse();
    auto tree = Tree(nodes).UpdateNodes();
    return reduce ? tree.Reduce() : tree;
}

auto InfixParser::Parse(std::string const& infix, detail::VariableMap const& vars, bool reduce) -> Tree
{
    auto tokens = detail::DefaultTokens();
    return Parse(infix, vars, tokens, reduce);
}

} // namespace Operon
