// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_PARSER_HPP
#define OPERON_PARSER_HPP

#include <stdexcept>
#include <unordered_map>

#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include <pratt-parser/parser.hpp>
#include <robin_hood.h>

namespace Operon {

namespace ParserBlocks {
    using TokenKind = pratt::token_kind;
    using Token = pratt::token<Operon::Vector<Node>>;

    struct Conv {
        auto operator()(double val) const noexcept -> Token::value_t
        {
            Node node(NodeType::Constant);
            node.Value = static_cast<Operon::Scalar>(val);
            return Token::value_t { node };
        }
    };

    struct Nud {
        using token_t = Token;
        using value_t = Token::value_t;

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
                switch (tok.opcode()) {
                case static_cast<size_t>(NodeType::Sub): {
                    result.push_back(Node(NodeType::Sub));
                    result.back().Arity = 1;
                    break;
                }
                //case static_cast<size_t>(NodeType::Abs): {
                //    result.push_back(Node(NodeType::Abs));
                //    break;
                //}
                //case static_cast<size_t>(NodeType::Acos): {
                //    result.push_back(Node(NodeType::Acos));
                //    break;
                //}
                //case static_cast<size_t>(NodeType::Asin): {
                //    result.push_back(Node(NodeType::Asin));
                //    break;
                //}
                //case static_cast<size_t>(NodeType::Atan): {
                //    result.push_back(Node(NodeType::Atan));
                //    break;
                //}
                case static_cast<size_t>(NodeType::Cbrt): {
                    result.push_back(Node(NodeType::Cbrt));
                    break;
                }
                //case static_cast<size_t>(NodeType::Ceil): {
                //    result.push_back(Node(NodeType::Ceil));
                //    break;
                //}
                case static_cast<size_t>(NodeType::Cos): {
                    result.push_back(Node(NodeType::Cos));
                    break;
                }
                //case static_cast<size_t>(NodeType::Cosh): {
                //    result.push_back(Node(NodeType::Cosh));
                //    break;
                //}
                case static_cast<size_t>(NodeType::Exp): {
                    result.push_back(Node(NodeType::Exp));
                    break;
                }
                //case static_cast<size_t>(NodeType::Floor): {
                //    result.push_back(Node(NodeType::Floor));
                //    break;
                //}
                case static_cast<size_t>(NodeType::Log): {
                    result.push_back(Node(NodeType::Log));
                    break;
                }
                //case static_cast<size_t>(NodeType::Logabs): {
                //    result.push_back(Node(NodeType::Logabs));
                //    break;
                //}
                //case static_cast<size_t>(NodeType::Log1p): {
                //    result.push_back(Node(NodeType::Log1p));
                //    break;
                //}
                case static_cast<size_t>(NodeType::Sin): {
                    result.push_back(Node(NodeType::Sin));
                    break;
                }
                //case static_cast<size_t>(NodeType::Sinh): {
                //    result.push_back(Node(NodeType::Sinh));
                //    break;
                //}
                case static_cast<size_t>(NodeType::Sqrt): {
                    result.push_back(Node(NodeType::Sqrt));
                    break;
                }
                //case static_cast<size_t>(NodeType::Sqrtabs): {
                //    result.push_back(Node(NodeType::Sqrtabs));
                //    break;
                //}
                case static_cast<size_t>(NodeType::Square): {
                    result.push_back(Node(NodeType::Square));
                    break;
                }
                case static_cast<size_t>(NodeType::Tan): {
                    result.push_back(Node(NodeType::Tan));
                    break;
                }
                case static_cast<size_t>(NodeType::Tanh): {
                    result.push_back(Node(NodeType::Tanh));
                    break;
                }
                default: {
                    throw std::runtime_error(fmt::format("nud: unsupported token {}\n", tok.name()));
                };
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
            switch (tok.opcode()) {
            case static_cast<size_t>(NodeType::Add): {
                rhs.emplace_back(NodeType::Add);
                break;
            }
            case static_cast<size_t>(NodeType::Sub): {
                rhs.emplace_back(NodeType::Sub);
                break;
            }
            case static_cast<size_t>(NodeType::Mul): {
                rhs.emplace_back(NodeType::Mul);
                break;
            }
            case static_cast<size_t>(NodeType::Div): {
                rhs.emplace_back(NodeType::Div);
                break;
            }
            case static_cast<size_t>(NodeType::Aq): {
                rhs.emplace_back(NodeType::Aq);
                break;
            }
            //case static_cast<size_t>(NodeType::Fmax): {
            //    rhs.push_back(Node(NodeType::Fmax));
            //    break;
            //}
            //case static_cast<size_t>(NodeType::Fmin): {
            //    rhs.push_back(Node(NodeType::Fmin));
            //    break;
            //}
            case static_cast<size_t>(NodeType::Pow): {
                rhs.emplace_back(NodeType::Pow);
                break;
            }
            default: {
                throw std::runtime_error(fmt::format("led: unsupported token ", tok.name()));
            }
            };

            return rhs;
        }
    };

} // namespace ParserBlocks

struct InfixParser {
    using Nud = ParserBlocks::Nud;
    using Led = ParserBlocks::Led;
    using Conv = ParserBlocks::Conv;
    using Token = ParserBlocks::Token;
    using TokenKind = ParserBlocks::TokenKind;

    template <typename T, typename U>
    static auto Parse(std::string const& infix, T const& toks, U const& vars) -> Tree
    {
        auto nodes = pratt::parser<Nud, Led, Conv, T, U>(infix, toks, vars).parse();
        return Tree(nodes).UpdateNodes();
    }

    static auto DefaultTokens()
    {
        return robin_hood::unordered_flat_map<std::string_view, Token> {
            { "+", Token(TokenKind::dynamic, "add", static_cast<size_t>(NodeType::Add), 10, pratt::associativity::left) },
            { "-", Token(TokenKind::dynamic, "sub", static_cast<size_t>(NodeType::Sub), 10, pratt::associativity::left) },
            { "*", Token(TokenKind::dynamic, "mul", static_cast<size_t>(NodeType::Mul), 20, pratt::associativity::left) },
            { "/", Token(TokenKind::dynamic, "div", static_cast<size_t>(NodeType::Div), 20, pratt::associativity::left) },
            { "^", Token(TokenKind::dynamic, "pow", static_cast<size_t>(NodeType::Pow), 30, pratt::associativity::right) },
            //{ "max", Token(TokenKind::dynamic, "max", static_cast<size_t>(NodeType::Fmax), 30, pratt::associativity::left) },
            //{ "min", Token(TokenKind::dynamic, "min", static_cast<size_t>(NodeType::Fmin), 30, pratt::associativity::left) },
            { "pow", Token(TokenKind::dynamic, "pow", static_cast<size_t>(NodeType::Pow), 30, pratt::associativity::right) },
            //{ "abs", Token(TokenKind::dynamic, "abs", static_cast<size_t>(NodeType::Abs), 30, pratt::associativity::none) },
            //{ "acos", Token(TokenKind::dynamic, "acos", static_cast<size_t>(NodeType::Acos), 30, pratt::associativity::none) },
            //{ "asin", Token(TokenKind::dynamic, "asin", static_cast<size_t>(NodeType::Asin), 30, pratt::associativity::none) },
            //{ "atan", Token(TokenKind::dynamic, "atan", static_cast<size_t>(NodeType::Atan), 30, pratt::associativity::none) },
            { "cbrt", Token(TokenKind::dynamic, "cbrt", static_cast<size_t>(NodeType::Cbrt), 30, pratt::associativity::none) },
            //{ "ceil", Token(TokenKind::dynamic, "ceil", static_cast<size_t>(NodeType::Ceil), 30, pratt::associativity::none) },
            { "cos", Token(TokenKind::dynamic, "cos", static_cast<size_t>(NodeType::Cos), 30, pratt::associativity::none) },
            //{ "cosh", Token(TokenKind::dynamic, "cosh", static_cast<size_t>(NodeType::Cosh), 30, pratt::associativity::none) },
            { "exp", Token(TokenKind::dynamic, "exp", static_cast<size_t>(NodeType::Exp), 30, pratt::associativity::none) },
            //{ "floor", Token(TokenKind::dynamic, "floor", static_cast<size_t>(NodeType::Floor), 30, pratt::associativity::none) },
            { "log", Token(TokenKind::dynamic, "log", static_cast<size_t>(NodeType::Log), 30, pratt::associativity::none) },
            //{ "log1p", Token(TokenKind::dynamic, "log1p", static_cast<size_t>(NodeType::Log1p), 30, pratt::associativity::none) },
            { "sin", Token(TokenKind::dynamic, "sin", static_cast<size_t>(NodeType::Sin), 30, pratt::associativity::none) },
            //{ "sinh", Token(TokenKind::dynamic, "sinh", static_cast<size_t>(NodeType::Sinh), 30, pratt::associativity::none) },
            { "sqrt", Token(TokenKind::dynamic, "sqrt", static_cast<size_t>(NodeType::Sqrt), 30, pratt::associativity::none) },
            { "square", Token(TokenKind::dynamic, "square", static_cast<size_t>(NodeType::Square), 30, pratt::associativity::right) },
            { "tan", Token(TokenKind::dynamic, "tan", static_cast<size_t>(NodeType::Tan), 30, pratt::associativity::none) },
            { "tanh", Token(TokenKind::dynamic, "tanh", static_cast<size_t>(NodeType::Tanh), 30, pratt::associativity::none) },
            { "(", Token(pratt::token_kind::lparen, "(", 0UL /* don't care */, 0, pratt::associativity::none) },
            { ")", Token(pratt::token_kind::rparen, ")", 0UL /* don't care */, 0, pratt::associativity::none) },
            { "eof", Token(pratt::token_kind::eof, "eof", 0UL /* don't care */, 0, pratt::associativity::none) }
        };
    }
};
} // namespace Operon

#endif

