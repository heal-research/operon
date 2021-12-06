// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PARSER_HPP
#define OPERON_PARSER_HPP

#include <stdexcept>
#include <unordered_map>

#include "core/tree.hpp"
#include "core/types.hpp"
#include "parser.hpp"

namespace Operon {

using token_kind = pratt::token_kind;
using token = pratt::token<Operon::Vector<Node>>;

struct conv {
    token::value_t operator()(double val) const noexcept {
        Node node(NodeType::Constant);
        node.Value = static_cast<Operon::Scalar>(val);
        return token::value_t { node };
    }
};

struct nud {
    using token_t = token;
    using value_t = token::value_t;

    template<typename Parser>
    value_t operator()(Parser& parser, token_kind tok, token_t const& left) const {
        if (tok == token_kind::constant) { 
            return left.value; 
        }

        if (tok == token_kind::variable) {
            auto hash = parser.template get_desc<Operon::Hash>(left.name);
            if (!hash.has_value()) {
                throw std::invalid_argument(fmt::format("unknown variable name {}\n", left.name));
            }
            return value_t { Node(NodeType::Variable, hash.value()) };
        }

        auto bp = pratt::token_precedence[tok]; // binding power
        if (tok == token_kind::lparen) {
            return parser.parse_bp(bp, token_kind::rparen).value;
        }
        auto result = std::move(parser.parse_bp(bp, token_kind::eof).value);

        switch (tok) {
        case token_kind::sub:    { result.push_back(Node(NodeType::Sub)); result.back().Arity = 1; break; }	
        case token_kind::abs:    { result.push_back(Node(NodeType::Abs)); break; }
        case token_kind::acos:   { result.push_back(Node(NodeType::Acos)); break; }
        case token_kind::asin:   { result.push_back(Node(NodeType::Asin)); break; }
        case token_kind::atan:   { result.push_back(Node(NodeType::Atan)); break; }
        case token_kind::cbrt:   { result.push_back(Node(NodeType::Cbrt)); break; }
        case token_kind::ceil:   { result.push_back(Node(NodeType::Ceil)); break; }
        case token_kind::cos:    { result.push_back(Node(NodeType::Cos)); break; }
        case token_kind::cosh:   { result.push_back(Node(NodeType::Cosh)); break; }
        case token_kind::erc:    { result.push_back(Node(NodeType::Erf)); break; }
        case token_kind::erfc:   { result.push_back(Node(NodeType::Erfc)); break; }
        case token_kind::exp:    { result.push_back(Node(NodeType::Exp)); break; }
        case token_kind::floor:  { result.push_back(Node(NodeType::Floor)); break; }
        case token_kind::log:    { result.push_back(Node(NodeType::Log)); break; }
        case token_kind::log1p:  { result.push_back(Node(NodeType::Log1p)); break; }
        case token_kind::sin:    { result.push_back(Node(NodeType::Sin)); break; }
        case token_kind::sinh:   { result.push_back(Node(NodeType::Sinh)); break; }
        case token_kind::sqrt:   { result.push_back(Node(NodeType::Sqrt)); break; }
        case token_kind::square: { result.push_back(Node(NodeType::Square)); break; }
        case token_kind::tan:    { result.push_back(Node(NodeType::Tan)); break; }
        case token_kind::tanh:   { result.push_back(Node(NodeType::Tanh)); break; }
        default: {
            throw std::runtime_error(fmt::format("nud: unsupported token {}\n", std::string(pratt::token_name[static_cast<int>(tok)])));
        };
        }
        
        return result;
    }
};

struct led {
    using token_t = token;
    using value_t = token::value_t;

    template<typename Parser>
    value_t operator()(Parser&, token_kind tok, token_t const& left, token_t& right) const {
        auto const& lhs = left.value;
        auto &rhs = right.value;

        rhs.reserve(lhs.size() + rhs.size() + 1);
        std::copy(lhs.begin(), lhs.end(), std::back_inserter(rhs));

        switch (tok) {
        case token_kind::add:  { rhs.push_back(Node(NodeType::Add)); break; }
        case token_kind::sub:  { rhs.push_back(Node(NodeType::Sub)); break; }
        case token_kind::mul:  { rhs.push_back(Node(NodeType::Mul)); break; }
        case token_kind::div:  { rhs.push_back(Node(NodeType::Div)); break; }
        case token_kind::aq:   { rhs.push_back(Node(NodeType::Aq)); break; }
        case token_kind::fmax: { rhs.push_back(Node(NodeType::Fmax)); break; }
        case token_kind::fmin: { rhs.push_back(Node(NodeType::Fmin)); break; }
        case token_kind::pow:  { rhs.push_back(Node(NodeType::Pow)); break; }
        default: {
            throw std::runtime_error(fmt::format("led: unsupported token ", std::string(pratt::token_name[static_cast<int>(tok)])));
        }
        };

        return rhs;
    }
};

struct InfixParser {
    template<typename Map>
    static Tree Parse(std::string const& infix, Map const& vars) {
        auto nodes = pratt::parser<nud, led, conv, Map>(infix, vars).parse();
        return Tree(nodes).UpdateNodes();
    }
};
} // namespace

#endif
