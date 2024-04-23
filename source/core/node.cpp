// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/core/node.hpp"

#include <iterator>       // for pair
#include <unordered_map>  // for unordered_map, _Map_base<>::mapped_type
#include <utility>        // for make_pair, pair
#include <string>         // for string

#include "operon/core/types.hpp"

using std::pair;
using std::string;
using std::unordered_map;

namespace Operon {
    static const Operon::Map<NodeType, pair<string, string>> NodeDesc = {
        { NodeType::Add,      std::make_pair("+", "n-ary addition f(a,b,c,...) = a + b + c + ...") },
        { NodeType::Mul,      std::make_pair("*", "n-ary multiplication f(a,b,c,...) = a * b * c * ..." ) },
        { NodeType::Sub,      std::make_pair("-", "n-ary subtraction f(a,b,c,...) = a - (b + c + ...)" ) },
        { NodeType::Div,      std::make_pair("/", "n-ary division f(a,b,c,..) = a / (b * c * ...)" ) },
        { NodeType::Fmin,     std::make_pair("fmin", "minimum function f(a,b) = min(a,b)" ) },
        { NodeType::Fmax,     std::make_pair("fmax", "maximum function f(a,b) = max(a,b)" ) },
        { NodeType::Aq,       std::make_pair("aq", "analytical quotient f(a,b) = a / sqrt(1 + b^2)" ) },
        { NodeType::Pow,      std::make_pair("pow", "raise to power f(a,b) = a^b" ) },
        { NodeType::Abs,      std::make_pair("abs", "absolute value function f(a) = abs(a)" ) },
        { NodeType::Acos,     std::make_pair("acos", "inverse cosine function f(a) = acos(a)" ) },
        { NodeType::Asin,     std::make_pair("asin", "inverse sine function f(a) = asin(a)" ) },
        { NodeType::Atan,     std::make_pair("atan", "inverse tangent function f(a) = atan(a)" ) },
        { NodeType::Cbrt,     std::make_pair("cbrt", "cube root function f(a) = cbrt(a)" ) },
        { NodeType::Ceil,     std::make_pair("ceil", "ceiling function f(a) = ceil(a)" ) },
        { NodeType::Cos,      std::make_pair("cos", "cosine function f(a) = cos(a)" ) },
        { NodeType::Cosh,     std::make_pair("cosh", "hyperbolic cosine function f(a) = cosh(a)" ) },
        { NodeType::Exp,      std::make_pair("exp", "e raised to the given power f(a) = e^a" ) },
        { NodeType::Floor,    std::make_pair("floor", "floor function f(a) = floor(a)" ) },
        { NodeType::Log,      std::make_pair("log", "natural (base e) logarithm f(a) = ln(a)" ) },
        { NodeType::Logabs,   std::make_pair("logabs", "natural (base e) logarithm of absolute value f(a) = ln(|a|)" ) },
        { NodeType::Log1p,    std::make_pair("log1p", "f(a) = ln(a + 1), accurate even when a is close to zero" ) },
        { NodeType::Sin,      std::make_pair("sin", "sine function f(a) = sin(a)" ) },
        { NodeType::Sinh,     std::make_pair("sinh", "hyperbolic sine function f(a) = sinh(a)" ) },
        { NodeType::Sqrt,     std::make_pair("sqrt", "square root function f(a) = sqrt(a)" ) },
        { NodeType::Sqrtabs,  std::make_pair("sqrtabs", "square root of absolute value function f(a) = sqrt(|a|)" ) },
        { NodeType::Tan,      std::make_pair("tan", "tangent function f(a) = tan(a)" ) },
        { NodeType::Tanh,     std::make_pair("tanh", "hyperbolic tangent function f(a) = tanh(a)" ) },
        { NodeType::Square,   std::make_pair("square", "square function f(a) = a^2" ) },
        { NodeType::Dynamic,  std::make_pair("dyn", "user-defined function" ) },
        { NodeType::Constant, std::make_pair("constant", "a constant value" ) },
        { NodeType::Variable, std::make_pair("variable", "a dataset input with an associated weight" ) },
    };

    auto Node::Name() const noexcept -> std::string const& { return NodeDesc.at(Type).first; }
    auto Node::Desc() const noexcept -> std::string const& { return NodeDesc.at(Type).second; }

} // namespace Operon

