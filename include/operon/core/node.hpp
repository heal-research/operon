// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef NODE_HPP
#define NODE_HPP

#include <array>
#include <bitset>
#include <cstdint>
#include <type_traits>

#include "types.hpp"
#include <fmt/format.h>

namespace Operon {
enum class NodeType : uint32_t {
    Add  = 1u << 0,
    Mul  = 1u << 1,
    Sub  = 1u << 2,
    Div  = 1u << 3,
    Aq   = 1u << 4,
    Fmax = 1u << 5,
    Fmin = 1u << 6,
    Pow  = 1u << 7,

    Abs   = 1u << 8,
    Acos  = 1u << 9,
    Asin  = 1u << 10,
    Atan  = 1u << 11,
    Cbrt  = 1u << 12,
    Ceil  = 1u << 13,
    Cos   = 1u << 14,
    Cosh  = 1u << 15,
    Exp   = 1u << 18,
    Floor = 1u << 19,
    Log   = 1u << 20,
    Log1p = 1u << 21,
    Sin   = 1u << 22,
    Sinh  = 1u << 23,
    Sqrt  = 1u << 24,
    Tan   = 1u << 25,
    Tanh  = 1u << 26,
    Square = 1u << 27,
    
    Dynamic = 1u << 28,
    Constant = 1u << 29,
    Variable = 1u << 30
};

using utype = std::underlying_type_t<NodeType>;
struct NodeTypes {
    // magic number keeping track of the number of different node types
    static constexpr size_t Count = 31;
    // returns the index of the given type in the NodeType enum
    static size_t GetIndex(NodeType type)
    {
        return std::bitset<Count>(static_cast<utype>(type) - 1).count();
    }
};

inline constexpr NodeType operator&(NodeType lhs, NodeType rhs) { return static_cast<NodeType>(static_cast<utype>(lhs) & static_cast<utype>(rhs)); }
inline constexpr NodeType operator|(NodeType lhs, NodeType rhs) { return static_cast<NodeType>(static_cast<utype>(lhs) | static_cast<utype>(rhs)); }
inline constexpr NodeType operator^(NodeType lhs, NodeType rhs) { return static_cast<NodeType>(static_cast<utype>(lhs) ^ static_cast<utype>(rhs)); }
inline constexpr NodeType operator~(NodeType lhs) { return static_cast<NodeType>(~static_cast<utype>(lhs)); }
inline NodeType& operator&=(NodeType& lhs, NodeType rhs)
{
    lhs = lhs & rhs;
    return lhs;
}
inline NodeType& operator|=(NodeType& lhs, NodeType rhs)
{
    lhs = lhs | rhs;
    return lhs;
}
inline NodeType& operator^=(NodeType& lhs, NodeType rhs)
{
    lhs = lhs ^ rhs;
    return lhs;
}

namespace {
    std::array<std::pair<std::string, std::string>, NodeTypes::Count> nodeDesc = {
        std::make_pair("+", "n-ary addition f(a,b,c,...) = a + b + c + ..."),
        std::make_pair("*", "n-ary multiplication f(a,b,c,...) = a * b * c * ..." ),
        std::make_pair("-", "n-ary subtraction f(a,b,c,...) = a - (b + c + ...)" ),
        std::make_pair("/", "n-ary division f(a,b,c,..) = a / (b * c * ...)" ),
        std::make_pair("aq", "analytical quotient f(a,b) = a / sqrt(1 + b^2)" ),
        std::make_pair("fmax", "maximum function f(a,b) = max(a,b)" ),
        std::make_pair("fmin", "minimum function f(a,b) = min(a,b)" ),
        std::make_pair("pow", "raise to power f(a,b) = a^b" ),
        std::make_pair("abs", "absolute value function f(a) = abs(a)" ),	
        std::make_pair("acos", "inverse cosine function f(a) = acos(a)" ),
        std::make_pair("asin", "inverse sine function f(a) = asin(a)" ),
        std::make_pair("atan", "inverse tangent function f(a) = atan(a)" ),
        std::make_pair("cbrt", "cube root function f(a) = cbrt(a)" ),
        std::make_pair("ceil", "ceiling function f(a) = ceil(a)" ),	
        std::make_pair("cos", "cosine function f(a) = cos(a)" ),
        std::make_pair("cosh", "hyperbolic cosine function f(a) = cosh(a)" ),
        std::make_pair("exp", "e raised to the given power f(a) = e^a" ),
        std::make_pair("floor", "floor function f(a) = floor(a)" ),
        std::make_pair("log", "natural (base e) logarithm f(a) = ln(a)" ),
        std::make_pair("log1p", "f(a) = ln(a + 1), accurate even when a is close to zero" ),
        std::make_pair("sin", "sine function f(a) = sin(a)" ),
        std::make_pair("sinh", "hyperbolic sine function f(a) = sinh(a)" ),
        std::make_pair("sqrt", "square root function f(a) = sqrt(a)" ),
        std::make_pair("square", "square function f(a) = a^2" ),
        std::make_pair("tan", "tangent function f(a) = tan(a)" ),
        std::make_pair("tanh", "hyperbolic tangent function f(a) = tanh(a)" ),
        std::make_pair("dyn", "user-defined function" ),
        std::make_pair("constant", "a constant value" ),
        std::make_pair("variable", "a dataset input with an associated weight" ),
    };
}

struct Node {
    Operon::Hash HashValue;
    Operon::Hash CalculatedHashValue; // for arithmetic terminal nodes whose hash value depends on their children
    Operon::Scalar Value; // value for constants or weighting factor for variables
    uint16_t Arity; // 0-65535
    uint16_t Length; // 0-65535
    uint16_t Depth; // 0-65535
    uint16_t Level;
    uint16_t Parent; // index of parent node
    NodeType Type;
    bool IsEnabled;

    Node() = default; 

    explicit Node(NodeType type) noexcept
        : Node(type, static_cast<Operon::Hash>(type))
    {
    }
    explicit Node(NodeType type, Operon::Hash hashValue) noexcept
        : HashValue(hashValue)
        , CalculatedHashValue(hashValue)
        , Type(type)
    {
        Arity = 0;
        if (Type < NodeType::Abs) // Add, Mul, Sub, Div, Aq, Fmax, Fmin, Pow
        {
            Arity = 2;
        } else if (Type < NodeType::Constant) // Abs, Asin, ... Square
        {
            Arity = 1;
        }
        Length = Arity;

        IsEnabled = true;

        Value = 1.;
    }

    const std::string& Name() const noexcept { return nodeDesc[NodeTypes::GetIndex(Type)].first; }
    const std::string& Desc() const noexcept { return nodeDesc[NodeTypes::GetIndex(Type)].second; }

    // comparison operators
    inline bool operator==(const Node& rhs) const noexcept
    {
        return CalculatedHashValue == rhs.CalculatedHashValue;
    }

    inline bool operator!=(const Node& rhs) const noexcept
    {
        return !((*this) == rhs);
    }

    inline bool operator<(const Node& rhs) const noexcept
    {
        return HashValue == rhs.HashValue ? CalculatedHashValue < rhs.CalculatedHashValue : HashValue < rhs.HashValue;
    }

    inline bool operator<=(const Node& rhs) const noexcept
    {
        return ((*this) == rhs || (*this) < rhs);
    }

    inline bool operator>(const Node& rhs) const noexcept
    {
        return !((*this) <= rhs);
    }

    inline bool operator>=(const Node& rhs) const noexcept
    {
        return !((*this) < rhs);
    }

    inline constexpr bool IsLeaf() const noexcept { return Arity == 0; }
    inline constexpr bool IsCommutative() const noexcept { return Type < NodeType::Sub; }

    template <NodeType... T>
    inline bool Is() const { return ((Type == T) || ...); }

    inline bool IsConstant() const { return Is<NodeType::Constant>(); }
    inline bool IsVariable() const { return Is<NodeType::Variable>(); }
    inline bool IsAddition() const { return Is<NodeType::Add>(); }
    inline bool IsSubtraction() const { return Is<NodeType::Sub>(); }
    inline bool IsMultiplication() const { return Is<NodeType::Mul>(); }
    inline bool IsDivision() const { return Is<NodeType::Div>(); }
    inline bool IsAq() const { return Is<NodeType::Aq>(); }
    inline bool IsFmax() const { return Is<NodeType::Fmax>(); }
    inline bool IsFmin() const { return Is<NodeType::Fmin>(); }
    inline bool IsPow() const { return Is<NodeType::Pow>(); }
    inline bool IsAbs() const { return Is<NodeType::Abs>(); }
    inline bool IsAcos() const { return Is<NodeType::Acos>(); }
    inline bool IsAsin() const { return Is<NodeType::Asin>(); }
    inline bool IsAtan() const { return Is<NodeType::Atan>(); }
    inline bool IsCubeRoot() const { return Is<NodeType::Cbrt>(); }
    inline bool IsCeil() const { return Is<NodeType::Ceil>(); }
    inline bool IsCos() const { return Is<NodeType::Cos>(); }
    inline bool IsCosh() const { return Is<NodeType::Cosh>(); }
    inline bool IsExp() const { return Is<NodeType::Exp>(); }
    inline bool IsFloor() const { return Is<NodeType::Floor>(); }
    inline bool IsLog() const { return Is<NodeType::Log>(); }
    inline bool IsLog1p() const { return Is<NodeType::Log1p>(); }
    inline bool IsSin() const { return Is<NodeType::Sin>(); }
    inline bool IsSinh() const { return Is<NodeType::Sinh>(); }
    inline bool IsSquareRoot() const { return Is<NodeType::Sqrt>(); }
    inline bool IsTan() const { return Is<NodeType::Tan>(); }
    inline bool IsTanh() const { return Is<NodeType::Tanh>(); }
    inline bool IsSquare() const { return Is<NodeType::Square>(); }
    inline bool IsDynamic() const { return Is<NodeType::Dynamic>(); }
};
}
#endif // NODE_H

