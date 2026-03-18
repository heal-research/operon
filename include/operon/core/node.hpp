// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CORE_NODE_HPP
#define OPERON_CORE_NODE_HPP

#include <cstdint>
#include <type_traits>

#include "operon/collections/bitset.hpp"
#include "operon/operon_export.hpp"
#include "types.hpp"

namespace Operon {
enum class NodeType : uint8_t {
    // n-ary symbols
    Add      = 0,
    Mul,
    Sub,
    Div,
    Fmin,
    Fmax,

    // binary symbols
    Aq,
    Pow,
    Powabs,

    // unary symbols
    Abs,
    Acos,
    Asin,
    Atan,
    Cbrt,
    Ceil,
    Cos,
    Cosh,
    Exp,
    Floor,
    Log,
    Logabs,
    Log1p,
    Sin,
    Sinh,
    Sqrt,
    Sqrtabs,
    Tan,
    Tanh,
    Square,

    // nullary symbols (dynamic can be anything)
    Dynamic,
    Constant,
    Variable
};

using UnderlyingNodeType = std::underlying_type_t<NodeType>;

struct NodeTypes {
    // total number of distinct node types
    static auto constexpr Count = static_cast<std::size_t>(NodeType::Variable) + 1UL;

    // returns the index of the given type in the NodeType enum
    static constexpr auto GetIndex(NodeType type) -> size_t
    {
        return static_cast<std::size_t>(type);
    }

    static auto constexpr NoType{static_cast<NodeType>(0xFFU)};
};

// A bitset over all node types, used to represent primitive set configurations.
using PrimitiveSetConfig = Bitset<NodeTypes::Count>;

// Build a PrimitiveSetConfig from two individual NodeType values.
inline constexpr auto operator|(NodeType lhs, NodeType rhs) -> PrimitiveSetConfig
{
    PrimitiveSetConfig c{};
    c.Set(NodeTypes::GetIndex(lhs));
    c.Set(NodeTypes::GetIndex(rhs));
    return c;
}

// Extend a PrimitiveSetConfig with one more NodeType.
inline constexpr auto operator|(PrimitiveSetConfig lhs, NodeType rhs) -> PrimitiveSetConfig
{
    lhs.Set(NodeTypes::GetIndex(rhs));
    return lhs;
}

inline constexpr auto operator|=(PrimitiveSetConfig& lhs, NodeType rhs) -> PrimitiveSetConfig&
{
    lhs.Set(NodeTypes::GetIndex(rhs));
    return lhs;
}

struct Node {
    Operon::Hash HashValue; // needs to be unique for each node type
    Operon::Hash mutable CalculatedHashValue; // for arithmetic terminal nodes whose hash value depends on their children
    Operon::Scalar Value; // value for constants or weighting factor for variables
    uint16_t Arity; // 0-65535
    uint16_t Length; // 0-65535
    uint16_t Depth; // 0-65535
    uint16_t Level; // length of the path to the root node
    uint16_t Parent; // index of parent node
    NodeType Type;
    bool IsEnabled;
    bool Optimize;

    Node() = default;

    explicit Node(NodeType type) noexcept
        : Node(type, static_cast<Operon::Hash>(type))
    {
    }

    explicit Node(NodeType type, Operon::Hash hashValue) noexcept
        : HashValue(hashValue)
        , CalculatedHashValue(hashValue)
        , Arity(0UL)
        , Length(0UL)
        , Depth(1UL)
        , Level(0UL)
        , Parent(0UL)
        , Type(type)
    {
        if (Type < NodeType::Abs) // Add, Mul, Sub, Div, Fmin, Fmax, Aq, Pow, Powabs
        {
            Arity = 2;
        } else if (Type < NodeType::Dynamic) // Abs through Square (unary)
        {
            Arity = 1;
        }
        Length = Arity;
        IsEnabled = true;
        Optimize = IsLeaf(); // we only optimize leaf nodes
        Value = 1.;
    }

    static auto Constant(double value)
    {
        Node node(NodeType::Constant);
        node.Value = static_cast<Operon::Scalar>(value);
        return node;
    }

    [[nodiscard]] OPERON_EXPORT auto Name() const noexcept -> std::string const&;
    [[nodiscard]] OPERON_EXPORT auto Desc() const noexcept -> std::string const&;

    // Register a display name (and optional description) for a Dynamic node hash.
    // After registration, Name() and Desc() return the provided strings instead of "dyn".
    static OPERON_EXPORT void RegisterName(Operon::Hash hash, std::string name, std::string desc = {});

    // comparison operators
    inline auto operator==(const Node& rhs) const noexcept -> bool
    {
        return CalculatedHashValue == rhs.CalculatedHashValue;
    }

    inline auto operator!=(const Node& rhs) const noexcept -> bool
    {
        return !((*this) == rhs);
    }

    inline auto operator<(const Node& rhs) const noexcept -> bool
    {
        return std::tie(HashValue, CalculatedHashValue) < std::tie(rhs.HashValue, rhs.CalculatedHashValue);
    }

    inline auto operator<=(const Node& rhs) const noexcept -> bool
    {
        return ((*this) == rhs || (*this) < rhs);
    }

    inline auto operator>(const Node& rhs) const noexcept -> bool
    {
        return !((*this) <= rhs);
    }

    inline auto operator>=(const Node& rhs) const noexcept -> bool
    {
        return !((*this) < rhs);
    }

    [[nodiscard]] inline auto IsLeaf() const noexcept -> bool { return Arity == 0; }
    [[nodiscard]] inline auto IsCommutative() const noexcept -> bool { return Is<NodeType::Add, NodeType::Mul, NodeType::Fmin, NodeType::Fmax>(); }

    template <NodeType... T>
    [[nodiscard]] inline auto Is() const -> bool { return ((Type == T) || ...); }

    [[nodiscard]] inline auto IsConstant() const -> bool { return Is<NodeType::Constant>(); }
    [[nodiscard]] inline auto IsVariable() const -> bool { return Is<NodeType::Variable>(); }
    [[nodiscard]] inline auto IsAddition() const -> bool { return Is<NodeType::Add>(); }
    [[nodiscard]] inline auto IsSubtraction() const -> bool { return Is<NodeType::Sub>(); }
    [[nodiscard]] inline auto IsMultiplication() const -> bool { return Is<NodeType::Mul>(); }
    [[nodiscard]] inline auto IsDivision() const -> bool { return Is<NodeType::Div>(); }
    [[nodiscard]] inline auto IsAq() const -> bool { return Is<NodeType::Aq>(); }
    [[nodiscard]] inline auto IsPow() const -> bool { return Is<NodeType::Pow>(); }
    [[nodiscard]] inline auto IsPowabs() const -> bool { return Is<NodeType::Powabs>(); }
    [[nodiscard]] inline auto IsExp() const -> bool { return Is<NodeType::Exp>(); }
    [[nodiscard]] inline auto IsLog() const -> bool { return Is<NodeType::Log>(); }
    [[nodiscard]] inline auto IsSin() const -> bool { return Is<NodeType::Sin>(); }
    [[nodiscard]] inline auto IsCos() const -> bool { return Is<NodeType::Cos>(); }
    [[nodiscard]] inline auto IsTan() const -> bool { return Is<NodeType::Tan>(); }
    [[nodiscard]] inline auto IsTanh() const -> bool { return Is<NodeType::Tanh>(); }
    [[nodiscard]] inline auto IsSquareRoot() const -> bool { return Is<NodeType::Sqrt>(); }
    [[nodiscard]] inline auto IsCubeRoot() const -> bool { return Is<NodeType::Cbrt>(); }
    [[nodiscard]] inline auto IsSquare() const -> bool { return Is<NodeType::Square>(); }
    [[nodiscard]] inline auto IsDynamic() const -> bool { return Is<NodeType::Dynamic>(); }

    template<NodeType Type>
    static auto constexpr IsNary = Type < NodeType::Aq;

    template<NodeType Type>
    static auto constexpr IsBinary = Type > NodeType::Fmax && Type < NodeType::Abs;

    template<NodeType Type>
    static auto constexpr IsUnary = Type > NodeType::Powabs && Type < NodeType::Dynamic;

    template<NodeType Type>
    static auto constexpr IsNullary = Type > NodeType::Square;
};
} // namespace Operon
#endif
