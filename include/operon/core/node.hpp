// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CORE_NODE_HPP
#define OPERON_CORE_NODE_HPP

#include <cstdint>
#include <type_traits>

#include "operon/operon_export.hpp"
#include "types.hpp"

namespace Operon {
enum class NodeType : uint32_t {
    // n-ary symbols
    Add      = 1U << 0U,
    Mul      = 1U << 1U,
    Sub      = 1U << 2U,
    Div      = 1U << 3U,
    Fmin     = 1U << 4U,
    Fmax     = 1U << 5U,

    // binary symbols
    Aq       = 1U << 6U,
    Pow      = 1U << 7U,

    // unary symbols
    Abs      = 1U << 8U,
    Acos     = 1U << 9U,
    Asin     = 1U << 10U,
    Atan     = 1U << 11U,
    Cbrt     = 1U << 12U,
    Ceil     = 1U << 13U,
    Cos      = 1U << 14U,
    Cosh     = 1U << 15U,
    Exp      = 1U << 16U,
    Floor    = 1U << 17U,
    Log      = 1U << 18U,
    Logabs   = 1U << 19U,
    Log1p    = 1U << 20U,
    Sin      = 1U << 21U,
    Sinh     = 1U << 22U,
    Sqrt     = 1U << 23U,
    Sqrtabs  = 1U << 24U,
    Tan      = 1U << 25U,
    Tanh     = 1U << 26U,
    Square   = 1U << 27U,

    // nullary symbols (dynamic can be anything)
    Dynamic  = 1U << 28U,
    Constant = 1U << 29U,
    Variable = 1U << 30U
};

using PrimitiveSetConfig = NodeType;

using UnderlyingNodeType = std::underlying_type_t<NodeType>;
struct NodeTypes {
    // magic number keeping track of the number of different node types
    static auto constexpr Count = std::countr_zero(static_cast<uint64_t>(NodeType::Variable)) + 1UL;

    // returns the index of the given type in the NodeType enum
    static auto GetIndex(NodeType type) -> size_t
    {
        return std::countr_zero(static_cast<uint32_t>(type));
    }

    static auto constexpr NoType{NodeType{123456}};
};

inline constexpr auto operator&(NodeType lhs, NodeType rhs) -> NodeType { return static_cast<NodeType>(static_cast<UnderlyingNodeType>(lhs) & static_cast<UnderlyingNodeType>(rhs)); }
inline constexpr auto operator|(NodeType lhs, NodeType rhs) -> NodeType { return static_cast<NodeType>(static_cast<UnderlyingNodeType>(lhs) | static_cast<UnderlyingNodeType>(rhs)); }
inline constexpr auto operator^(NodeType lhs, NodeType rhs) -> NodeType { return static_cast<NodeType>(static_cast<UnderlyingNodeType>(lhs) ^ static_cast<UnderlyingNodeType>(rhs)); }
inline constexpr auto operator~(NodeType lhs) -> NodeType { return static_cast<NodeType>(~static_cast<UnderlyingNodeType>(lhs)); }
inline auto operator&=(NodeType& lhs, NodeType rhs) -> NodeType&
{
    lhs = lhs & rhs;
    return lhs;
}
inline auto operator|=(NodeType& lhs, NodeType rhs) -> NodeType&
{
    lhs = lhs | rhs;
    return lhs;
}
inline auto operator^=(NodeType& lhs, NodeType rhs) -> NodeType&
{
    lhs = lhs ^ rhs;
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
        if (Type < NodeType::Abs) // Add, Mul, Sub, Div, Aq, Pow
        {
            Arity = 2;
        } else if (Type < NodeType::Dynamic) // Log, Exp, Sin, Cos, Tan, Tanh, Sqrt, Cbrt, Square
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
    static auto constexpr IsUnary = Type > NodeType::Pow && Type < NodeType::Dynamic;

    template<NodeType Type>
    static auto constexpr IsNullary = Type > NodeType::Square;
};
} // namespace Operon
#endif
