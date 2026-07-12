// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

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
    Variable,
    Ref       // structural sharing: a backward reference to another node by index
};

// Arity and category (IsNary/IsBinary/IsUnary/IsNullary below, and the arity
// inference in Node::Node) are derived from enumerator *position* in the
// NodeType enum above, not from any per-type tag. These asserts pin the
// ordering invariants that inference relies on, so a future reorder of the
// enum is a compile error rather than a silent arity miscompute.
static_assert(NodeType::Fmax < NodeType::Aq, "n-ary/binary boundary: Fmax must be the last n-ary symbol");
static_assert(NodeType::Powabs < NodeType::Abs, "binary/unary boundary: Powabs must be the last binary symbol");
static_assert(NodeType::Square < NodeType::Dynamic, "unary/nullary boundary: Square must be the last unary symbol");

using UnderlyingNodeType = std::underlying_type_t<NodeType>;

struct NodeTypes {
    // total number of distinct node types
    static auto constexpr Count = static_cast<std::size_t>(NodeType::Ref) + 1UL;

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
constexpr auto operator|(NodeType lhs, NodeType rhs) -> PrimitiveSetConfig
{
    PrimitiveSetConfig c{};
    c.Set(NodeTypes::GetIndex(lhs));
    c.Set(NodeTypes::GetIndex(rhs));
    return c;
}

// Extend a PrimitiveSetConfig with one more NodeType.
constexpr auto operator|(PrimitiveSetConfig lhs, NodeType rhs) -> PrimitiveSetConfig
{
    lhs.Set(NodeTypes::GetIndex(rhs));
    return lhs;
}

constexpr auto operator|=(PrimitiveSetConfig& lhs, NodeType rhs) -> PrimitiveSetConfig&
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
    uint16_t RefTo; // only meaningful when Type == NodeType::Ref; must point backward (RefTo < index of this node)

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
        , RefTo(0)
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
        Optimize = IsLeaf() && !IsRef();
        Value = 1.;
    }

    static auto Constant(double value)
    {
        Node node(NodeType::Constant);
        node.Value = static_cast<Operon::Scalar>(value);
        return node;
    }

    static auto Ref(uint16_t target) noexcept
    {
        Node node(NodeType::Ref);
        node.RefTo    = target;
        return node;
    }

    [[nodiscard]] OPERON_EXPORT auto Name() const noexcept -> std::string const&;
    [[nodiscard]] OPERON_EXPORT auto Desc() const noexcept -> std::string const&;

    // Register a display name (and optional description) for a Dynamic node hash.
    // After registration, Name() and Desc() return the provided strings instead of "dyn".
    static OPERON_EXPORT void RegisterName(Operon::Hash hash, std::string name, std::string desc = {});

    // comparison operators
    auto operator==(const Node& rhs) const noexcept -> bool
    {
        return CalculatedHashValue == rhs.CalculatedHashValue;
    }

    auto operator!=(const Node& rhs) const noexcept -> bool
    {
        return !((*this) == rhs);
    }

    auto operator<(const Node& rhs) const noexcept -> bool
    {
        return std::tie(HashValue, CalculatedHashValue) < std::tie(rhs.HashValue, rhs.CalculatedHashValue);
    }

    auto operator<=(const Node& rhs) const noexcept -> bool
    {
        return ((*this) == rhs || (*this) < rhs);
    }

    auto operator>(const Node& rhs) const noexcept -> bool
    {
        return !((*this) <= rhs);
    }

    auto operator>=(const Node& rhs) const noexcept -> bool
    {
        return !((*this) < rhs);
    }

    [[nodiscard]] auto IsLeaf() const noexcept -> bool { return Arity == 0; }
    [[nodiscard]] auto IsCommutative() const noexcept -> bool { return Is<NodeType::Add, NodeType::Mul, NodeType::Fmin, NodeType::Fmax>(); }

    template <NodeType... T>
    [[nodiscard]] auto Is() const -> bool { return ((Type == T) || ...); }

    [[nodiscard]] auto IsConstant() const -> bool { return Is<NodeType::Constant>(); }
    [[nodiscard]] auto IsVariable() const -> bool { return Is<NodeType::Variable>(); }
    [[nodiscard]] auto IsRef()      const -> bool { return Is<NodeType::Ref>(); }
    [[nodiscard]] auto IsAddition() const -> bool { return Is<NodeType::Add>(); }
    [[nodiscard]] auto IsSubtraction() const -> bool { return Is<NodeType::Sub>(); }
    [[nodiscard]] auto IsMultiplication() const -> bool { return Is<NodeType::Mul>(); }
    [[nodiscard]] auto IsDivision() const -> bool { return Is<NodeType::Div>(); }
    [[nodiscard]] auto IsAq() const -> bool { return Is<NodeType::Aq>(); }
    [[nodiscard]] auto IsPow() const -> bool { return Is<NodeType::Pow>(); }
    [[nodiscard]] auto IsPowabs() const -> bool { return Is<NodeType::Powabs>(); }
    [[nodiscard]] auto IsExp() const -> bool { return Is<NodeType::Exp>(); }
    [[nodiscard]] auto IsLog() const -> bool { return Is<NodeType::Log>(); }
    [[nodiscard]] auto IsSin() const -> bool { return Is<NodeType::Sin>(); }
    [[nodiscard]] auto IsCos() const -> bool { return Is<NodeType::Cos>(); }
    [[nodiscard]] auto IsTan() const -> bool { return Is<NodeType::Tan>(); }
    [[nodiscard]] auto IsTanh() const -> bool { return Is<NodeType::Tanh>(); }
    [[nodiscard]] auto IsSquareRoot() const -> bool { return Is<NodeType::Sqrt>(); }
    [[nodiscard]] auto IsCubeRoot() const -> bool { return Is<NodeType::Cbrt>(); }
    [[nodiscard]] auto IsSquare() const -> bool { return Is<NodeType::Square>(); }
    [[nodiscard]] auto IsDynamic() const -> bool { return Is<NodeType::Dynamic>(); }

    template<NodeType Type>
    static auto constexpr IsNary = Type < NodeType::Aq;

    template<NodeType Type>
    static auto constexpr IsBinary = Type > NodeType::Fmax && Type < NodeType::Abs;

    template<NodeType Type>
    static auto constexpr IsUnary = Type > NodeType::Powabs && Type < NodeType::Dynamic;

    template<NodeType Type>
    static auto constexpr IsNullary = Type > NodeType::Square; // Dynamic, Constant, Variable, Ref
};
} // namespace Operon
#endif
