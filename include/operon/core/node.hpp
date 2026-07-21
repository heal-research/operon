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
    Constant,
    Variable,
    Ref,      // structural sharing: a backward reference to another node by index
    Function  // any named operation (built-in math op or user-registered), identified by HashValue
};

// Stable compile-time Operon::Hash constants for every built-in math op.
// A `Function`-typed Node representing a built-in op has HashValue equal to
// static_cast<Hash>(the matching BuiltinOp) — small sequential integers,
// chosen deliberately (not derived from anything else) so call sites can
// `switch (n.HashValue) { case Hash(BuiltinOp::Add): ... }` with ordinary
// compile-time case labels instead of a runtime map lookup. Carries no
// arity/category semantics of its own; see IsNaryOp/IsBinaryOp/IsUnaryOp
// below for that. Node::Function(hash, arity) is how a Node gets one of
// these values as its HashValue — there is no longer a NodeType enumerator
// per op, so nothing derives a BuiltinOp value from a Node's Type.
enum class BuiltinOp : Operon::Hash {
    // n-ary symbols
    Add = 0,
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
    Square
};

// One past BuiltinOp's last value — the built-in hash range is
// [0, BuiltinOpCount). Anything reserving "the range built-in ops occupy"
// (see symbol_library.hpp's ValidateUserHash) should use this, not
// NodeTypes::Count — the two are unrelated now that NodeType no longer has
// one enumerator per math op.
static auto constexpr BuiltinOpCount = static_cast<Operon::Hash>(BuiltinOp::Square) + 1;

// Composed functions (see InfixParser::ParseFunctionBody) are capped at
// arity <= kMaxComposedFunctionArity in v1 — a registration-time cap only,
// not assumed by consult-time machinery (see composed-functions design doc).
// Formal-parameter leaves get their own reserved hash band, immediately
// past the built-in range and disjoint from ordinary name-derived function
// hashes (see symbol_library.hpp's ValidateUserHash, which rejects a user
// hash landing here too) — so a param leaf can never collide with a
// dataset-bound Variable or a user-registered Function.
static auto constexpr kMaxComposedFunctionArity = 2UL;
constexpr auto ParamHash(std::size_t index) -> Operon::Hash
{
    return BuiltinOpCount + index;
}

// Sentinel "not a real op" value (mirrors NodeTypes::NoType), distinguishable
// from every genuine BuiltinOp value — used as a default template argument
// where "no specific op" needs a nameable type, e.g. Dispatch::Func/Diff's
// primary (unspecialized, "missing this specialization") templates.
static auto constexpr NoBuiltinOp = static_cast<BuiltinOp>(~Operon::Hash{0});

using UnderlyingNodeType = std::underlying_type_t<NodeType>;

struct NodeTypes {
    // total number of distinct node types
    static auto constexpr Count = static_cast<std::size_t>(NodeType::Function) + 1UL;

    // returns the index of the given type in the NodeType enum
    static constexpr auto GetIndex(NodeType type) -> size_t
    {
        return static_cast<std::size_t>(type);
    }

    static auto constexpr NoType{static_cast<NodeType>(0xFFU)};
};

// A bitset over "which built-in ops / terminal categories are enabled",
// used by PrimitiveSet::SetConfig. [0, BuiltinOpCount) map to BuiltinOp
// ordinals directly; the next NodeTypes::Count (4) slots map to
// Constant/Variable/Ref and a shared "any Function present" slot — the
// latter mirrors the pre-collapse Dynamic slot, since individual
// user-registered functions (whose HashValue lands outside
// [0, BuiltinOpCount)) were never individually representable in this
// bitset either; only PrimitiveSet's own per-hash Enable/Disable does that.
// Same total bit count as the pre-collapse 33-value NodeType bitset
// (BuiltinOpCount=29 + NodeTypes::Count=4), just re-derived from two
// smaller enums instead of one that no longer exists.
using PrimitiveSetConfig = Bitset<BuiltinOpCount + NodeTypes::Count>;

constexpr auto ToConfig(BuiltinOp op) -> PrimitiveSetConfig
{
    PrimitiveSetConfig c{};
    c.Set(static_cast<std::size_t>(op));
    return c;
}

constexpr auto ToConfig(NodeType type) -> PrimitiveSetConfig
{
    PrimitiveSetConfig c{};
    c.Set(BuiltinOpCount + NodeTypes::GetIndex(type));
    return c;
}

// The PrimitiveSetConfig bit index for a given (Type, HashValue) pair, e.g.
// from an existing Node: a built-in op (Type == Function, HashValue in
// [0, BuiltinOpCount)) maps to that BuiltinOp's own index; any other
// Function node (a user-registered function) maps to the shared "any
// Function present" slot (see the PrimitiveSetConfig comment above);
// Constant/Variable/Ref map to their own slot.
constexpr auto ConfigIndex(NodeType type, Operon::Hash hash) -> std::size_t
{
    if (type == NodeType::Function && hash < BuiltinOpCount) {
        return hash;
    }
    return BuiltinOpCount + NodeTypes::GetIndex(type);
}

// Combine a BuiltinOp/NodeType (any order/mix) into a PrimitiveSetConfig —
// e.g. `NodeType::Constant | NodeType::Variable | BuiltinOp::Add | BuiltinOp::Sub`.
constexpr auto operator|(BuiltinOp lhs, BuiltinOp rhs) -> PrimitiveSetConfig { return ToConfig(lhs) | ToConfig(rhs); }
constexpr auto operator|(NodeType lhs, NodeType rhs) -> PrimitiveSetConfig { return ToConfig(lhs) | ToConfig(rhs); }
constexpr auto operator|(NodeType lhs, BuiltinOp rhs) -> PrimitiveSetConfig { return ToConfig(lhs) | ToConfig(rhs); }
constexpr auto operator|(BuiltinOp lhs, NodeType rhs) -> PrimitiveSetConfig { return ToConfig(lhs) | ToConfig(rhs); }

// Extend a PrimitiveSetConfig with one more BuiltinOp/NodeType.
constexpr auto operator|(PrimitiveSetConfig lhs, BuiltinOp rhs) -> PrimitiveSetConfig { return lhs | ToConfig(rhs); }
constexpr auto operator|(PrimitiveSetConfig lhs, NodeType rhs) -> PrimitiveSetConfig { return lhs | ToConfig(rhs); }

constexpr auto operator|=(PrimitiveSetConfig& lhs, BuiltinOp rhs) -> PrimitiveSetConfig& { lhs = lhs | rhs; return lhs; }
constexpr auto operator|=(PrimitiveSetConfig& lhs, NodeType rhs) -> PrimitiveSetConfig& { lhs = lhs | rhs; return lhs; }

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

    // Fixed sentinel hashes for the terminal categories' default (no
    // explicit hash given) construction. Deliberately arbitrary constants
    // far outside BuiltinOpCount's occupied range, rather than
    // static_cast<Hash>(type) (what pre-collapse code did): basing a
    // terminal's default hash on its own small NodeType ordinal would make
    // "terminal hashes never collide with a built-in op's HashValue"
    // depend on remembering to keep NodeType's terminal enumerators
    // numbered after every BuiltinOp value — true by construction once,
    // silently breakable by a later edit to either enum. These values have
    // no such dependency: they're simply nowhere near [0, BuiltinOpCount).
    // Function has no meaningful default — every real Function node is
    // built via Node::Function(hash, arity) with a caller-supplied hash —
    // but it still gets a distinct, non-colliding sentinel so the plain
    // Node(NodeType::Function) path is inert rather than dangerous if
    // something calls it by mistake.
    static auto constexpr DefaultHash(NodeType type) noexcept -> Operon::Hash
    {
        switch (type) {
        case NodeType::Constant: return Operon::Hash{0x00000000434F4E53ULL}; // "CONS"
        case NodeType::Variable: return Operon::Hash{0x0000000056415249ULL}; // "VARI"
        case NodeType::Ref:      return Operon::Hash{0x0000000000524546ULL}; // "REF"
        case NodeType::Function: return Operon::Hash{0x0000000046554E43ULL}; // "FUNC"
        }
        return Operon::Hash{0};
    }

    explicit Node(NodeType type) noexcept
        : Node(type, DefaultHash(type))
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

    // The generalized "any named operation" factory: a built-in math op
    // (hash = static_cast<Hash>(some BuiltinOp)) or a user-registered
    // function (hash = a name-derived hash outside BuiltinOpCount's range,
    // see symbol_library.hpp's ValidateUserHash) are both just Function
    // nodes now, distinguished only by HashValue. Arity is caller-supplied
    // rather than looked up from a registry: Node construction is the
    // hottest path in the library (tree creators, crossover, mutation,
    // Simplify's constant-folding), and every call site already has a
    // PrimitiveSet/FunctionInfo in scope with the arity known, so a
    // mandatory hash-map probe here would add cost to code that's
    // currently a couple of integer assignments.
    static auto Function(Operon::Hash hash, uint16_t arity) noexcept
    {
        Node node(NodeType::Function, hash);
        node.Arity  = arity;
        node.Length = arity;
        // The two-arg ctor computes Optimize from Arity=0 (leaf) before
        // this factory overrides Arity to the real value above - recompute
        // explicitly rather than leave a stale true. Function nodes (arity
        // >= 1, always) are never coefficient-optimized regardless.
        node.Optimize = false;
        return node;
    }

    // Not noexcept: unlike the old std::string const& return, this now
    // copies (and, on invariant violation, can throw std::out_of_range).
    [[nodiscard]] OPERON_EXPORT auto Name() const -> std::string;
    [[nodiscard]] OPERON_EXPORT auto Desc() const -> std::string;

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
    [[nodiscard]] auto IsCommutative() const noexcept -> bool { return IsOp<BuiltinOp::Add, BuiltinOp::Mul, BuiltinOp::Fmin, BuiltinOp::Fmax>(); }

    template <NodeType... T>
    [[nodiscard]] auto Is() const -> bool { return ((Type == T) || ...); }

    // BuiltinOp counterpart of Is<NodeType...>() above: compares HashValue
    // instead of Type. Equivalent to the NodeType overload for any node
    // constructed the ordinary way (Node(NodeType) sets
    // HashValue = static_cast<Hash>(Type)), but usable where only a hash is
    // in scope (e.g. dispatch keyed by HashValue). Not a strict no-op vs.
    // the NodeType overload in one theoretical edge case: a Variable node's
    // HashValue is an unconstrained Hasher{}(name) (unlike registered
    // functions', which ValidateUserHash keeps out of the built-in ordinal
    // range), so a name whose hash happens to collide into
    // [0, NodeTypes::Count) would make this overload misclassify it as a
    // math op. Astronomically unlikely (a ~29-in-2^64 chance) and not
    // guarded against, same as it wasn't before this overload existed.
    //
    // Named IsOp rather than a same-named Is<BuiltinOp...>() overload: a
    // real clang-cl (MSVC ABI) mangling bug collapses two function templates
    // overloaded solely by an enum-typed non-type template parameter pack
    // into the identical mangled symbol whenever the packs' underlying
    // integer values coincide (e.g. Is<NodeType::Ref>() [value 2] and
    // Is<BuiltinOp::Sub>() [value 2] both mangled to
    // `??$Is@$02@Node@Operon@@QEBA_NXZ`), causing a duplicate-definition
    // link error — hit in practice by PR #140's Windows CI
    // (`error: definition with same mangled name`). Distinct names sidestep
    // the collision entirely; this is a compiler ABI limitation, not
    // something fixable by choosing different enum values (any two enums
    // with overlapping value ranges will eventually collide again).
    template <BuiltinOp... Op>
    [[nodiscard]] auto IsOp() const -> bool { return ((HashValue == static_cast<Operon::Hash>(Op)) || ...); }

    [[nodiscard]] auto IsConstant() const -> bool { return Is<NodeType::Constant>(); }
    [[nodiscard]] auto IsVariable() const -> bool { return Is<NodeType::Variable>(); }
    [[nodiscard]] auto IsRef()      const -> bool { return Is<NodeType::Ref>(); }
    [[nodiscard]] auto IsAddition() const -> bool { return IsOp<BuiltinOp::Add>(); }
    [[nodiscard]] auto IsSubtraction() const -> bool { return IsOp<BuiltinOp::Sub>(); }
    [[nodiscard]] auto IsMultiplication() const -> bool { return IsOp<BuiltinOp::Mul>(); }
    [[nodiscard]] auto IsDivision() const -> bool { return IsOp<BuiltinOp::Div>(); }
    [[nodiscard]] auto IsAq() const -> bool { return IsOp<BuiltinOp::Aq>(); }
    [[nodiscard]] auto IsPow() const -> bool { return IsOp<BuiltinOp::Pow>(); }
    [[nodiscard]] auto IsPowabs() const -> bool { return IsOp<BuiltinOp::Powabs>(); }
    [[nodiscard]] auto IsExp() const -> bool { return IsOp<BuiltinOp::Exp>(); }
    [[nodiscard]] auto IsLog() const -> bool { return IsOp<BuiltinOp::Log>(); }
    [[nodiscard]] auto IsSin() const -> bool { return IsOp<BuiltinOp::Sin>(); }
    [[nodiscard]] auto IsCos() const -> bool { return IsOp<BuiltinOp::Cos>(); }
    [[nodiscard]] auto IsTan() const -> bool { return IsOp<BuiltinOp::Tan>(); }
    [[nodiscard]] auto IsTanh() const -> bool { return IsOp<BuiltinOp::Tanh>(); }
    [[nodiscard]] auto IsSquareRoot() const -> bool { return IsOp<BuiltinOp::Sqrt>(); }
    [[nodiscard]] auto IsCubeRoot() const -> bool { return IsOp<BuiltinOp::Cbrt>(); }
    [[nodiscard]] auto IsSquare() const -> bool { return IsOp<BuiltinOp::Square>(); }
    [[nodiscard]] auto IsFunction() const -> bool { return Is<NodeType::Function>(); }

    // BuiltinOp counterparts of the old NodeType-keyed IsNary/IsBinary/
    // IsUnary/IsNullary, which no longer type-check (NodeType has no
    // per-op enumerators left to compare against). No IsNullary
    // counterpart: BuiltinOp only covers the math-op subset, never the
    // terminal categories (Constant/Variable/Ref) that IsNullary used to
    // distinguish — use `!Is<NodeType::Function>()` for that instead.
    template<BuiltinOp Op>
    static auto constexpr IsNaryOp = Op <= BuiltinOp::Fmax;

    template<BuiltinOp Op>
    static auto constexpr IsBinaryOp = Op > BuiltinOp::Fmax && Op <= BuiltinOp::Powabs;

    template<BuiltinOp Op>
    static auto constexpr IsUnaryOp = Op > BuiltinOp::Powabs;
};
} // namespace Operon
#endif
