// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_GRAMMAR_HPP
#define OPERON_CORE_GRAMMAR_HPP

#include <array>
#include <optional>
#include <span>
#include <string_view>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Nonterminals for the weighted-sum grammar; SimpleExpr/SimpleTerm prevent unary nesting.
enum class GrammarSymbol : uint8_t {
    Expression, // Constant*Term + Constant | Constant*Term + Expression
    Term, // RecurringFactor | Term * Term (flattened by Tree::Reduce())
    RecurringFactor, // Variable | unary(SimpleExpr) for each enabled unary NodeType | binary(SimpleExpr, SimpleExpr)
                     // for each enabled binary NodeType
    SimpleExpr, // Constant*SimpleTerm + Constant | Constant*SimpleTerm + SimpleExpr
    SimpleTerm, // Variable | SimpleTerm * SimpleTerm (flattened by Tree::Reduce())
};

struct GrammarSymbols {
    static constexpr auto Count = static_cast<std::size_t>(GrammarSymbol::SimpleTerm) + 1UL;
    static constexpr auto GetIndex(GrammarSymbol s) -> std::size_t { return static_cast<std::size_t>(s); }
};

// A production operand is either a nonterminal or a fixed, non-optimizable constant.
struct ProductionOperand {
    std::optional<GrammarSymbol> Symbol; // nullopt <=> this is a fixed operand
    Operon::Scalar FixedValue {};

    ProductionOperand(GrammarSymbol s)
        : Symbol(s)
    {
    } // NOLINT(*explicit*) - see class comment

    static auto Fixed(Operon::Scalar value) -> ProductionOperand
    {
        ProductionOperand op { GrammarSymbol::Expression };
        op.Symbol.reset();
        op.FixedValue = value;
        return op;
    }

    [[nodiscard]] auto IsFixed() const noexcept -> bool { return !Symbol.has_value(); }
    [[nodiscard]] auto ToNode() const -> Node
    {
        EXPECT(IsFixed());
        auto n = Node::Constant(static_cast<double>(FixedValue));
        n.Optimize = false;
        return n;
    }

    auto operator==(ProductionOperand const& rhs) const noexcept -> bool
    {
        return Symbol == rhs.Symbol && (Symbol.has_value() || FixedValue == rhs.FixedValue);
    }
};

// Recipe for combining operand subtrees.
struct Production {
    // NoBuiltinOp denotes a coercion.
    BuiltinOp Op { NoBuiltinOp };
    // Ordered nonterminal or fixed operands.
    std::vector<ProductionOperand> Operands;
    // Adds an optimizable weight to the first operand.
    bool WeightFirstOperand { false };
    // Adds an optimizable bias.
    bool TrailingConstant { false };
    // Enables symmetric budget splitting for commutative self-combines.
    bool Commutative { true };
    // Fixed scale applied to the production result.
    Operon::Scalar ResultScale { 1 };

    // A coercion has one non-fixed operand and no root operator.
    [[nodiscard]] auto IsCoercion() const noexcept -> bool
    {
        return Op == NoBuiltinOp && Operands.size() == 1 && !Operands.front().IsFixed();
    }
};

// Function vocabulary used by enumeration presets.
enum class EnumerationFunction : uint8_t {
    Add,
    Sub,
    Mul,
    Div,
    Pow,
    Inv,
    Square,
    Cube,
    SqrtAbs,
    LogAbs,
    Exp,
    Sin,
    Log10Abs,
    TenExp,
    Log,
    Sqrt,
    Cbrt,
    Aq,
};

struct EnumerationFunctions {
    static constexpr auto Count = static_cast<std::size_t>(EnumerationFunction::Aq) + 1UL;
    static constexpr auto GetIndex(EnumerationFunction f) -> std::size_t { return static_cast<std::size_t>(f); }
};

using EnumerationFunctionSet = Bitset<EnumerationFunctions::Count>;

constexpr auto ToFunctionSet(EnumerationFunction f) -> EnumerationFunctionSet
{
    EnumerationFunctionSet s {};
    s.Set(EnumerationFunctions::GetIndex(f));
    return s;
}

constexpr auto operator|(EnumerationFunction lhs, EnumerationFunction rhs) -> EnumerationFunctionSet
{
    return ToFunctionSet(lhs) | ToFunctionSet(rhs);
}

constexpr auto operator|(EnumerationFunctionSet lhs, EnumerationFunction rhs) -> EnumerationFunctionSet
{
    return lhs | ToFunctionSet(rhs);
}

constexpr auto operator|=(EnumerationFunctionSet& lhs, EnumerationFunction rhs) -> EnumerationFunctionSet&
{
    lhs = lhs | rhs;
    return lhs;
}

// Canonical name used by CLI parsing and display.
[[nodiscard]] OPERON_EXPORT auto EnumerationFunctionName(EnumerationFunction f) -> std::string_view;

// Parses an exact function name.
[[nodiscard]] OPERON_EXPORT auto ParseEnumerationFunction(std::string_view name) -> std::optional<EnumerationFunction>;

// Built-ins required to evaluate an enumeration function set.
[[nodiscard]] OPERON_EXPORT auto UnderlyingPrimitives(EnumerationFunctionSet functions) -> PrimitiveSetConfig;

// ESR-compatible named function sets; each includes add, sub, mul, div, and pow.
enum class EnumerationPreset : uint8_t {
    KeepDuplicates, // square, exp, inv, sqrt_abs, log_abs
    CoreMaths, // inv
    ExtMaths, // inv, sqrt_abs, square, exp
    OscMaths, // inv, sin
    Base10Maths, // tenexp, inv, log10_abs
    BaseEMaths, // inv, exp, log_abs
};

// Parses a named preset.
[[nodiscard]] OPERON_EXPORT auto ParseEnumerationPreset(std::string_view name) -> std::optional<EnumerationPreset>;

// Returns the complete function set for a preset.
[[nodiscard]] OPERON_EXPORT auto PresetFunctions(EnumerationPreset preset) -> EnumerationFunctionSet;

// Configurable grammar for exhaustive enumeration.
class OPERON_EXPORT Grammar {
public:
    Grammar()
        : Grammar(PrimitiveSetConfig {}, {})
    {
    }
    Grammar(PrimitiveSetConfig enabledFunctions, std::vector<Operon::Hash> variableHashes);

    // Rebuilds from legacy primitive-set configuration.
    auto Configure(PrimitiveSetConfig config) -> Grammar&;

    // Rebuilds from enumeration functions.
    auto Configure(EnumerationFunctionSet functions) -> Grammar&;

    // Rebuilds terminal variables.
    auto SetVariables(std::vector<Operon::Hash> variableHashes) -> Grammar&;

    [[nodiscard]] auto Productions(GrammarSymbol nt) const -> std::span<Production const>
    {
        return rules_.at(GrammarSymbols::GetIndex(nt));
    }

    [[nodiscard]] auto VariableHashes() const -> std::span<Operon::Hash const> { return variables_; }
    // Legacy PrimitiveSetConfig this Grammar was last Configure()'d from -
    // only meaningful after Configure(PrimitiveSetConfig); reads as the
    // default-constructed PrimitiveSetConfig{} after Configure(EnumerationFunctionSet).
    [[nodiscard]] auto Config() const -> PrimitiveSetConfig { return config_; }
    [[nodiscard]] auto Functions() const -> EnumerationFunctionSet { return functions_; }

    // Whether `nt` terminates directly in a Variable leaf (i.e. whether the
    // DP engine's terminal-seeding step should populate this nonterminal's
    // budget-1 bucket with one Variable-leaf tree per VariableHashes() entry).
    [[nodiscard]] auto AllowsVariable(GrammarSymbol nt) const -> bool
    {
        return nt == GrammarSymbol::RecurringFactor || nt == GrammarSymbol::SimpleTerm;
    }

    // Minimum achievable Complexity (see Grammar::MinComplexity's definition:
    // count of all non-Constant nodes) for a nonterminal - a fixed point over
    // the production table, computed once in Configure()/SetVariables(). Used
    // by the DP engine to skip budget/operand splits that could never be
    // satisfied by any derivable expression.
    [[nodiscard]] auto MinComplexity(GrammarSymbol nt) const -> size_t
    {
        return minComplexity_.at(GrammarSymbols::GetIndex(nt));
    }

private:
    void Rebuild();

    std::array<std::vector<Production>, GrammarSymbols::Count> rules_;
    std::array<size_t, GrammarSymbols::Count> minComplexity_ {};
    std::vector<Operon::Hash> variables_;
    PrimitiveSetConfig config_;
    EnumerationFunctionSet functions_;
};

} // namespace Operon

#endif
