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

// Nonterminal categories for exhaustive grammar enumeration, reproducing the
// language of symreg-cpp's grammar (a sum of weighted products of "recurring
// factors", where each unary transcendental wraps a shallow, non-recursive
// sub-grammar) using operon's NodeType vocabulary. See grammar.cpp for the
// concrete production list.
//
// SimpleExpr/SimpleTerm are the cut-down sub-grammar used inside unary-
// function arguments: SimpleTerm bottoms out at bare variables only (no
// RecurringFactor/unary recursion), which is what actually keeps unary
// nesting shallow (e.g. no log(sin(...))) - not a restriction on how many
// weighted terms SimpleExpr itself may sum.
//
// Deviates from symreg-cpp in one respect: symreg-cpp's InvFactor/InvExpr/
// InvTerm (a dedicated 1/x reciprocal wrapper) has no corresponding operon
// NodeType, and no corresponding production either - operon instead reuses
// unary Div (arity-1 Node::Div means 1/x, see interpreter/functions.hpp) for
// the same role, wired up alongside Sub/Pow/Add/Mul/Aq as RecurringFactor
// binary/unary productions (see RecipeFor in grammar.cpp) once a
// ProductionOperand supplies the fixed non-optimizable second operand a
// production like Cube (Pow(arg, 3)) or TenExp (Pow(10, arg)) needs -
// Production::Operands can hold either a nonterminal or a fixed Constant
// slot (see ProductionOperand below). This is only reachable via
// Configure(EnumerationFunctionSet); Configure(PrimitiveSetConfig) remains a
// narrow legacy-behavior-preserving shim (see its own doc comment) that
// never wires Sub/Div/Pow/Inv/Cube/etc.
enum class GrammarSymbol : uint8_t {
    Expression,      // Constant*Term + Constant | Constant*Term + Expression
    Term,            // RecurringFactor | Term * Term (flattened by Tree::Reduce())
    RecurringFactor, // Variable | unary(SimpleExpr) for each enabled unary NodeType | binary(SimpleExpr, SimpleExpr) for each enabled binary NodeType
    SimpleExpr,      // Constant*SimpleTerm + Constant | Constant*SimpleTerm + SimpleExpr
    SimpleTerm,      // Variable | SimpleTerm * SimpleTerm (flattened by Tree::Reduce())
};

struct GrammarSymbols {
    static constexpr auto Count = static_cast<std::size_t>(GrammarSymbol::SimpleTerm) + 1UL;
    static constexpr auto GetIndex(GrammarSymbol s) -> std::size_t { return static_cast<std::size_t>(s); }
};

// One operand slot in a Production's Operands list: either a nonterminal
// (a lower-budget subtree drawn from that GrammarSymbol's DP bucket) or a
// fixed, non-optimizable Node::Constant spliced in verbatim (e.g. the "3" in
// Pow(arg, 3) for cube, or the "10" in Pow(10, arg) for tenexp). A fixed
// operand contributes zero to SymbolicComplexity (Constants are excluded,
// see enumeration.hpp) and is never touched by coefficient fitting
// (Node::Constant's Optimize flag is forced false in ToNode() below).
//
// Implicit construction from GrammarSymbol keeps every existing production
// literal (e.g. `.Operands = { GrammarSymbol::Term }`) compiling unchanged;
// use ProductionOperand::Fixed(value) for a fixed-constant slot.
struct ProductionOperand {
    std::optional<GrammarSymbol> Symbol; // nullopt <=> this is a fixed operand
    Operon::Scalar FixedValue{};

    ProductionOperand(GrammarSymbol s) : Symbol(s) { } // NOLINT(*explicit*) - see class comment

    static auto Fixed(Operon::Scalar value) -> ProductionOperand
    {
        ProductionOperand op{ GrammarSymbol::Expression };
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

// A production is a recipe for building one new Tree from already-built
// operand subtrees (drawn from lower-budget DP buckets - see
// algorithms/enumeration.hpp), not a literal CFG string-rewrite rule: the
// enumeration engine only ever expands one level at a time, so there is no
// need for symreg-cpp's generic grammar-string worklist machinery.
struct Production {
    // Operator Node appended when combining Operands. NoBuiltinOp marks
    // a pure coercion (e.g. Term -> RecurringFactor): the operand's tree is
    // used as-is, no new node is appended, and no complexity is added.
    BuiltinOp Op{NoBuiltinOp};
    // Nonterminal/fixed-constant operands combined, in order - see
    // ProductionOperand. A self-combine (e.g. {Term, Term} for Op=Mul)
    // relies on Tree::Reduce() to flatten the resulting nested Mul into one
    // flat n-ary node, so it enumerates the same language as a strictly
    // right-recursive "Factor * Term" rule would, via the more DP-natural
    // "split the budget between two same-category operands" recurrence.
    std::vector<ProductionOperand> Operands;
    // Whether the first Operand gets an implicit "* Constant" (a free,
    // optimizable weight) prepended before combining via Op. Only ever true
    // for Expression/SimpleExpr's Term/SimpleTerm operand - the recursive
    // continuation operand (Expression/SimpleExpr itself) is never
    // reweighted, since it will contribute its own weight further down.
    bool WeightFirstOperand{false};
    // Whether an implicit trailing "+ Constant" bias leaf is appended as one
    // more Add operand. Affects Tree::Length() but not Complexity (constants
    // are excluded from the complexity count - see Grammar::MinComplexity).
    bool TrailingConstant{false};
    // Whether Op is commutative in the sense enumeration.cpp's ProcessNonterminal
    // cares about: for a same-symbol two-operand production (Operands[0] ==
    // Operands[1], e.g. Term*Term), swapping which bucket contributes b0 vs b1
    // yields the same set of realized trees, so ProcessNonterminal skips the
    // b0 > b1 half as redundant work. True by default (matches every op this
    // grammar produced before Aq: Add, Mul). MUST be set false for a
    // same-symbol production whose Op isn't actually commutative (e.g. Aq:
    // aq(a, b) != aq(b, a)) - otherwise that skip silently drops every
    // candidate whose first operand comes from the larger-budget bucket.
    bool Commutative{true};
    // Fixed scalar multiplier applied to the emitted root operator's result
    // (e.g. 1/ln(10) for log10_abs = Logabs(arg) * (1/ln(10))). Default 1
    // means "no wrapping": ResultScale != 1 makes ProcessNonterminal append
    // one more non-optimizable Constant(ResultScale) + Mul on top of the
    // production's own Op node. Only meaningful on a non-coercion production
    // (IsCoercion() productions never emit an Op node to scale).
    Operon::Scalar ResultScale{1};

    // A coercion has exactly one nonterminal operand and emits no operator
    // node at all (the operand's own tree is used as-is) - e.g. Term ->
    // RecurringFactor. Not merely "Op == NoBuiltinOp": a malformed recipe
    // with NoBuiltinOp but a fixed operand or more than one operand isn't a
    // valid coercion either (see Grammar::Rebuild's recipe validation).
    [[nodiscard]] auto IsCoercion() const noexcept -> bool
    {
        return Op == NoBuiltinOp && Operands.size() == 1 && !Operands.front().IsFixed();
    }
};

// One built-in math function/operator the ESR-parity `--function-set`
// vocabulary can enable on RecurringFactor, in addition to the fixed
// weighted-sum Expression/Term shape every Grammar always builds (see
// GrammarSymbol above). Each maps to one RecurringFactor Production in
// Grammar::Rebuild (see grammar.cpp) - binary Add/Sub/Mul/Div/Pow combine
// two SimpleExpr operands (the same shape Aq already used); the remaining
// values are unary SimpleExpr wraps, except Cube/TenExp (one SimpleExpr
// operand plus one fixed-constant operand) and Log10Abs (a Logabs wrap with
// ResultScale = 1/ln(10)). No new BuiltinOp values are introduced: every
// entry reuses an existing built-in (Inv is arity-1 Div, i.e. 1/x - see
// Tree::Simplify's Div-arity-1 handling in source/core/tree.cpp).
enum class EnumerationFunction : uint8_t {
    Add, Sub, Mul, Div, Pow,
    Inv, Square, Cube, SqrtAbs, LogAbs, Exp, Sin,
    Log10Abs, TenExp,
    Log, Sqrt, Cbrt, Aq,
};

struct EnumerationFunctions {
    static constexpr auto Count = static_cast<std::size_t>(EnumerationFunction::Aq) + 1UL;
    static constexpr auto GetIndex(EnumerationFunction f) -> std::size_t { return static_cast<std::size_t>(f); }
};

using EnumerationFunctionSet = Bitset<EnumerationFunctions::Count>;

constexpr auto ToFunctionSet(EnumerationFunction f) -> EnumerationFunctionSet
{
    EnumerationFunctionSet s{};
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

// Canonical ESR-parity name for an EnumerationFunction, used by both --enable-symbols/--disable-symbols
// parsing (see ParseEnumerationFunction) and --show-primitives output. The six values already reachable
// pre-#218 (Log/Exp/Sin/Sqrt/Cbrt/Aq) keep their existing PrimitiveSet symbol spelling for continuity;
// the rest use ESR's own names (e.g. "sqrt_abs"/"log_abs", distinct from the plain PrimitiveSet
// "sqrtabs"/"logabs" spelling, since those name a different production - a RecurringFactor wrap, not a
// generic interpreter primitive).
[[nodiscard]] OPERON_EXPORT auto EnumerationFunctionName(EnumerationFunction f) -> std::string_view;

// Inverse of EnumerationFunctionName - case-sensitive exact match. Returns nullopt for any other input.
[[nodiscard]] OPERON_EXPORT auto ParseEnumerationFunction(std::string_view name) -> std::optional<EnumerationFunction>;

// The PrimitiveSetConfig covering every BuiltinOp an evaluator needs enabled to interpret a tree built
// from `functions` (e.g. Cube/TenExp both need BuiltinOp::Pow; Inv needs BuiltinOp::Div; Log10Abs needs
// BuiltinOp::Logabs) plus NodeType::Constant/Variable (every enumerated tree has both regardless of
// `functions`). Does not include BuiltinOp::Add/Mul (the always-on Expression/Term weighted-sum shape) -
// callers combine this with their own baseline (e.g. PrimitiveSet::Arithmetic) via `|`.
[[nodiscard]] OPERON_EXPORT auto UnderlyingPrimitives(EnumerationFunctionSet functions) -> PrimitiveSetConfig;

// ESR's named unary-function-set presets (Bartlett et al. 2023,
// arXiv:2211.11461). Every preset additionally enables the five binary ops
// {Add, Sub, Mul, Div, Pow} on RecurringFactor (see PresetFunctions) - Aq is
// deliberately never part of a named preset (it has no ESR counterpart; it
// remains reachable only via a `custom` EnumerationFunctionSet).
enum class EnumerationPreset : uint8_t {
    KeepDuplicates, // square, exp, inv, sqrt_abs, log_abs
    CoreMaths,      // inv
    ExtMaths,       // inv, sqrt_abs, square, exp
    OscMaths,       // inv, sin
    Base10Maths,    // tenexp, inv, log10_abs
    BaseEMaths,     // inv, exp, log_abs
};

// Parses one of the exact ESR preset names ("keep_duplicates", "core_maths",
// "ext_maths", "osc_maths", "base10_maths", "base_e_maths"). Returns
// nullopt for any other input, including "custom" (custom is not a preset -
// it means "build the set from --enable-symbols/--disable-symbols instead",
// which has no EnumerationPreset value at all).
[[nodiscard]] OPERON_EXPORT auto ParseEnumerationPreset(std::string_view name) -> std::optional<EnumerationPreset>;

// The full EnumerationFunctionSet for a named preset (its unary functions
// plus the five always-on binary ops), suitable for Grammar::Configure.
[[nodiscard]] OPERON_EXPORT auto PresetFunctions(EnumerationPreset preset) -> EnumerationFunctionSet;

// Queryable, config/dataset-parameterized grammar for exhaustive expression
// enumeration - plays the same role for the enumeration algorithm that
// PrimitiveSet plays for stochastic tree generation.
//
// The Expression/Term/SimpleExpr/SimpleTerm production shapes (the "sum of
// weighted products of recurring factors" outer grammar) are always built
// exactly the same way, in both Configure() overloads below - only
// RecurringFactor's production set is functions_-dependent.
// Configure(PrimitiveSetConfig) is a compatibility shim for existing
// library callers: it translates six PrimitiveSetConfig bits (Log, Exp,
// Sin, Sqrt, Cbrt, Aq) into the equivalent EnumerationFunctionSet bits and
// rebuilds from that - it does not enable Square/Inv/Cube/etc. For every
// bit except Aq this reproduces the exact pre-EnumerationFunctionSet
// grammar (only Log/Exp/Sin/Sqrt/Cbrt were ever wired onto RecurringFactor
// before this addition). Aq is the one exception: the legacy grammar had no
// Aq production at all (Production had no way to express Aq's two-
// SimpleExpr-operand shape), so any PrimitiveSetConfig with the Aq bit set
// now reaches a strictly larger language (a new binary Aq(SimpleExpr,
// SimpleExpr) production) than before this change - not a behavior-
// preservation guarantee for that one bit, but filling a real prior gap.
class OPERON_EXPORT Grammar {
public:
    // Equivalent to Grammar(PrimitiveSetConfig{}, {}): Rebuild() still runs, so
    // MinComplexity() consistently reports Unreachable everywhere rather than
    // leaving minComplexity_ zero-initialized (which would misreport every
    // nonterminal as trivially achievable at complexity 0).
    Grammar() : Grammar(PrimitiveSetConfig{}, {}) {}
    Grammar(PrimitiveSetConfig enabledFunctions, std::vector<Operon::Hash> variableHashes);

    // (Re)builds the unary-wrap productions on RecurringFactor from a
    // PrimitiveSetConfig bitset - the same bitset type as
    // PrimitiveSet::Config(), so a Grammar can be constructed straight from
    // Problem::GetPrimitiveSet().Config(). See the class comment: this is a
    // narrow compatibility translation, not a general "enable any
    // PrimitiveSet function" switch.
    auto Configure(PrimitiveSetConfig config) -> Grammar&;

    // (Re)builds RecurringFactor directly from an EnumerationFunctionSet -
    // the `--function-set` entry point (both named presets via
    // PresetFunctions() and `custom` via a caller-assembled set).
    auto Configure(EnumerationFunctionSet functions) -> Grammar&;

    // (Re)builds which variable hashes seed RecurringFactor's/SimpleTerm's
    // terminal (budget-1) buckets, without touching function-derived rules.
    auto SetVariables(std::vector<Operon::Hash> variableHashes) -> Grammar&;

    [[nodiscard]] auto Productions(GrammarSymbol nt) const -> std::span<Production const> {
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
    [[nodiscard]] auto AllowsVariable(GrammarSymbol nt) const -> bool {
        return nt == GrammarSymbol::RecurringFactor || nt == GrammarSymbol::SimpleTerm;
    }

    // Minimum achievable Complexity (see Grammar::MinComplexity's definition:
    // count of all non-Constant nodes) for a nonterminal - a fixed point over
    // the production table, computed once in Configure()/SetVariables(). Used
    // by the DP engine to skip budget/operand splits that could never be
    // satisfied by any derivable expression.
    [[nodiscard]] auto MinComplexity(GrammarSymbol nt) const -> size_t {
        return minComplexity_.at(GrammarSymbols::GetIndex(nt));
    }

private:
    void Rebuild();

    std::array<std::vector<Production>, GrammarSymbols::Count> rules_;
    std::array<size_t, GrammarSymbols::Count> minComplexity_{};
    std::vector<Operon::Hash> variables_;
    PrimitiveSetConfig config_;
    EnumerationFunctionSet functions_;
};

} // namespace Operon

#endif
