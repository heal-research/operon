// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CORE_GRAMMAR_HPP
#define OPERON_CORE_GRAMMAR_HPP

#include <array>
#include <span>
#include <vector>

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
// No InvFactor/InvExpr/InvTerm (1/x reciprocal wrapper): no corresponding operon NodeType or production.
// Productions cover Add/Mul/Aq plus the five unary wraps below; Div/Sub/Pow are unreachable. Production cannot
// express a fixed-constant operand (e.g. Div's numerator) mixed with a nonterminal operand.
// Aq (analytic quotient, x/sqrt(1+y^2)): same numerator/denominator shape as Div, no pole (denominator >= 1),
// fits the existing two-operand production shape.
enum class GrammarSymbol : uint8_t {
    Expression,      // Constant*Term + Constant | Constant*Term + Expression
    Term,            // RecurringFactor | Term * Term (flattened by Tree::Reduce())
    RecurringFactor, // Variable | unary(SimpleExpr) for each enabled unary NodeType
    SimpleExpr,      // Constant*SimpleTerm + Constant | Constant*SimpleTerm + SimpleExpr
    SimpleTerm,      // Variable | SimpleTerm * SimpleTerm (flattened by Tree::Reduce())
};

struct GrammarSymbols {
    static constexpr auto Count = static_cast<std::size_t>(GrammarSymbol::SimpleTerm) + 1UL;
    static constexpr auto GetIndex(GrammarSymbol s) -> std::size_t { return static_cast<std::size_t>(s); }
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
    // Nonterminal categories combined, in order. A self-combine (e.g.
    // {Term, Term} for Op=Mul) relies on Tree::Reduce() to flatten the
    // resulting nested Mul into one flat n-ary node, so it enumerates the
    // same language as a strictly right-recursive "Factor * Term" rule would,
    // via the more DP-natural "split the budget between two same-category
    // operands" recurrence.
    std::vector<GrammarSymbol> Operands;
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
    // Whether Op is commutative for a same-symbol two-operand production (Operands[0] == Operands[1]).
    // ProcessNonterminal skips the b0 > b1 half as redundant work when true. Default true (matches Add, Mul).
    // MUST be false for a non-commutative same-symbol Op (e.g. Aq: aq(a, b) != aq(b, a)) - otherwise the skip
    // drops every candidate whose first operand comes from the larger-budget bucket.
    bool Commutative{true};

    [[nodiscard]] auto IsCoercion() const noexcept -> bool { return Op == NoBuiltinOp; }
};


// Queryable, config/dataset-parameterized grammar for exhaustive expression
// enumeration - plays the same role for the enumeration algorithm that
// PrimitiveSet plays for stochastic tree generation.
//
// Configure()'s PrimitiveSetConfig gates only Log/Exp/Sin/Sqrt/Cbrt (UnaryWraps, see grammar.cpp) and Aq, all on
// RecurringFactor - not a general "enable any PrimitiveSet function" switch. TypeCoherent reaches Log/Exp/Sin;
// Full additionally reaches Sqrt/Cbrt/Aq. Pow, Cos, Square, Tan, Tanh, Sub, Div are never produced regardless
// of config (see GrammarSymbol comment above).
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
    // Problem::GetPrimitiveSet().Config().
    auto Configure(PrimitiveSetConfig config) -> Grammar&;

    // (Re)builds which variable hashes seed RecurringFactor's/SimpleTerm's
    // terminal (budget-1) buckets, without touching function-derived rules.
    auto SetVariables(std::vector<Operon::Hash> variableHashes) -> Grammar&;

    [[nodiscard]] auto Productions(GrammarSymbol nt) const -> std::span<Production const> {
        return rules_.at(GrammarSymbols::GetIndex(nt));
    }

    [[nodiscard]] auto VariableHashes() const -> std::span<Operon::Hash const> { return variables_; }
    [[nodiscard]] auto Config() const -> PrimitiveSetConfig { return config_; }

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
};

} // namespace Operon

#endif
