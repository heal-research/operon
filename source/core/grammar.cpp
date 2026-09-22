// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/grammar.hpp"

#include <cmath>
#include <limits>
#include <utility>

namespace Operon {

namespace {
    // Every named ESR preset enables these five binary ops on RecurringFactor
    // (RecurringFactor -> Op(SimpleExpr, SimpleExpr)) in addition to its own
    // unary set - see PresetFunctions().
    constexpr EnumerationFunctionSet BinaryFunctions
        = EnumerationFunction::Add | EnumerationFunction::Sub | EnumerationFunction::Mul
        | EnumerationFunction::Div | EnumerationFunction::Pow;

    // RecurringFactor production recipe for one EnumerationFunction value -
    // see grammar.hpp's EnumerationFunction doc comment for the mapping
    // rationale (which existing BuiltinOp each reuses, and why).
    auto RecipeFor(EnumerationFunction f) -> Production
    {
        switch (f) {
        case EnumerationFunction::Add:
            return Production{ .Op = BuiltinOp::Add, .Operands = { GrammarSymbol::SimpleExpr, GrammarSymbol::SimpleExpr }, .Commutative = true };
        case EnumerationFunction::Sub:
            return Production{ .Op = BuiltinOp::Sub, .Operands = { GrammarSymbol::SimpleExpr, GrammarSymbol::SimpleExpr }, .Commutative = false };
        case EnumerationFunction::Mul:
            return Production{ .Op = BuiltinOp::Mul, .Operands = { GrammarSymbol::SimpleExpr, GrammarSymbol::SimpleExpr }, .Commutative = true };
        case EnumerationFunction::Div:
            return Production{ .Op = BuiltinOp::Div, .Operands = { GrammarSymbol::SimpleExpr, GrammarSymbol::SimpleExpr }, .Commutative = false };
        case EnumerationFunction::Pow:
            return Production{ .Op = BuiltinOp::Pow, .Operands = { GrammarSymbol::SimpleExpr, GrammarSymbol::SimpleExpr }, .Commutative = false };
        case EnumerationFunction::Inv: // 1/x - unary Div (see Tree::Simplify's Div-arity-1 handling)
            return Production{ .Op = BuiltinOp::Div, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Square:
            return Production{ .Op = BuiltinOp::Square, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Cube: // x^3 - Pow with a fixed, non-optimizable exponent operand
            return Production{ .Op = BuiltinOp::Pow, .Operands = { GrammarSymbol::SimpleExpr, ProductionOperand::Fixed(Operon::Scalar{3}) } };
        case EnumerationFunction::SqrtAbs:
            return Production{ .Op = BuiltinOp::Sqrtabs, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::LogAbs:
            return Production{ .Op = BuiltinOp::Logabs, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Exp:
            return Production{ .Op = BuiltinOp::Exp, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Sin:
            return Production{ .Op = BuiltinOp::Sin, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Log10Abs: // log10|x| = logabs(x) * (1/ln(10))
            return Production{ .Op = BuiltinOp::Logabs, .Operands = { GrammarSymbol::SimpleExpr }, .ResultScale = static_cast<Operon::Scalar>(1.0 / std::log(10.0)) };
        case EnumerationFunction::TenExp: // 10^x - Pow with a fixed, non-optimizable base operand
            return Production{ .Op = BuiltinOp::Pow, .Operands = { ProductionOperand::Fixed(Operon::Scalar{10}), GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Log:
            return Production{ .Op = BuiltinOp::Log, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Sqrt:
            return Production{ .Op = BuiltinOp::Sqrt, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Cbrt:
            return Production{ .Op = BuiltinOp::Cbrt, .Operands = { GrammarSymbol::SimpleExpr } };
        case EnumerationFunction::Aq: // analytic quotient, x/sqrt(1+y^2) - same numerator/denominator shape as
                                       // Div but no pole (denominator always >= 1); not part of any named preset.
            return Production{ .Op = BuiltinOp::Aq, .Operands = { GrammarSymbol::SimpleExpr, GrammarSymbol::SimpleExpr }, .Commutative = false };
        }
        std::unreachable();
    }

constexpr std::array<std::pair<EnumerationFunction, std::string_view>, EnumerationFunctions::Count> NameTable {{
    { EnumerationFunction::Add, "add" },
    { EnumerationFunction::Sub, "sub" },
    { EnumerationFunction::Mul, "mul" },
    { EnumerationFunction::Div, "div" },
    { EnumerationFunction::Pow, "pow" },
    { EnumerationFunction::Inv, "inv" },
    { EnumerationFunction::Square, "square" },
    { EnumerationFunction::Cube, "cube" },
    { EnumerationFunction::SqrtAbs, "sqrt_abs" },
    { EnumerationFunction::LogAbs, "log_abs" },
    { EnumerationFunction::Exp, "exp" },
    { EnumerationFunction::Sin, "sin" },
    { EnumerationFunction::Log10Abs, "log10_abs" },
    { EnumerationFunction::TenExp, "tenexp" },
    { EnumerationFunction::Log, "log" },
    { EnumerationFunction::Sqrt, "sqrt" },
    { EnumerationFunction::Cbrt, "cbrt" },
    { EnumerationFunction::Aq, "aq" },
}};
} // namespace

auto EnumerationFunctionName(EnumerationFunction f) -> std::string_view
{
    for (auto const& [fn, name] : NameTable) { if (fn == f) { return name; } }
    std::unreachable();
}

auto ParseEnumerationFunction(std::string_view name) -> std::optional<EnumerationFunction>
{
    for (auto const& [fn, candidate] : NameTable) { if (candidate == name) { return fn; } }
    return std::nullopt;
}

auto UnderlyingPrimitives(EnumerationFunctionSet functions) -> PrimitiveSetConfig
{
    PrimitiveSetConfig config = NodeType::Constant | NodeType::Variable;
    functions.ForEach([&](std::size_t bit) {
        switch (static_cast<EnumerationFunction>(bit)) {
        case EnumerationFunction::Add:                                      config |= BuiltinOp::Add; break;
        case EnumerationFunction::Sub:                                      config |= BuiltinOp::Sub; break;
        case EnumerationFunction::Mul:                                      config |= BuiltinOp::Mul; break;
        case EnumerationFunction::Div:      case EnumerationFunction::Inv:  config |= BuiltinOp::Div; break;
        case EnumerationFunction::Pow:      case EnumerationFunction::Cube:
        case EnumerationFunction::TenExp:                                   config |= BuiltinOp::Pow; break;
        case EnumerationFunction::Square:                                   config |= BuiltinOp::Square; break;
        case EnumerationFunction::SqrtAbs:                                  config |= BuiltinOp::Sqrtabs; break;
        case EnumerationFunction::LogAbs:   case EnumerationFunction::Log10Abs:
                                                                             config |= BuiltinOp::Logabs; break;
        case EnumerationFunction::Exp:                                      config |= BuiltinOp::Exp; break;
        case EnumerationFunction::Sin:                                      config |= BuiltinOp::Sin; break;
        case EnumerationFunction::Log:                                      config |= BuiltinOp::Log; break;
        case EnumerationFunction::Sqrt:                                     config |= BuiltinOp::Sqrt; break;
        case EnumerationFunction::Cbrt:                                     config |= BuiltinOp::Cbrt; break;
        case EnumerationFunction::Aq:                                       config |= BuiltinOp::Aq; break;
        }
    });
    return config;
}

auto ParseEnumerationPreset(std::string_view name) -> std::optional<EnumerationPreset>
{
    if (name == "keep_duplicates") { return EnumerationPreset::KeepDuplicates; }
    if (name == "core_maths")      { return EnumerationPreset::CoreMaths; }
    if (name == "ext_maths")       { return EnumerationPreset::ExtMaths; }
    if (name == "osc_maths")       { return EnumerationPreset::OscMaths; }
    if (name == "base10_maths")    { return EnumerationPreset::Base10Maths; }
    if (name == "base_e_maths")    { return EnumerationPreset::BaseEMaths; }
    return std::nullopt;
}

auto PresetFunctions(EnumerationPreset preset) -> EnumerationFunctionSet
{
    EnumerationFunctionSet unary{};
    switch (preset) {
    case EnumerationPreset::KeepDuplicates:
        unary = EnumerationFunction::Square | EnumerationFunction::Exp | EnumerationFunction::Inv
            | EnumerationFunction::SqrtAbs | EnumerationFunction::LogAbs;
        break;
    case EnumerationPreset::CoreMaths:
        unary = ToFunctionSet(EnumerationFunction::Inv);
        break;
    case EnumerationPreset::ExtMaths:
        unary = EnumerationFunction::Inv | EnumerationFunction::SqrtAbs | EnumerationFunction::Square | EnumerationFunction::Exp;
        break;
    case EnumerationPreset::OscMaths:
        unary = EnumerationFunction::Inv | EnumerationFunction::Sin;
        break;
    case EnumerationPreset::Base10Maths:
        unary = EnumerationFunction::TenExp | EnumerationFunction::Inv | EnumerationFunction::Log10Abs;
        break;
    case EnumerationPreset::BaseEMaths:
        unary = EnumerationFunction::Inv | EnumerationFunction::Exp | EnumerationFunction::LogAbs;
        break;
    }
    return unary | BinaryFunctions;
}

Grammar::Grammar(PrimitiveSetConfig enabledFunctions, std::vector<Operon::Hash> variableHashes)
    : variables_(std::move(variableHashes))
{
    Configure(enabledFunctions);
    // Configure() already calls Rebuild(); re-apply variables_ (set above,
    // before functions_ existed) is unnecessary since Rebuild() reads
    // variables_ directly - nothing further to do here.
}

auto Grammar::Configure(PrimitiveSetConfig config) -> Grammar&
{
    config_ = config;
    // Narrow compatibility translation - see the class comment in grammar.hpp:
    // reproduces the pre-existing grammar exactly for Log/Exp/Sin/Sqrt/Cbrt;
    // Aq is a new production not present in the legacy grammar (see the
    // class comment's Aq note), nothing else from the new ESR vocabulary.
    EnumerationFunctionSet functions{};
    auto test = [&](BuiltinOp op) { return config.Test(static_cast<std::size_t>(op)); };
    if (test(BuiltinOp::Log))  { functions |= EnumerationFunction::Log; }
    if (test(BuiltinOp::Exp))  { functions |= EnumerationFunction::Exp; }
    if (test(BuiltinOp::Sin))  { functions |= EnumerationFunction::Sin; }
    if (test(BuiltinOp::Sqrt)) { functions |= EnumerationFunction::Sqrt; }
    if (test(BuiltinOp::Cbrt)) { functions |= EnumerationFunction::Cbrt; }
    if (test(BuiltinOp::Aq))   { functions |= EnumerationFunction::Aq; }
    functions_ = functions;
    Rebuild();
    return *this;
}

auto Grammar::Configure(EnumerationFunctionSet functions) -> Grammar&
{
    functions_ = functions;
    config_ = PrimitiveSetConfig{}; // stale/meaningless once configured this way - see Config()'s doc comment
    Rebuild();
    return *this;
}

auto Grammar::SetVariables(std::vector<Operon::Hash> variableHashes) -> Grammar&
{
    variables_ = std::move(variableHashes);
    Rebuild();
    return *this;
}

void Grammar::Rebuild()
{
    for (auto& r : rules_) { r.clear(); }

    auto& recurringFactor = rules_[GrammarSymbols::GetIndex(GrammarSymbol::RecurringFactor)];
    functions_.ForEach([&](std::size_t bit) {
        recurringFactor.push_back(RecipeFor(static_cast<EnumerationFunction>(bit)));
    });

    rules_[GrammarSymbols::GetIndex(GrammarSymbol::Term)] = {
        Production{ .Op = NoBuiltinOp, .Operands = { GrammarSymbol::RecurringFactor } }, // coercion
        Production{ .Op = BuiltinOp::Mul, .Operands = { GrammarSymbol::Term, GrammarSymbol::Term } },
    };

    rules_[GrammarSymbols::GetIndex(GrammarSymbol::SimpleTerm)] = {
        Production{ .Op = BuiltinOp::Mul, .Operands = { GrammarSymbol::SimpleTerm, GrammarSymbol::SimpleTerm } },
    };

    rules_[GrammarSymbols::GetIndex(GrammarSymbol::Expression)] = {
        Production{ .Op = BuiltinOp::Add, .Operands = { GrammarSymbol::Term }, .WeightFirstOperand = true, .TrailingConstant = true },
        Production{ .Op = BuiltinOp::Add, .Operands = { GrammarSymbol::Term, GrammarSymbol::Expression }, .WeightFirstOperand = true, .TrailingConstant = false },
    };

    rules_[GrammarSymbols::GetIndex(GrammarSymbol::SimpleExpr)] = {
        Production{ .Op = BuiltinOp::Add, .Operands = { GrammarSymbol::SimpleTerm }, .WeightFirstOperand = true, .TrailingConstant = true },
        Production{ .Op = BuiltinOp::Add, .Operands = { GrammarSymbol::SimpleTerm, GrammarSymbol::SimpleExpr }, .WeightFirstOperand = true, .TrailingConstant = false },
    };

    // Recipe validation - defensive, not a live user-facing error path today
    // (every production above is built by this file alone; RecipeFor's
    // switch is exhaustive over EnumerationFunction). Catches a future
    // malformed recipe (wrong operand count for Op's arity, a coercion with
    // more than one operand, more than two nonterminal operands) at the
    // point it's introduced rather than as a silently-wrong enumerated tree.
    for (auto const& row : rules_) {
        for (auto const& p : row) {
            std::size_t nonterminalCount = 0;
            for (auto const& operand : p.Operands) { if (!operand.IsFixed()) { ++nonterminalCount; } }
            EXPECT(nonterminalCount >= 1 && nonterminalCount <= 2);
            if (p.IsCoercion()) {
                EXPECT(p.Operands.size() == 1);
                continue;
            }
            EXPECT(!p.Operands.empty());
            auto const opv = static_cast<Operon::Hash>(p.Op);
            auto const naryMax = static_cast<Operon::Hash>(BuiltinOp::Fmax);
            auto const binMax = static_cast<Operon::Hash>(BuiltinOp::Powabs);
            auto const totalOperands = p.Operands.size() + (p.TrailingConstant ? 1UL : 0UL);
            if (opv > binMax) { // unary
                EXPECT(totalOperands == 1);
            } else if (opv > naryMax) { // strictly binary (Aq/Pow/Powabs)
                EXPECT(totalOperands == 2);
            } else { // n-ary (Add/Mul/Sub/Div/Fmin/Fmax) - any arity >= 1 is valid (Sub/Div arity 1 are
                     // negation/inversion, not identity - see Tree::Simplify)
                EXPECT(totalOperands >= 1);
            }
        }
    }

    // Fixed-point over the production table for MinComplexity. Complexity
    // counts every non-Constant node (see grammar.hpp), so a coercion
    // contributes 0, and any other production contributes: 1 for its Op
    // node, +1 more if WeightFirstOperand (the implicit weight Mul), +1 more
    // if ResultScale != 1 (the implicit scale Mul) - fixed operands (see
    // ProductionOperand) are Constant leaves and never contribute.
    constexpr auto Unreachable = std::numeric_limits<size_t>::max();
    minComplexity_.fill(Unreachable);

    if (!variables_.empty()) {
        // Both nonterminals here return true from AllowsVariable() by
        // construction (see grammar.hpp) - no guard needed.
        for (auto nt : { GrammarSymbol::RecurringFactor, GrammarSymbol::SimpleTerm }) {
            minComplexity_[GrammarSymbols::GetIndex(nt)] = 1;
        }
    }

    for (bool changed = true; changed;) {
        changed = false;
        for (std::size_t i = 0; i < GrammarSymbols::Count; ++i) {
            for (auto const& p : rules_[i]) {
                bool reachable = true;
                size_t total = p.IsCoercion() ? 0 : (1
                    + (p.WeightFirstOperand ? 1 : 0)
                    + (p.ResultScale != Operon::Scalar{1} ? 1 : 0));
                for (auto const& operand : p.Operands) {
                    if (operand.IsFixed()) { continue; } // Constant leaf, contributes 0
                    auto c = minComplexity_[GrammarSymbols::GetIndex(*operand.Symbol)];
                    if (c == Unreachable) { reachable = false; break; }
                    total += c;
                }
                if (reachable && total < minComplexity_[i]) {
                    minComplexity_[i] = total;
                    changed = true;
                }
            }
        }
    }
}

} // namespace Operon
