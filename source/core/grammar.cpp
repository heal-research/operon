// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/grammar.hpp"

#include <limits>

namespace Operon {

namespace {
    // Unary ops that may wrap a SimpleExpr as a RecurringFactor, gated
    // by whether each is enabled in the configured PrimitiveSetConfig.
    // Mirrors symreg-cpp's grammar.cpp (LogFactor/ExpFactor/SinFactor/
    // SqrtFactor/CbrtFactor); InvFactor is intentionally not reproduced (see
    // grammar.hpp's GrammarSymbol comment).
    constexpr std::array UnaryWraps { BuiltinOp::Log, BuiltinOp::Exp, BuiltinOp::Sin, BuiltinOp::Sqrt, BuiltinOp::Cbrt };
} // namespace

Grammar::Grammar(PrimitiveSetConfig enabledFunctions, std::vector<Operon::Hash> variableHashes)
    : variables_(std::move(variableHashes))
    , config_(enabledFunctions)
{
    Rebuild();
}

auto Grammar::Configure(PrimitiveSetConfig config) -> Grammar&
{
    config_ = config;
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
    for (auto op : UnaryWraps) {
        if (config_.Test(static_cast<std::size_t>(op))) {
            recurringFactor.push_back(Production{ .Op = op, .Operands = { GrammarSymbol::SimpleExpr } });
        }
    }

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

    // Fixed-point over the production table for MinComplexity. Complexity
    // counts every non-Constant node (see grammar.hpp), so a coercion or a
    // weighted-operand production contributes: 1 per non-coercion Op node,
    // plus 1 more for the implicit Mul introduced by WeightFirstOperand
    // (TrailingConstant never adds complexity - constants are excluded).
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
                size_t total = p.IsCoercion() ? 0 : (1 + (p.WeightFirstOperand ? 1 : 0));
                for (auto operand : p.Operands) {
                    auto c = minComplexity_[GrammarSymbols::GetIndex(operand)];
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
