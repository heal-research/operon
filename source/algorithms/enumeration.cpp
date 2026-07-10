// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/enumeration.hpp"

#include <algorithm>
#include <array>

namespace Operon {

auto SymbolicComplexity(Operon::Tree const& tree) noexcept -> std::size_t
{
    auto const& nodes = tree.Nodes();
    return static_cast<std::size_t>(std::ranges::count_if(nodes, [](auto const& n) { return !n.IsConstant(); }));
}

namespace {
    // Dependency order within a single budget level: each nonterminal here
    // only ever reads buckets of nonterminals earlier in this list (at the
    // same or a smaller budget) or of itself at a strictly smaller budget -
    // never a nonterminal later in the list at the same budget. This is what
    // makes it safe to fully complete one nonterminal's candidate generation
    // (including any Simplify()-driven "shrink into a lower bucket") before
    // moving on to the next.
    constexpr std::array ProcessingOrder {
        GrammarSymbol::SimpleTerm,
        GrammarSymbol::SimpleExpr,
        GrammarSymbol::RecurringFactor,
        GrammarSymbol::Term,
        GrammarSymbol::Expression,
    };

    auto AppendNodes(Operon::Vector<Node>& out, Operon::Vector<Node> const& in) -> void {
        out.insert(out.end(), in.begin(), in.end());
    }

    // Placeholder values for the not-yet-fit weight/bias Constants introduced
    // by WeightFirstOperand/TrailingConstant. Deliberately NOT 1.0 (Mul's
    // identity) / 0.0 (Add's identity): Tree::Simplify() removes x*1 and x+0
    // nodes, so using the identity values here would make Simplify() strip
    // away the very Mul/Add "glue" a not-yet-fitted weight/bias needs to
    // survive on, before coefficient fitting ever gets a chance to explore a
    // non-trivial value. Content-hash ignores Constant leaf values entirely
    // (see hash/content_hash.hpp), so this choice has no effect on dedup.
    constexpr Operon::Scalar WeightPlaceholder{2.0};
    constexpr Operon::Scalar BiasPlaceholder{1.0};

    // How many nominal budget levels beyond maxComplexity_ the DP must search
    // to discover every tree whose *realized* complexity is exactly
    // maxComplexity_ (see enumeration.hpp's "Budget accounting note"). Derived
    // by hand: peeling off one operand at a time (b0=1, b1=target-1) always
    // has overshoot exactly 1 for both the Mul self-combine (one operand,
    // the size->=2 side, already Mul-rooted) and the Add recursion (the
    // Expression operand is always Add-rooted) - and that overshoot doesn't
    // compound across recursive steps, since each step reads its operand from
    // the operand's own already-shrunk bucket, not a carried-over nominal
    // value. Verified empirically by the SimpleTerm closed-form completeness
    // test in test/source/implementation/enumeration.cpp.
    constexpr std::size_t WorkingBudgetMargin = 1;
} // namespace

EnumerationEngine::EnumerationEngine(Operon::Grammar grammar, std::size_t maxComplexity, Operon::RandomGenerator& rng)
    : grammar_(std::move(grammar))
    , maxComplexity_(maxComplexity)
    , workingCeiling_(maxComplexity_ + WorkingBudgetMargin)
    , zobrist_(rng, /*maxLength=*/1, grammar_.VariableHashes())
{
    buckets_.resize(GrammarSymbols::Count);
    seen_.resize(GrammarSymbols::Count);
    for (auto& row : buckets_) { row.resize(workingCeiling_ + 1); }
    for (auto& row : seen_) { row.resize(workingCeiling_ + 1); }
}

auto EnumerationEngine::Bucket(GrammarSymbol nt, std::size_t budget) const -> std::span<Operon::Tree const>
{
    return buckets_[GrammarSymbols::GetIndex(nt)][budget];
}

auto EnumerationEngine::TryInsert(GrammarSymbol nt, Operon::Tree tree) -> bool
{
    tree.Reduce();
    tree.Simplify();
    auto complexity = SymbolicComplexity(tree);
    if (complexity == 0 || complexity > maxComplexity_) { return false; }

    auto idx = GrammarSymbols::GetIndex(nt);
    auto hash = ComputeContentHash(tree, zobrist_);

    bool const novel = seen_[idx][complexity].lazy_emplace_l(
        hash,
        [](auto&) { /* already present, nothing to update */ },
        [&](auto const& ctor) { ctor(hash); }
    );
    if (novel) {
        buckets_[idx][complexity].push_back(std::move(tree));
    }
    return novel;
}

void EnumerationEngine::SeedTerminals()
{
    for (auto varHash : grammar_.VariableHashes()) {
        Node n(NodeType::Variable);
        n.HashValue = varHash;
        for (auto nt : { GrammarSymbol::RecurringFactor, GrammarSymbol::SimpleTerm }) {
            Tree t = Tree({ n }).UpdateNodes();
            TryInsert(nt, std::move(t));
        }
    }
}

void EnumerationEngine::ProcessNonterminal(GrammarSymbol nt, std::size_t budget)
{
    for (auto const& p : grammar_.Productions(nt)) {
        if (p.IsCoercion()) {
            // Single operand, no new node appended: same budget as the operand.
            auto operandIdx = GrammarSymbols::GetIndex(p.Operands.front());
            for (auto const& t : buckets_[operandIdx][budget]) {
                TryInsert(nt, t);
            }
            continue;
        }

        std::size_t const fixedCost = 1UL + (p.WeightFirstOperand ? 1UL : 0UL); // Op node + optional weight Mul node
        if (budget <= fixedCost) { continue; }
        auto const remaining = budget - fixedCost;

        if (p.Operands.size() == 1) {
            auto const operand = p.Operands.front();
            if (remaining < grammar_.MinComplexity(operand)) { continue; }
            auto operandIdx = GrammarSymbols::GetIndex(operand);
            for (auto const& t : buckets_[operandIdx][remaining]) {
                Operon::Vector<Node> nodes;
                if (p.WeightFirstOperand) { nodes.push_back(Node::Constant(WeightPlaceholder)); }
                AppendNodes(nodes, t.Nodes());
                if (p.WeightFirstOperand) { nodes.push_back(Node(NodeType::Mul)); }
                if (p.TrailingConstant) { nodes.push_back(Node::Constant(BiasPlaceholder)); }
                nodes.push_back(Node(p.Op));
                TryInsert(nt, Tree(std::move(nodes)).UpdateNodes());
            }
        } else {
            auto const op0 = p.Operands[0];
            auto const op1 = p.Operands[1];
            auto const idx0 = GrammarSymbols::GetIndex(op0);
            auto const idx1 = GrammarSymbols::GetIndex(op1);
            auto const min0 = grammar_.MinComplexity(op0);
            auto const min1 = grammar_.MinComplexity(op1);
            // A self-combine (e.g. Term*Term) relies on Tree::Reduce() to
            // flatten the result, so (b0,b1) and (b1,b0) enumerate the same
            // final shapes - skip the symmetric half to avoid redundant work
            // (harmless either way, since TryInsert would just dedup them,
            // but there's no reason to pay for it twice).
            bool const selfCombineUnweighted = (op0 == op1) && !p.WeightFirstOperand;

            for (std::size_t b0 = min0; b0 <= remaining; ++b0) {
                if (remaining - b0 < min1) { continue; }
                auto const b1 = remaining - b0;
                if (selfCombineUnweighted && b0 > b1) { continue; }

                for (auto const& t0 : buckets_[idx0][b0]) {
                    for (auto const& t1 : buckets_[idx1][b1]) {
                        Operon::Vector<Node> nodes;
                        if (p.WeightFirstOperand) { nodes.push_back(Node::Constant(WeightPlaceholder)); }
                        AppendNodes(nodes, t0.Nodes());
                        if (p.WeightFirstOperand) { nodes.push_back(Node(NodeType::Mul)); }
                        AppendNodes(nodes, t1.Nodes());
                        nodes.push_back(Node(p.Op));
                        TryInsert(nt, Tree(std::move(nodes)).UpdateNodes());
                    }
                }
            }
        }
    }
}

void EnumerationEngine::Build()
{
    SeedTerminals();
    // Searches up to workingCeiling_ (> maxComplexity_) so combinations whose
    // *nominal* budget overshoots maxComplexity_ but whose realized (post-
    // Reduce()) complexity doesn't still get tried - see WorkingBudgetMargin.
    // TryInsert's complexity check guarantees nothing is actually stored past
    // maxComplexity_, so the caller-visible ceiling is unaffected.
    for (std::size_t budget = 1; budget <= workingCeiling_; ++budget) {
        for (auto nt : ProcessingOrder) {
            ProcessNonterminal(nt, budget);
        }
    }
}

} // namespace Operon
