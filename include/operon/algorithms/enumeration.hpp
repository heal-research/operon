// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_ENUMERATION_HPP
#define OPERON_ALGORITHMS_ENUMERATION_HPP

#include <span>
#include <vector>

#include <gtl/phmap.hpp>

#include "operon/core/grammar.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/content_hash.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operon_export.hpp"
#include "operon/random/random.hpp"

namespace Operon {

// Complexity for grammar enumeration: count of all non-Constant nodes
// (variables + every operator, unary and n-ary alike). Deviates slightly from
// symreg-cpp (which excludes bare Add/Mul "glue" from the count) but is
// simple, well-defined directly on a Tree, and serves the same pruning
// intent - free-weight/bias Constants (see Grammar's WeightFirstOperand/
// TrailingConstant) never contribute, since they're optimized values, not
// distinct structural symbols.
[[nodiscard]] OPERON_EXPORT auto Complexity(Operon::Tree const& tree) noexcept -> std::size_t;

// Bottom-up dynamic-programming enumeration engine: builds, for each grammar
// nonterminal and each complexity budget 1..maxComplexity, the set of
// canonical (Reduce()+Simplify()'d, content-hash-deduplicated) trees
// derivable as that nonterminal within that budget - by combining
// already-built, already-deduplicated smaller trees, per Grammar's
// production table.
//
// This phase does NOT fit coefficients (every Constant leaf keeps its
// construction-time placeholder value, Optimize=true) - CoefficientOptimizer
// integration and the top-level Run()/stop-condition driver are a separate,
// later addition (see the project plan). What's here is fully testable on
// its own: that productions combine into valid Trees, that Reduce/Simplify
// interact correctly with the budget accounting, and that content-hash dedup
// collapses duplicates reached via different derivation paths.
//
// Budget accounting note: combining two operands' node counts plus a fixed
// per-Op cost is only an accurate prediction of the *realized* (post-
// Reduce()) complexity when neither operand's own root is already the same
// Op being applied. Term/SimpleTerm's Mul self-combine and Expression/
// SimpleExpr's Add(Term, Expression) recursion both violate this (Reduce()
// merges the new Op into an operand's pre-existing same-type root instead of
// adding a distinct node), so the naive per-combination budget overshoots the
// true complexity by a small constant - see WorkingBudgetMargin in
// enumeration.cpp for how the DP compensates.
//
// Thread-safety: the per-(nonterminal, budget) dedup sets are
// gtl::parallel_flat_hash_set_m (the same primitive ZobristCache uses for its
// transposition cache, see hash/zobrist.hpp), so concurrent candidate
// generation can check-and-insert without a global lock - even though this
// phase's Build() only ever runs single-threaded; parallelizing the
// candidate-generation loop itself is later work.
class OPERON_EXPORT EnumerationEngine {
public:
    EnumerationEngine(Operon::Grammar grammar, std::size_t maxComplexity, Operon::RandomGenerator& rng);

    // Runs the bottom-up construction for budgets 1..maxComplexity in order.
    void Build();

    [[nodiscard]] auto Bucket(GrammarSymbol nt, std::size_t budget) const -> std::span<Operon::Tree const>;

    [[nodiscard]] auto GetGrammar() const -> Operon::Grammar const& { return grammar_; }
    [[nodiscard]] auto MaxComplexity() const -> std::size_t { return maxComplexity_; }

private:
    // Seeds RecurringFactor[1] and SimpleTerm[1] with one Variable-leaf Tree
    // per Grammar::VariableHashes() entry - the only way either nonterminal
    // terminates directly (see Grammar::AllowsVariable).
    void SeedTerminals();

    // Applies every Production of `nt` at `budget`, building candidate Trees
    // from already-completed lower-budget buckets (and, for Term's coercion
    // from RecurringFactor, the same-budget bucket of a nonterminal processed
    // earlier in this level - see the fixed per-level order in Build()).
    void ProcessNonterminal(GrammarSymbol nt, std::size_t budget);

    // Reduce()+Simplify()s `tree`, computes its realized Complexity (which
    // can only be <= the budget it was built for - simplification never adds
    // nodes) and content hash, and inserts it into nt's bucket at that
    // realized complexity if not already present there. Returns whether it
    // was novel (i.e. actually inserted).
    auto TryInsert(GrammarSymbol nt, Operon::Tree tree) -> bool;

    Operon::Grammar grammar_;
    std::size_t maxComplexity_;
    // Internal working budget ceiling, strictly >= maxComplexity_ - see
    // WorkingBudgetMargin in enumeration.cpp for why the DP needs to search
    // beyond the caller-visible ceiling. buckets_/seen_ are sized to this,
    // not to maxComplexity_; rows beyond maxComplexity_ are scratch space
    // that TryInsert's complexity check guarantees stays empty.
    std::size_t workingCeiling_;
    Operon::Zobrist zobrist_;
    std::vector<std::vector<std::vector<Operon::Tree>>> buckets_; // [GrammarSymbol index][budget][candidate]
    std::vector<std::vector<gtl::parallel_flat_hash_set_m<Operon::Hash>>> seen_; // [GrammarSymbol index][budget]
};

} // namespace Operon

#endif
