// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_ENUMERATION_HPP
#define OPERON_ALGORITHMS_ENUMERATION_HPP

#include <atomic>
#include <functional>
#include <span>
#include <utility>
#include <vector>

#include <gsl/pointers>
#include <gtl/phmap.hpp>

#include "operon/algorithms/ga_base.hpp" // for Operon::ReportCallback
#include "operon/core/grammar.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/content_hash.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/evaluator.hpp" // for EvaluatorBase, Individual
#include "operon/operators/local_search.hpp" // for CoefficientOptimizer, OptimizerBase/OptimizerSummary fwd decls
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
[[nodiscard]] OPERON_EXPORT auto SymbolicComplexity(Operon::Tree const& tree) noexcept -> std::size_t;

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
// transposition cache, see hash/zobrist.hpp), so the seen_ check-and-insert
// itself is safe under concurrent access. This does NOT make TryInsert as a
// whole thread-safe yet: buckets_'s push_back (see TryInsert) is a plain,
// unguarded std::vector append - parallelizing the candidate-generation loop
// is later work, and will need to guard that append too (e.g. per-bucket
// mutex, or a lock-free append structure), not just reuse seen_'s primitive.
// Build() only ever runs single-threaded today, so this is moot for now.
class OPERON_EXPORT EnumerationEngine {
public:
    EnumerationEngine(Operon::Grammar grammar, std::size_t maxComplexity, Operon::RandomGenerator& rng);

    // Runs the bottom-up construction for budgets 1..maxComplexity in order.
    // `shouldStop`, if set, is checked once after each budget level finishes
    // (not per candidate, and not before the level starts - a level's own
    // fan-out always runs to completion first, so a caller's progress report
    // reflects that level's results rather than the previous one) and stops
    // the construction early if it returns true.
    void Build(Operon::ReportCallback shouldStop = {});

    // Invoked, if set, whenever a genuinely novel Expression-category tree is
    // inserted (i.e. not a duplicate already reached via another derivation) -
    // Expression is the only nonterminal meant to represent a complete
    // candidate model; Term/RecurringFactor/SimpleExpr/SimpleTerm are purely
    // compositional intermediates. The hook receives a mutable reference so a
    // caller (GrammarEnumerationAlgorithm) can fit coefficients in place
    // before the tree is stored.
    void SetOnNovelExpression(std::function<void(Operon::Tree&)> hook) { onNovelExpression_ = std::move(hook); }

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

    // Reduce()+Simplify()s `tree`, computes its realized SymbolicComplexity
    // (which can only be <= the budget it was built for - simplification
    // never adds nodes) and content hash, and inserts it into nt's bucket at
    // that realized complexity if not already present there. Returns whether
    // it was novel (i.e. actually inserted).
    //
    // This can insert into a bucket at a smaller budget than the one
    // currently being processed by Build() (a "shrink") - safe because
    // Build()'s budget loop only ever moves forward: once budget B has been
    // fully processed, nothing reads bucket[B] again until some later,
    // larger budget's ProcessNonterminal call does, and any shrink-driven
    // insertion into bucket[B] happens strictly before that (it's itself
    // triggered by processing some budget > B). So a late arrival in an
    // already-"finished" bucket is still visible to every future reader.
    //
    // Dedup relies on hash equality alone (seen_ stores only the 64-bit
    // content hash, not the tree) - a collision would silently drop a
    // distinct tree. Negligible at 64 bits, and the completeness tests'
    // exact closed-form bucket counts are evidence none has occurred in
    // practice, but this isn't a structural guarantee.
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
    std::function<void(Operon::Tree&)> onNovelExpression_;
};

struct EnumerationConfig {
    std::size_t MaxComplexity{20};
    std::size_t TopK{10}; // how many best-fitness models to retain (see GrammarEnumerationAlgorithm::BestTrees)
};

// Top-level driver: wraps EnumerationEngine with coefficient fitting (via the
// existing CoefficientOptimizer - this fully replaces symreg-cpp's Ceres
// dependency with operon's own optimizer stack, no new fitting code needed)
// and a stop-condition/reporting surface mirroring GeneticAlgorithmBase's
// ReportCallback/StopRequested()/RequestStop() idiom (ga_base.hpp), even
// though this doesn't inherit from GeneticAlgorithmBase - there's no
// population/generation model here, just a level-by-level DP construction.
//
// Coefficient fitting and fitness scoring are deliberately separate
// concerns, mirroring BasicOffspringGenerator's evaluate step
// (operators/generator.hpp): `optimizer` only drives CoefficientOptimizer's
// internal loss (used to fit parameters, e.g. LM's sum-of-squares) and is
// never itself surfaced as a score; `evaluator` is the user-selectable
// ErrorMetric (R2/NMSE/MSE/MAE/...) that actually ranks candidates in
// BestTrees(), same as GP/NSGA2. Always taking the optimizer's resulting
// tree (regardless of OptimizerSummary::Success) and re-scoring it via
// evaluator - rather than trusting OptimizerSummary's own cost fields -
// keeps this consistent with the rest of the codebase and avoids coupling
// ranking to whichever internal loss a given OptimizerBase happens to use.
class OPERON_EXPORT GrammarEnumerationAlgorithm {
public:
    // `rng` is used once here to build the engine's Zobrist salt table (see
    // EnumerationEngine); Run()'s own `rng` argument is independent and used
    // for coefficient fitting - callers may pass the same generator to both
    // or different ones.
    GrammarEnumerationAlgorithm(EnumerationConfig config, Operon::Grammar grammar, gsl::not_null<Operon::OptimizerBase const*> optimizer, gsl::not_null<Operon::EvaluatorBase const*> evaluator, Operon::RandomGenerator& rng);

    // Fits coefficients (via CoefficientOptimizer) for every novel Expression
    // discovered during construction, scores the result via `evaluator`, and
    // tracks the config.TopK best (lower = better, matching every Operon
    // ErrorMetric's minimization convention) in BestTrees(). Stops early if
    // `report` returns true, or if RequestStop() was called.
    void Run(Operon::RandomGenerator& rng, Operon::ReportCallback report = {});

    [[nodiscard]] auto StopRequested() const -> bool { return stopRequested_.load(std::memory_order_acquire); }
    void RequestStop() { stopRequested_.store(true, std::memory_order_release); }

    // Best-fitness Expression trees found so far, sorted ascending by
    // evaluator score (lower = better), capped at EnumerationConfig::TopK.
    [[nodiscard]] auto BestTrees() const -> std::span<std::pair<Operon::Scalar, Operon::Tree> const> { return best_; }

    [[nodiscard]] auto GetEngine() const -> EnumerationEngine const& { return engine_; }

private:
    void ConsiderBest(Operon::Scalar fitness, Operon::Tree tree);

    EnumerationConfig config_;
    EnumerationEngine engine_;
    gsl::not_null<Operon::OptimizerBase const*> optimizer_;
    gsl::not_null<Operon::EvaluatorBase const*> evaluator_;
    std::atomic<bool> stopRequested_{false};
    std::vector<std::pair<Operon::Scalar, Operon::Tree>> best_; // sorted ascending by .first, size() <= config_.TopK
};

} // namespace Operon

#endif
