// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/enumeration.hpp"

#include <algorithm>
#include <array>

#include "operon/optimizer/optimizer.hpp" // for OptimizerSummary::FinalCost

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
    // by WeightFirstOperand/TrailingConstant. Each deliberately avoids the
    // identity element of the Op it sits next to: WeightPlaceholder != 1.0
    // (Mul's identity) and BiasPlaceholder != 0.0 (Add's identity), because
    // Tree::Simplify() removes x*1 and x+0 nodes - using either identity
    // value here would let Simplify() strip away the very Mul/Add "glue" a
    // not-yet-fitted weight/bias needs to survive on, before coefficient
    // fitting ever gets a chance to explore a non-trivial value. Content-hash
    // ignores Constant leaf values entirely (see hash/content_hash.hpp), so
    // this choice has no effect on dedup.
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
    // maxLength=1 is correct only because zobrist_ is used exclusively via
    // ComputeContentHash (see TryInsert), which always calls
    // Zobrist::ComputeHash with pos=0 - it has no array-position dimension.
    // This table is NOT sized to double as a general-purpose Zobrist::ComputeHash(tree)
    // (whole-tree transposition hash) for any tree longer than 1 node; don't
    // repurpose zobrist_ for that without resizing it first.
    , zobrist_(rng, /*maxLength=*/1, grammar_.VariableHashes())
{
    buckets_.resize(GrammarSymbols::Count);
    seen_.resize(GrammarSymbols::Count);
    for (auto& row : buckets_) { row.resize(workingCeiling_ + 1); }
    for (auto& row : seen_) { row.resize(workingCeiling_ + 1); }
}

auto EnumerationEngine::Bucket(GrammarSymbol nt, std::size_t budget) const -> std::span<Operon::Tree const>
{
    EXPECT(budget <= maxComplexity_); // contract is [0, MaxComplexity()], not [0, workingCeiling_]
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
        if (nt == GrammarSymbol::Expression && onNovelExpression_) {
            onNovelExpression_(tree); // may fit coefficients in place
        }
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
            //
            // WorkingBudgetMargin's value (1) was hand-derived specifically
            // for the current production table (see Grammar::Rebuild in
            // grammar.cpp): every flattening self-combine/recursion here has
            // at most one operand that's already rooted in the combining Op
            // at a time (see enumeration.hpp's "Budget accounting note").
            // A future production violating that (e.g. a ternary self-combine,
            // or one where both operands can simultaneously already be
            // Op-rooted) could need a larger margin - the completeness tests
            // in test/source/implementation/enumeration.cpp are the guard;
            // if a new production is added there, extend those tests first.
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

void EnumerationEngine::Build(Operon::ReportCallback shouldStop)
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
        // Checked after this level's own processing (not before), so a
        // caller's progress report reflects this level's results rather than
        // lagging one level behind. Note this runs once more than a caller
        // might expect - budget == workingCeiling_ (the WorkingBudgetMargin
        // level) still processes candidates and still triggers this check,
        // even though every candidate at that level has realized complexity
        // > maxComplexity_ and TryInsert rejects all of them. Harmless (the
        // callback just sees "nothing new" one extra time) so not worth
        // special-casing out.
        if (shouldStop && shouldStop()) { return; }
    }
}

GrammarEnumerationAlgorithm::GrammarEnumerationAlgorithm(EnumerationConfig config, Operon::Grammar grammar, gsl::not_null<Operon::OptimizerBase const*> optimizer, gsl::not_null<Operon::EvaluatorBase const*> evaluator, Operon::RandomGenerator& rng)
    : config_(config)
    , engine_(std::move(grammar), config.MaxComplexity, rng)
    , optimizer_(optimizer)
    , evaluator_(evaluator)
{
    // ConsiderBest/BestTrees rank candidates by fitness.front() alone - a
    // multi-objective evaluator (e.g. MultiEvaluator, ObjectiveCount() > 1)
    // would silently have every objective past the first ignored, giving
    // plausible-looking but wrong model selection with no diagnostic. Reject
    // it here rather than let it compile and misbehave.
    EXPECT(evaluator_->ObjectiveCount() == 1);
}

void GrammarEnumerationAlgorithm::ConsiderBest(Operon::Scalar fitness, Operon::Tree tree)
{
    // TopK == 0 means "keep nothing" - handle it explicitly before the
    // capacity check below, which would otherwise call best_.back() on an
    // empty vector (0 >= 0 is true) and crash.
    if (config_.TopK == 0) { return; }

    // best_ is kept sorted ascending at all times (see the member comment),
    // so a novel candidate that's already worse than the current worst kept
    // entry (once at capacity) can be rejected in O(1) instead of paying for
    // an insertion + resort that would just be undone by the trailing
    // resize() anyway. Otherwise, insert in sorted position directly rather
    // than push_back + full re-sort on every call - this only costs a linear
    // shift, not an O(n log n) sort, per novel Expression.
    if (best_.size() >= config_.TopK && fitness >= best_.back().first) { return; }

    auto pos = std::ranges::upper_bound(best_, fitness, {}, [](auto const& p) { return p.first; });
    best_.emplace(pos, fitness, std::move(tree));
    if (best_.size() > config_.TopK) { best_.pop_back(); }
}

void GrammarEnumerationAlgorithm::Run(Operon::RandomGenerator& rng, Operon::ReportCallback report)
{
    // Iterations() == 0 makes CoefficientOptimizer return a default-
    // constructed OptimizerSummary (FinalCost == 0.0, Success == false)
    // without ever calling optimizer_->Optimize() (see local_search.cpp) -
    // every novel Expression would then tie at cost 0.0 and the top-K
    // ranking below becomes meaningless. optimizer_ is caller-supplied and
    // mutable (OptimizerBase::SetIterations()), so this is a real
    // precondition, not just a theoretical one.
    EXPECT(optimizer_->Iterations() > 0);

    Operon::CoefficientOptimizer coeffOptimizer{optimizer_};
    // Reused across every novel Expression rather than letting the 2-arg
    // evaluator_ overload allocate its own scratch buffer per call (see
    // EvaluatorBase::Evaluate) - enumeration can produce thousands of
    // candidates per run, so a per-candidate heap allocation here adds up.
    std::vector<Operon::Scalar> evalBuf(evaluator_->GetProblem()->TrainingRange().Size());
    engine_.SetOnNovelExpression([&](Operon::Tree& tree) {
        // Always take the optimizer's resulting tree, regardless of
        // OptimizerSummary::Success - mirrors BasicOffspringGenerator's own
        // evaluate step (operators/generator.hpp), which never branches on
        // Success either. Fitness is then computed fresh via evaluator_
        // rather than trusting OptimizerSummary's own cost fields, which
        // reflect whatever internal loss optimizer_ happens to minimize
        // (e.g. LM's sum-of-squares), not the user-selected ErrorMetric.
        tree = std::get<0>(coeffOptimizer(rng, tree));
        Operon::Individual ind{1};
        // Copy (not move) here: `tree` is a reference into the engine's own
        // novel-candidate slot (see TryInsert), which moves it into storage
        // right after this hook returns - moving from it here would leave
        // that slot empty. ind.Genotype is a distinct, already-owned copy
        // with nothing else pending on it, so it can be moved into
        // ConsiderBest below instead of copying `tree` a second time.
        ind.Genotype = tree;
        // Re-simplify the *copy* only - `tree` itself must stay exactly as
        // TryInsert computed its bucket/dedup complexity for (see above),
        // but nothing stops the fitted coefficients from turning a
        // placeholder weight/bias into an identity or annihilator element
        // that Reduce()/Simplify() would have folded away had it been there
        // from the start: a WeightFirstOperand Constant multiplying a
        // *compound* Term (e.g. Constant * sin(...)) fitted to exactly 1.0,
        // or a TrailingConstant bias fitted to exactly 0.0. (A bare Variable
        // operand's own weight - e.g. the "1.000000" in a printed
        // "1.000000 * X6" - is not an example of this: InfixFormatter always
        // prints a Variable's weight explicitly, and Simplify() correctly
        // never strips a Variable node just because its weight is 1, since
        // the variable's contribution itself would be lost.)
        ind.Genotype.Reduce();
        ind.Genotype.Simplify();
        auto fitness = (*evaluator_)(rng, ind, evalBuf);
        ConsiderBest(fitness.front(), std::move(ind.Genotype));
    });

    Operon::ReportCallback shouldStop = [&]() -> bool {
        if (StopRequested()) { return true; }
        if (report && report()) { RequestStop(); return true; }
        return false;
    };

    engine_.Build(std::move(shouldStop));

    // onNovelExpression_ captures coeffOptimizer (and rng) by reference, both
    // function-locals about to go out of scope - clear the hook so a stale
    // reference can't be invoked from any future entry point (defensive:
    // today nothing public can trigger that, since GetEngine() returns a
    // const& and Build() is non-const, but this shouldn't rely on that).
    engine_.SetOnNovelExpression(nullptr);
}

} // namespace Operon
