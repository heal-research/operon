// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/enumeration.hpp"

#include <algorithm>
#include <array>
#include <cmath>
#include <thread>
#include <unordered_map>

#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>

#include "operon/core/serialization.hpp" // for ToBeve (representative tie-break)
#include "operon/optimizer/optimizer.hpp" // for FitResult/FitFailure::FinalCost

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
    // has overshoot exactly 1 for a self-combine with at most one operand
    // already rooted in the combining Op at a time (Mul self-combine, the
    // Expression/SimpleExpr Add recursion) - and that overshoot doesn't
    // compound across recursive steps, since each step reads its operand from
    // the operand's own already-shrunk bucket, not a carried-over nominal
    // value. Verified empirically by the SimpleTerm closed-form completeness
    // test in test/source/implementation/enumeration.cpp.
    //
    // Margin 2 (not 1) is required once RecurringFactor's ESR Add(SimpleExpr,
    // SimpleExpr) production (see grammar.cpp's RecipeFor) is enabled:
    // SimpleExpr's own root is *unconditionally* Add (both of its
    // productions emit Op=Add - see Grammar::Rebuild), so Reduce()'s
    // same-HashValue child-merge fires on *both* operands simultaneously,
    // removing 2 Add header nodes (one per operand) for the cost of the 1
    // new outer Add node this production nominally budgets for - a realized
    // overshoot of 2, not 1. Mul(SimpleExpr, SimpleExpr) doesn't have this
    // problem (SimpleExpr is never Mul-rooted), and Sub/Div/Pow/Aq are never
    // flattened by Reduce() at all (IsCommutative() is false for all of
    // them). Raising this margin for every production (rather than only the
    // one that needs it) just searches one extra fringe budget level
    // everywhere else - correctness-preserving, only a constant amount of
    // extra unproductive work.
    constexpr std::size_t WorkingBudgetMargin = 2;
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
    bucketMutex_.resize(GrammarSymbols::Count);
    for (auto& row : buckets_) { row.resize(workingCeiling_ + 1); }
    for (auto& row : seen_) { row.resize(workingCeiling_ + 1); }
    // std::mutex isn't movable, so this row can't use resize() the way
    // buckets_/seen_ do (resize() on growth may need to move existing
    // elements) - each row is constructed once, directly at its final size,
    // instead.
    for (auto& row : bucketMutex_) { row = std::vector<std::mutex>(workingCeiling_ + 1); }
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
        std::scoped_lock lock(bucketMutex_[idx][complexity]);
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

void EnumerationEngine::ProcessNonterminal(tf::Executor& executor, GrammarSymbol nt, std::size_t budget)
{
    tf::Taskflow taskflow;

    // Appends the Op node (arity = number of direct children already pushed
    // onto `nodes`, i.e. p.Operands.size() plus one more if p.TrailingConstant
    // added a bias sibling) and, if p.ResultScale != 1, one more
    // non-optimizable Constant(ResultScale) + Mul(2) wrapping the whole
    // thing - see grammar.hpp's Production::ResultScale.
    auto appendOpAndScale = [](Operon::Vector<Node>& nodes, Production const& p, std::size_t childCount) {
        nodes.push_back(Node::Function(static_cast<Hash>(p.Op), static_cast<uint16_t>(childCount)));
        if (p.ResultScale != Operon::Scalar{1}) {
            auto scale = Node::Constant(static_cast<double>(p.ResultScale));
            scale.Optimize = false;
            nodes.push_back(std::move(scale));
            nodes.push_back(Node::Function(static_cast<Hash>(BuiltinOp::Mul), 2));
        }
    };

    for (auto const& p : grammar_.Productions(nt)) {
        if (p.IsCoercion()) {
            // Single nonterminal operand, no new node appended: same budget as the operand.
            auto operandIdx = GrammarSymbols::GetIndex(*p.Operands.front().Symbol);
            auto const& bucket = buckets_[operandIdx][budget];
            if (bucket.empty()) { continue; }
            taskflow.for_each_index(std::size_t{0}, bucket.size(), std::size_t{1}, [this, nt, &bucket](std::size_t i) {
                TryInsert(nt, bucket[i]);
            });
            continue;
        }

        std::size_t const nonterminalCount = static_cast<std::size_t>(
            std::ranges::count_if(p.Operands, [](auto const& o) { return !o.IsFixed(); }));

        // Op node + optional weight Mul node + optional ResultScale Mul node.
        std::size_t const fixedCost = 1UL
            + (p.WeightFirstOperand ? 1UL : 0UL)
            + (p.ResultScale != Operon::Scalar{1} ? 1UL : 0UL);
        if (budget <= fixedCost) { continue; }
        auto const remaining = budget - fixedCost;

        if (nonterminalCount == 1) {
            // Exactly one nonterminal operand, plus zero or one fixed
            // (non-optimizable Constant) operand elsewhere in Operands order
            // - covers every unary SimpleExpr wrap (Square/SqrtAbs/LogAbs/
            // Exp/Sin/Inv/Log/Sqrt/Cbrt/Log10Abs, Operands.size()==1) and
            // Cube/TenExp (Operands.size()==2, one fixed + one nonterminal -
            // see grammar.cpp's RecipeFor).
            std::size_t ntSlot = 0;
            for (; ntSlot < p.Operands.size(); ++ntSlot) { if (!p.Operands[ntSlot].IsFixed()) { break; } }
            auto const operand = *p.Operands[ntSlot].Symbol;
            if (remaining < grammar_.MinComplexity(operand)) { continue; }
            auto operandIdx = GrammarSymbols::GetIndex(operand);
            auto const& bucket = buckets_[operandIdx][remaining];
            if (bucket.empty()) { continue; }
            taskflow.for_each_index(std::size_t{0}, bucket.size(), std::size_t{1}, [this, nt, &p, &bucket, ntSlot, &appendOpAndScale](std::size_t i) {
                auto const& t = bucket[i];
                Operon::Vector<Node> nodes;
                if (p.WeightFirstOperand) { nodes.push_back(Node::Constant(WeightPlaceholder)); }
                // Push slots in REVERSE Operands order (last slot first, slot 0 last/rightmost) - the
                // interpreter's binary/n-ary ops read their FIRST semantic operand (minuend, numerator,
                // Pow base, ...) from the position immediately preceding the op node (rightmost child,
                // Tree::Indices' first-returned index - see functions.hpp's Sub/Div/Pow and
                // pappus_backend.cpp's "operon lays out binary-op children with the semantic LEFT
                // operand at the higher index" comment), not from whichever operand happens to be
                // pushed first. Pushing Operands[0] LAST here is what makes it land at that rightmost
                // position, so Operands[0] is always the true first operand (e.g. Cube's SimpleExpr
                // base ends up as Pow's base, not its exponent) - required for Cube/TenExp's fixed
                // operand to land in the correct role, since (unlike the two-nonterminal branch below)
                // there is no symmetric second call to compensate for a swap.
                for (std::size_t k = 0; k < p.Operands.size(); ++k) {
                    std::size_t const slot = p.Operands.size() - 1 - k;
                    if (slot == ntSlot) { AppendNodes(nodes, t.Nodes()); } else { nodes.push_back(p.Operands[slot].ToNode()); }
                    // WeightFirstOperand only ever wraps a single-operand production's sole (nonterminal)
                    // operand - Cube/TenExp never set WeightFirstOperand, so this only fires when
                    // p.Operands.size() == 1 and slot == ntSlot == 0 (the loop's only iteration either way).
                    if (slot == 0 && p.WeightFirstOperand) { nodes.push_back(Node::Function(static_cast<Hash>(BuiltinOp::Mul), 2)); }
                }
                if (p.TrailingConstant) { nodes.push_back(Node::Constant(BiasPlaceholder)); }
                appendOpAndScale(nodes, p, p.Operands.size() + (p.TrailingConstant ? 1UL : 0UL));
                TryInsert(nt, Tree(std::move(nodes)).UpdateNodes());
            });
        } else {
            // Exactly two nonterminal operands (Term*Term, Expression's
            // Add(Term,Expression) recursion, and RecurringFactor's ESR
            // binary Add/Sub/Mul/Div/Pow/Aq(SimpleExpr,SimpleExpr)) - never
            // combined with a fixed operand in this grammar.
            EXPECT(p.Operands.size() == 2);
            auto const op0 = *p.Operands[0].Symbol;
            auto const op1 = *p.Operands[1].Symbol;
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
            // op0==op1 alone isn't sufficient - see grammar.hpp's Production::Commutative doc comment: a
            // same-symbol production whose Op isn't actually commutative (e.g. Aq) must not take this skip, since
            // (b0,b1) and (b1,b0) build genuinely different trees for it.
            bool const selfCombineUnweighted = (op0 == op1) && !p.WeightFirstOperand && p.Commutative;

            for (std::size_t b0 = min0; b0 <= remaining; ++b0) {
                if (remaining - b0 < min1) { continue; }
                auto const b1 = remaining - b0;
                if (selfCombineUnweighted && b0 > b1) { continue; }

                auto const& bucket0 = buckets_[idx0][b0];
                auto const& bucket1 = buckets_[idx1][b1];
                if (bucket0.empty() || bucket1.empty()) { continue; }

                // Flattened over the full bucket0 x bucket1 cross product, via k = i0*n1 + i1, rather than
                // parallelizing bucket0's index alone with bucket1 iterated sequentially inside each task - which
                // operand's bucket is the larger one is production-dependent (e.g. Expression's Add(Term,
                // Expression) recursion: Term/bucket0 stays small while Expression/bucket1 grows combinatorially
                // with recursion depth), so a single fixed parallelization axis would starve the executor
                // whenever that axis is the small one.
                auto const n0 = bucket0.size();
                auto const n1 = bucket1.size();
                taskflow.for_each_index(std::size_t{0}, n0 * n1, std::size_t{1}, [this, nt, &p, &bucket0, &bucket1, n1, &appendOpAndScale](std::size_t k) {
                    auto const& t0 = bucket0[k / n1];
                    auto const& t1 = bucket1[k % n1];
                    Operon::Vector<Node> nodes;
                    // Push Operands[1] (t1) first, then Operands[0] (t0) last/rightmost - see the
                    // single-nonterminal-operand branch's comment above for why rightmost is the
                    // interpreter's true first operand. For the pure two-nonterminal ops this branch
                    // handles (Sub/Div/Pow/Aq/Add/Mul(SimpleExpr,SimpleExpr), Add(Term,Expression)),
                    // the b0/b1 loop above already builds every (bucket0,bucket1) assignment - including
                    // both (t0,t1) and (t1,t0) - so which physical tree ends up representing which
                    // labeled operand order doesn't change the overall enumerated set for those; it's
                    // fixed here anyway to keep this branch's node layout consistent with the
                    // single-operand branch's (and CanonicalizeEnumerationTree's) operand-role
                    // convention rather than relying on that symmetry.
                    AppendNodes(nodes, t1.Nodes());
                    if (p.WeightFirstOperand) { nodes.push_back(Node::Constant(WeightPlaceholder)); }
                    AppendNodes(nodes, t0.Nodes());
                    if (p.WeightFirstOperand) { nodes.push_back(Node::Function(static_cast<Hash>(BuiltinOp::Mul), 2)); }
                    // Op's direct children: the second operand, then the (possibly Mul-wrapped) first
                    // operand - this branch always handles exactly two nonterminal operands.
                    appendOpAndScale(nodes, p, 2);
                    TryInsert(nt, Tree(std::move(nodes)).UpdateNodes());
                });
            }
        }
    }

    executor.run(taskflow).wait();
}

void EnumerationEngine::Build(tf::Executor& executor, Operon::ReportCallback shouldStop)
{
    SeedTerminals();
    // Searches up to workingCeiling_ (> maxComplexity_) so combinations whose
    // *nominal* budget overshoots maxComplexity_ but whose realized (post-
    // Reduce()) complexity doesn't still get tried - see WorkingBudgetMargin.
    // TryInsert's complexity check guarantees nothing is actually stored past
    // maxComplexity_, so the caller-visible ceiling is unaffected.
    for (std::size_t budget = 1; budget <= workingCeiling_; ++budget) {
        for (auto nt : ProcessingOrder) {
            ProcessNonterminal(executor, nt, budget);
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

void EnumerationEngine::Build(Operon::ReportCallback shouldStop, std::size_t threads)
{
    if (threads == 0) { threads = std::thread::hardware_concurrency(); }
    tf::Executor executor(threads);
    Build(executor, std::move(shouldStop));
}

auto MakeObjectiveScorer(gsl::not_null<Operon::EvaluatorBase const*> evaluator) -> EnumerationScorer
{
    // ConsiderBest/BestTrees rank candidates by Score alone - a multi-objective evaluator (e.g.
    // MultiEvaluator, ObjectiveCount() > 1) would silently have every objective past the first
    // ignored, giving plausible-looking but wrong model selection with no diagnostic. Reject it here
    // rather than let it compile and misbehave.
    EXPECT(evaluator->ObjectiveCount() == 1);
    return [evaluator](Operon::RandomGenerator& rng, Operon::Tree const& tree, double /*structureBits*/,
               Operon::Span<Operon::Scalar> buf) -> EnumerationScore {
        Operon::Individual ind{1};
        ind.Genotype = tree; // Score() (and hence the evaluator) never observes coefficient identity, a copy is fine
        auto fitness = (*evaluator)(rng, ind, buf);
        return EnumerationScore{ .Score = fitness.front() };
    };
}

GrammarEnumerationAlgorithm::GrammarEnumerationAlgorithm(EnumerationConfig config, Operon::Grammar grammar,
    gsl::not_null<Operon::OptimizerBase const*> optimizer, EnumerationScorer scorer, Operon::RandomGenerator& rng)
    : config_(config)
    , engine_(std::move(grammar), config.MaxComplexity, rng)
    , optimizer_(optimizer)
    , scorer_(std::move(scorer))
{
}

void GrammarEnumerationAlgorithm::ConsiderBest(EnumerationResult result)
{
    // TopK == 0 means "keep nothing" - handle it explicitly before the
    // capacity check below, which would otherwise call best_.back() on an
    // empty vector (0 >= 0 is true) and crash.
    if (config_.TopK == 0) { return; }

    // best_ is kept sorted ascending at all times (see the member comment) by (Score, CanonicalKey) -
    // CanonicalKey is the strict tie-break so two equal-Score results never compare equal here (a
    // stable, deterministic order regardless of insertion order).
    auto lessThan = [](EnumerationResult const& a, EnumerationResult const& b) {
        if (a.Score != b.Score) { return a.Score < b.Score; }
        return a.CanonicalKey < b.CanonicalKey;
    };

    // A novel candidate already worse than the current worst kept entry (once at capacity) can be
    // rejected in O(1) instead of paying for an insertion + resort that would just be undone by the
    // trailing pop_back() anyway. Otherwise, insert in sorted position directly rather than push_back
    // + full re-sort - this only costs a linear shift, not an O(n log n) sort, per candidate.
    if (best_.size() >= config_.TopK && !lessThan(result, best_.back())) { return; }

    auto pos = std::ranges::upper_bound(best_, result, lessThan);
    best_.insert(pos, std::move(result));
    if (best_.size() > config_.TopK) { best_.pop_back(); }
}

void GrammarEnumerationAlgorithm::Run(tf::Executor& executor, Operon::RandomGenerator& rng, Operon::ReportCallback report)
{
    // Iterations() == 0 makes CoefficientOptimizer return a default-
    // constructed FitFailure (FinalCost == 0.0) without ever calling
    // optimizer_->Optimize() (see local_search.cpp) - every candidate
    // would then tie at cost 0.0 and the top-K ranking below becomes
    // meaningless. optimizer_ is caller-supplied and mutable
    // (OptimizerBase::SetIterations()), so this is a real precondition, not
    // just a theoretical one.
    EXPECT(optimizer_->Iterations() > 0);

    Operon::ReportCallback shouldStop = [&]() -> bool {
        if (StopRequested()) { return true; }
        if (report && report()) { RequestStop(); return true; }
        return false;
    };

    // Phase 1: build - see EnumerationEngine::Build's doc comment. TryInsert no longer has a fitting
    // hook; this only discovers/dedups distinct trees per (nonterminal, budget). A stop mid-Build
    // still leaves whatever was built so far available to the group/fit/rank pass below.
    engine_.Build(executor, std::move(shouldStop));

    // Phase 2: canonical grouping + deterministic representative selection (see the class's Run doc
    // comment for the exact 3-level tie-break). One pass over every stored Expression-bucket tree,
    // across every complexity budget.
    //
    // Hazard (sound, not a false merge, but a coverage narrowing): a canonical class can contain
    // members whose *parametric* families genuinely differ in expressiveness even though the
    // canonicalizer's free-parameter abstraction gives them the same Key - e.g. Square(K*x+K)
    // (a perfect-square quadratic, 2 free parameters after folding) shares a class with the
    // directly-built 3-free-parameter quadratic K1*x^2+K2*x+K3, because Square's distribution
    // folds every Optimize==true Constant into the same "free" marker regardless of how many
    // independent parameters produced it structurally. This tie-break has no way to see that
    // distinction (BucketSize/Complexity/serialization all favor the smaller Square form), so the
    // richer family is never fit - only the class's cheapest member is. This is intentional
    // (matches the canonicalizer's documented "never a false positive, may lose recall" contract),
    // not a bug, but worth knowing if enumeration output seems to skip an expected coefficient shape.
    struct ClassMember {
        Operon::Tree Candidate;
        std::size_t Complexity{};
        std::size_t BucketSize{};
    };
    auto better = [](ClassMember const& a, ClassMember const& b) {
        if (a.BucketSize != b.BucketSize) { return a.BucketSize < b.BucketSize; }
        if (a.Complexity != b.Complexity) { return a.Complexity < b.Complexity; }
        return Operon::Serialization::ToBeve(a.Candidate) < Operon::Serialization::ToBeve(b.Candidate);
    };

    std::unordered_map<std::string, ClassMember> representatives; // canonical Key -> current best representative
    for (std::size_t budget = 1; budget <= engine_.MaxComplexity(); ++budget) {
        auto bucket = engine_.Bucket(GrammarSymbol::Expression, budget);
        if (bucket.empty()) { continue; }
        for (auto const& tree : bucket) {
            auto canon = Operon::CanonicalizeEnumerationTree(tree);
            ClassMember candidate{ .Candidate = tree, .Complexity = budget, .BucketSize = bucket.size() };
            if (auto [it, inserted] = representatives.try_emplace(canon.Key, std::move(candidate)); !inserted) {
                if (better(candidate, it->second)) { it->second = std::move(candidate); }
            }
        }
    }
    if (representatives.empty()) { return; }

    // Deterministic iteration order (by canonical Key) for the fit/rank pass below - not required for
    // correctness (each representative is fit/scored independently; ConsiderBest's tie-break makes
    // BestTrees() itself order-independent), but keeps which worker's RNG stream ends up fitting a
    // given representative reproducible for a fixed (seed, executor worker count) pair, same
    // guarantee the pre-restructuring TryInsert-hook design offered.
    std::vector<std::pair<std::string, ClassMember>> repList(
        std::make_move_iterator(representatives.begin()), std::make_move_iterator(representatives.end()));
    std::ranges::sort(repList, {}, [](auto const& p) { return p.first; });

    // Phase 3+4: fit + rank. Fans out across `executor` the same way EnumerationEngine::ProcessNonterminal
    // does - each task needs its own per-worker RandomGenerator/scratch-buffer slot (see the class's Run doc
    // comment), seeded once, up front, from `rng` (itself single-threaded at this point) for reproducibility.
    EXPECT(config_.EvaluationBufferSize > 0);
    Operon::CoefficientOptimizer coeffOptimizer{optimizer_};
    auto const numWorkers = executor.num_workers();
    std::vector<Operon::RandomGenerator> workerRngs;
    workerRngs.reserve(numWorkers);
    for (std::size_t i = 0; i < numWorkers; ++i) { workerRngs.emplace_back(rng()); }
    std::vector<std::vector<Operon::Scalar>> workerBufs(
        numWorkers, std::vector<Operon::Scalar>(config_.EvaluationBufferSize));

    // Guards ConsiderBest's best_ mutation - the only other piece of shared mutable state this task touches
    // besides the per-worker slots above. Held only around ConsiderBest itself, not the coefficient fit/score
    // work above it, so concurrent task completions only ever serialize on the O(TopK) insertion, not on
    // LM/scorer cost.
    std::mutex bestMutex;

    tf::Taskflow taskflow;
    taskflow.for_each_index(std::size_t{0}, repList.size(), std::size_t{1}, [&](std::size_t i) {
        auto const worker = executor.this_worker_id();
        // taskflow tasks only ever execute on a worker thread (this_worker_id() returns -1 only for
        // the calling/master thread, which never runs a task body here since taskflow.run(...).wait()
        // blocks the master on std::future::wait rather than pulling work) - EXPECT rather than
        // silently aliasing slot 0, which would otherwise race real worker 0's RNG/buffer.
        EXPECT(worker >= 0);
        auto const slot = static_cast<std::size_t>(worker);
        auto& localRng = workerRngs[slot];
        auto& evalBuf = workerBufs[slot];

        auto const& [key, member] = repList[i];
        // Always take the optimizer's resulting tree, regardless of whether the fit outcome was
        // FitResult or FitFailure - mirrors BasicOffspringGenerator's own evaluate step
        // (operators/generator.hpp), which never branches on the outcome either. Scoring is then
        // computed fresh via `scorer_` rather than trusting the outcome's own cost fields, which
        // reflect whatever internal loss optimizer_ happens to minimize (e.g. LM's sum-of-squares),
        // not the user-selected ranking.
        auto tree = std::get<0>(coeffOptimizer(localRng, member.Candidate));
        // Nothing stops the fitted coefficients from turning a placeholder weight/bias into an
        // identity or annihilator element that Reduce()/Simplify() would have folded away had it
        // been there from the start (e.g. a WeightFirstOperand Constant fitted to exactly 1.0, or a
        // TrailingConstant bias fitted to exactly 0.0) - re-simplify the fitted copy before scoring.
        tree.Reduce();
        tree.Simplify();

        auto const structureBits = std::log2(static_cast<double>(member.BucketSize));
        auto score = scorer_(localRng, tree, structureBits, evalBuf);

        std::scoped_lock lock(bestMutex);
        ConsiderBest(EnumerationResult{
            .Score = score.Score,
            .NegativeLogLikelihood = score.NegativeLogLikelihood,
            .ParameterCodeBits = score.ParameterCodeBits,
            .StructureCodeBits = score.StructureCodeBits,
            .CanonicalKey = key,
            .Tree = std::move(tree),
        });
    });
    executor.run(taskflow).wait();
}

void GrammarEnumerationAlgorithm::Run(Operon::RandomGenerator& rng, Operon::ReportCallback report, std::size_t threads)
{
    if (threads == 0) { threads = std::thread::hardware_concurrency(); }
    tf::Executor executor(threads);
    Run(executor, rng, std::move(report));
}

} // namespace Operon
