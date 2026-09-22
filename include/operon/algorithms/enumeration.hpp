// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_ENUMERATION_HPP
#define OPERON_ALGORITHMS_ENUMERATION_HPP

#include <array>
#include <cmath>
#include <mutex>
#include <span>
#include <string>
#include <utility>
#include <vector>

#include <gsl/pointers>
#include <gtl/phmap.hpp>

#include "operon/algorithms/enumeration_canonicalizer.hpp" // for CanonicalizeEnumerationTree
#include "operon/algorithms/stoppable.hpp" // for Operon::ReportCallback, StoppableAlgorithm
#include "operon/core/grammar.hpp"
#include "operon/core/tree.hpp"
#include "operon/hash/content_hash.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/information_criteria/minimum_description_length.hpp" // for ParameterDescriptionLength
#include "operon/operators/evaluator.hpp" // for EvaluatorBase, Individual, detail::ProfileSigma, FitLinearScaling
#include "operon/operators/local_search.hpp" // for CoefficientOptimizer, OptimizerBase/FitResult/FitFailure fwd decls
#include "operon/operon_export.hpp"
#include "operon/random/random.hpp"

// forward declaration - keeps taskflow.hpp (heavy) out of this public header; only enumeration.cpp/nsga2.cpp-style
// .cpp files that actually build a Taskflow need the full definition.
namespace tf { class Executor; }

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
// construction-time placeholder value, Optimize=true) and has no fitting
// hook of its own - TryInsert only reduces, simplifies, content-deduplicates,
// and stores trees. Coefficient fitting, algebraic canonical grouping, and
// scoring/ranking are GrammarEnumerationAlgorithm::Run's job, run as a
// separate post-Build phase (build, canonical-group, fit, rank - see Run's
// doc comment) - decoupling "what distinct trees exist" from "which of them
// get fit and how they're ranked" keeps this engine fully testable on its
// own: that productions combine into valid Trees, that Reduce/Simplify
// interact correctly with the budget accounting, and that content-hash dedup
// collapses duplicates reached via different derivation paths.
//
// Budget accounting note: combining two operands' node counts plus a fixed
// per-Op cost is only an accurate prediction of the *realized* (post-
// Reduce()) complexity when neither operand's own root is already the same
// Op being applied. Term/SimpleTerm's Mul self-combine, Expression/
// SimpleExpr's Add(Term, Expression) recursion, and RecurringFactor's ESR
// Add(SimpleExpr, SimpleExpr) production all violate this (Reduce() merges
// the new Op into an operand's pre-existing same-type root instead of adding
// a distinct node), so the naive per-combination budget overshoots the true
// complexity by a small constant - see WorkingBudgetMargin in enumeration.cpp
// for how the DP compensates.
// Thread-safety: the per-(nonterminal, budget) dedup sets are gtl::parallel_flat_hash_set_m (the same primitive
// ZobristCache uses for its transposition cache, see hash/zobrist.hpp), so the seen_ check-and-insert itself is
// safe under concurrent access. buckets_'s std::vector::push_back (see TryInsert) is additionally guarded by
// bucketMutex_ (one per (nonterminal, budget) cell, held only around the append itself - the Reduce()/
// Simplify()/hash-compute work that dominates TryInsert's cost runs outside the lock). Build(tf::Executor&, ...)
// fans ProcessNonterminal's candidate generation for one (nonterminal, budget) call out across that executor;
// nothing about that call's own reads can race its own writes - see ProcessNonterminal's doc comment for why -
// so the mutex only has to arbitrate concurrent *writes* into the same nonterminal's buckets.
class OPERON_EXPORT EnumerationEngine {
public:
    EnumerationEngine(Operon::Grammar grammar, std::size_t maxComplexity, Operon::RandomGenerator& rng);

    // Runs the bottom-up construction for budgets 1..maxComplexity in order.
    // `shouldStop`, if set, is checked once after each budget level finishes
    // (not per candidate, and not before the level starts - a level's own
    // fan-out always runs to completion first, so a caller's progress report
    // reflects that level's results rather than the previous one) and stops
    // the construction early if it returns true.
    //
    // `executor` drives each (nonterminal, budget) level's own candidate generation in parallel (see
    // ProcessNonterminal) - budget levels themselves, and nonterminals within a level, still run strictly in order
    // (required for correctness, see ProcessNonterminal's doc comment), so this only parallelizes work *within*
    // one ProcessNonterminal call, not across them.
    void Build(tf::Executor& executor, Operon::ReportCallback shouldStop = {});

    // Convenience overload: builds and owns a local tf::Executor with `threads` workers (0 =
    // std::thread::hardware_concurrency(), mirroring NSGA2::Run's analogous convenience overload) for the
    // duration of this call.
    void Build(Operon::ReportCallback shouldStop = {}, std::size_t threads = 0);

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
    //
    // Fans every production's candidate-building loop out across `executor` (one tf::Taskflow, run to completion
    // before returning) - safe because every read this call makes targets either a strictly lower budget (fully
    // built by an earlier, completed Build() level) or, for a same-budget coercion, an earlier-in-ProcessingOrder
    // nonterminal's bucket (also already fully built this level) - and because a self-combine production's own
    // writes can only "shrink" into complexity budget-1 (WorkingBudgetMargin, see enumeration.cpp), which is
    // always strictly above every b0/b1 this call itself reads (both bounded by budget - fixedCost - the other
    // operand's MinComplexity, and fixedCost + MinComplexity >= 2). So no task here ever reads a bucket another
    // task in the same call is concurrently writing - concurrent *writes* into the same nt's buckets (from
    // different productions/operand splits landing in the same complexity cell) are still possible and are what
    // TryInsert's bucketMutex_ guards.
    void ProcessNonterminal(tf::Executor& executor, GrammarSymbol nt, std::size_t budget);

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
    // One mutex per (GrammarSymbol, budget) cell, guarding just that cell's buckets_ push_backs - see TryInsert
    // and the class-level Thread-safety comment. Cell granularity (rather than one mutex per GrammarSymbol shared
    // across every budget) keeps concurrent inserts into different budget cells of the same nonterminal from
    // contending on a single lock. Sized/constructed like buckets_/seen_ (see the ctor) - std::mutex isn't
    // movable, so this can't use vector<vector<mutex>>'s resize() the way buckets_/seen_ do; the ctor
    // default-constructs each row's mutexes in place at their final size instead.
    std::vector<std::vector<std::mutex>> bucketMutex_;
};

// Which score ranks candidates in GrammarEnumerationAlgorithm::BestTrees() -
// see EnumerationConfig::Ranking and the two scorer factories below.
enum class EnumerationRanking : uint8_t {
    MinimumDescriptionLength, // ESR-parity default - see MakeMdlScorer
    Objective,                // the pre-existing evaluator-based (ErrorMetric) ranking - see MakeObjectiveScorer
};

struct EnumerationConfig {
    std::size_t MaxComplexity{20};
    std::size_t TopK{10}; // how many best-fitness models to retain (see GrammarEnumerationAlgorithm::BestTrees)
    EnumerationRanking Ranking{EnumerationRanking::MinimumDescriptionLength};
    // Size of the per-worker evaluation scratch buffer GrammarEnumerationAlgorithm::Run allocates and
    // passes to `scorer` (see EnumerationScorer) - must be >= the training range size the scorer's
    // closure was built against (MakeMdlScorer/MakeObjectiveScorer's problem->TrainingRange().Size()).
    // Required (EXPECT'd > 0 in Run()) whenever there is at least one Expression to score; the
    // algorithm itself has no way to query this from a fully type-erased EnumerationScorer.
    std::size_t EvaluationBufferSize{0};
};

// One candidate's ranking score, as produced by an EnumerationScorer.
// `Score` is always lower-is-better (both MakeMdlScorer and
// MakeObjectiveScorer honor this). For the MDL ranking, `Score` is the total
// description length in *bits* (NegativeLogLikelihood/ln(2) +
// ParameterCodeBits + StructureCodeBits); NegativeLogLikelihood is left in
// *nats* (the natural unit Gaussian/Poisson ComputeLikelihood produce),
// while ParameterCodeBits and StructureCodeBits are already bits - see
// MakeMdlScorer. For the Objective ranking, Score is the raw ErrorMetric
// value and the other three fields are NaN (not applicable).
struct EnumerationScore {
    Operon::Scalar Score{};
    double NegativeLogLikelihood{std::numeric_limits<double>::quiet_NaN()};
    double ParameterCodeBits{std::numeric_limits<double>::quiet_NaN()};
    double StructureCodeBits{std::numeric_limits<double>::quiet_NaN()};
};

// Scores one already-coefficient-fitted candidate tree. `structureBits` is
// supplied by the caller (GrammarEnumerationAlgorithm::Run computes it per
// canonical-class representative as log2 of that representative's own
// pre-canonical Expression-bucket size - see Run's doc comment); the MDL
// scorer folds it directly into Score, the Objective scorer ignores it.
// `buf` is caller-owned evaluation scratch space (>= the problem's training
// range size), reused across calls to avoid a per-candidate allocation.
using EnumerationScorer = Operon::MoveOnlyFunction<EnumerationScore(Operon::RandomGenerator&, Operon::Tree const&, double structureBits, Operon::Span<Operon::Scalar>)>;

// Builds the ESR-parity MDL EnumerationScorer (EnumerationRanking::MinimumDescriptionLength).
// Reuses the same Gaussian/Poisson prediction, linear-scaling, Jacobian, Fisher-diagonal, and
// profiled/fixed-sigma machinery as MinimumDescriptionLengthEvaluator (operators/evaluator.hpp) - the
// difference is this operates directly on a bare Tree (no Individual/ScoreContext; enumeration
// candidates aren't population individuals) and reports the negative log-likelihood, parameter, and
// structure codelength components separately instead of only their sum.
//
// Score = (negativeLogLikelihoodNats + parameterDescriptionLengthNats) / ln(2) + structureBits (all
// in bits); non-finite predictions, likelihoods, Fisher entries, or totals clamp Score to
// EvaluatorBase::ErrMax (see EnumerationScore's doc comment for the exact unit split).
//
// `sigma`: empty means profile sigma-hat per candidate (Gaussian) or use Poisson's unweighted
// likelihood; a 1-element span fixes a shared scalar sigma; an N-element span (N ==
// problem->TrainingRange().Size()) fixes a per-sample sigma. Captured by value into the returned
// closure, so the caller's own buffer can go out of scope after this call.
template <typename DTable, Concepts::Likelihood Lik>
    requires Concepts::HasFisherMatrix<Lik>
auto MakeMdlScorer(gsl::not_null<Operon::Problem const*> problem, gsl::not_null<DTable const*> dtable,
    std::vector<Operon::Scalar> sigma = {}) -> EnumerationScorer
{
    return [problem, dtable, sigma = std::move(sigma)](
               Operon::RandomGenerator& /*rng*/, Operon::Tree const& tree, double structureBits,
               Operon::Span<Operon::Scalar> buf) -> EnumerationScore {
        auto const trainingRange = problem->TrainingRange();
        auto const* dataset = problem->GetDataset();
        auto parameters = tree.GetCoefficients();

        EXPECT(buf.size() >= trainingRange.Size());
        auto yPred = buf.subspan(0, trainingRange.Size());
        Operon::Interpreter<Operon::Scalar, DTable> const interpreter{ dtable.get(), dataset, &tree };
        interpreter.Evaluate(parameters, trainingRange, yPred);

        auto yTrue = problem->TargetValues(trainingRange);
        auto const weights = problem->Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const>{});
        std::optional<Operon::LinearScaling> scaling{};
        if (problem->LinearScalingEnabled()) {
            scaling = Operon::FitLinearScaling(yPred, yTrue, weights, problem->LinearScalingOmitsNonFinite());
            scaling->ApplyInPlace(yPred);
        }

        Operon::Scalar profiledSigma{};
        if (sigma.empty() && Lik::UsesSigma) { profiledSigma = Operon::detail::ProfileSigma(yPred, yTrue); }
        auto const effectiveSigma = (sigma.empty() && Lik::UsesSigma)
            ? Operon::Span<Operon::Scalar const>{ &profiledSigma, 1 }
            : Operon::Span<Operon::Scalar const>{ sigma };

        Eigen::Matrix<Operon::Scalar, -1, -1> jac = interpreter.JacRev(parameters, trainingRange);
        if (scaling) { jac *= static_cast<Operon::Scalar>(scaling->Scale); }
        auto fisherMatrix = Lik::ComputeFisherMatrix(
            yPred, { jac.data(), static_cast<std::size_t>(jac.size()) }, effectiveSigma);
        auto fisherDiag = fisherMatrix.diagonal().array();

        auto const nllNats = static_cast<double>(Lik::ComputeLikelihood(yPred, yTrue, effectiveSigma));
        auto const paramNats = Operon::ParameterDescriptionLength(parameters, fisherDiag);

        constexpr double Ln2 = 0.6931471805599453094;
        auto const paramBits = paramNats / Ln2;
        auto score = (nllNats / Ln2) + paramBits + structureBits;
        if (!std::isfinite(score)) { score = static_cast<double>(EvaluatorBase::ErrMax); }

        return EnumerationScore{
            .Score = static_cast<Operon::Scalar>(score),
            .NegativeLogLikelihood = nllNats,
            .ParameterCodeBits = paramBits,
            .StructureCodeBits = structureBits,
        };
    };
}

// Builds an EnumerationScorer delegating to an existing single-objective EvaluatorBase
// (EnumerationRanking::Objective) - the pre-existing --objective-selectable ErrorMetric ranking,
// preserved as an opt-in alternative to MDL. `structureBits` is ignored; NegativeLogLikelihood/
// ParameterCodeBits/StructureCodeBits are NaN (not applicable to a raw error metric). Requires
// evaluator->ObjectiveCount() == 1 (enforced via an always-on EXPECT) - a multi-objective evaluator
// would silently have every objective past the first ignored by Score's scalar contract.
[[nodiscard]] OPERON_EXPORT auto MakeObjectiveScorer(gsl::not_null<Operon::EvaluatorBase const*> evaluator) -> EnumerationScorer;

// One ranked candidate returned by GrammarEnumerationAlgorithm::BestTrees() - see EnumerationScore
// for the component fields' semantics; CanonicalKey is this candidate's canonical-class identity
// (see CanonicalizeEnumerationTree), used as BestTrees()'s strict tie-break.
struct EnumerationResult {
    Operon::Scalar Score{};
    double NegativeLogLikelihood{std::numeric_limits<double>::quiet_NaN()};
    double ParameterCodeBits{std::numeric_limits<double>::quiet_NaN()};
    double StructureCodeBits{std::numeric_limits<double>::quiet_NaN()};
    std::string CanonicalKey;
    Operon::Tree Tree;
};

// Top-level driver: wraps EnumerationEngine with algebraic canonical grouping, coefficient fitting
// (via the existing CoefficientOptimizer - this fully replaces symreg-cpp's Ceres dependency with
// operon's own optimizer stack, no new fitting code needed), and scoring/ranking. Shares its
// stop-condition/reporting surface (ReportCallback/StopRequested()/RequestStop()) with
// GeneticAlgorithmBase-derived algorithms via StoppableAlgorithm, even though this doesn't inherit
// GeneticAlgorithmBase itself - there's no population/generation model here, just a level-by-level DP
// construction followed by a group/fit/rank pass.
//
// Coefficient fitting and scoring are deliberately separate concerns, mirroring
// BasicOffspringGenerator's evaluate step (operators/generator.hpp): `optimizer` only drives
// CoefficientOptimizer's internal loss (used to fit parameters, e.g. LM's sum-of-squares) and is
// never itself surfaced as a score; `scorer` (see EnumerationScorer, MakeMdlScorer/
// MakeObjectiveScorer) is what actually ranks candidates in BestTrees(). Always taking the
// optimizer's resulting tree (regardless of whether the fit outcome was FitResult or FitFailure) and
// re-scoring it via `scorer` - rather than trusting the outcome's own cost fields - keeps this
// consistent with the rest of the codebase and avoids coupling ranking to whichever internal loss a
// given OptimizerBase happens to use.
class OPERON_EXPORT GrammarEnumerationAlgorithm : public StoppableAlgorithm {
public:
    // `rng` is used once here to build the engine's Zobrist salt table (see
    // EnumerationEngine); Run()'s own `rng` argument is independent and used
    // for coefficient fitting - callers may pass the same generator to both
    // or different ones.
    GrammarEnumerationAlgorithm(EnumerationConfig config, Operon::Grammar grammar,
        gsl::not_null<Operon::OptimizerBase const*> optimizer, EnumerationScorer scorer, Operon::RandomGenerator& rng);

    // Runs EnumerationEngine::Build(), then a group/fit/rank pass:
    //   1. Canonical grouping: every stored Expression-bucket tree (across every complexity budget
    //      1..MaxComplexity()) is canonicalized (see CanonicalizeEnumerationTree) and grouped by its
    //      CanonicalExpression::Key into a canonical class.
    //   2. Representative selection: each class picks exactly one member to fit, deterministically -
    //      smallest pre-canonical Expression-bucket size at that member's own complexity (the
    //      log2 of which becomes its `structureBits`), then lower SymbolicComplexity, then
    //      lexicographically smaller strict (BEVE) tree serialization as the final tie-break.
    //   3. Fit: the representative's coefficients are fit via CoefficientOptimizer, exactly as
    //      before.
    //   4. Rank: `scorer` scores the fitted representative (passing its class's log2(bucket size) as
    //      structureBits); the config.TopK best (lower Score = better) are tracked in BestTrees(),
    //      with CanonicalKey as the strict tie-break on equal Score.
    // Stops early if `report` returns true, or if RequestStop() was called - a stop mid-Build still
    // runs the group/fit/rank pass over whatever was built so far.
    //
    // Precondition: optimizer->Iterations() > 0 (enforced via an always-on
    // EXPECT, not stripped under NDEBUG). At Iterations() == 0,
    // CoefficientOptimizer never calls Optimize() and every candidate ties
    // at cost 0.0, making the top-K ranking meaningless - callers that
    // expose `optimizer` configuration to end users (CLIs, language
    // bindings) must reject or default away iterations == 0 themselves, the
    // way operon_enum does, rather than let it reach here.
    //
    // Single-shot: Run() is not meant to be called more than once on the
    // same instance - unlike GeneticAlgorithmBase, this class has no
    // Reset() to clear StopRequested() between runs, and the underlying
    // EnumerationEngine::Build() is not re-runnable (its buckets/dedup sets
    // are already populated after the first call). Construct a new
    // GrammarEnumerationAlgorithm for another run, mirroring the one-shot
    // (not warm-restartable) contract already implied by EnumerationEngine.
    //
    // `executor` drives both EnumerationEngine::Build's parallel candidate generation and the
    // fit/rank pass's per-representative work, fanned out the same way - each task keyed off
    // tf::Executor::this_worker_id() for its own per-worker RandomGenerator/scratch-buffer slot,
    // mirroring the per-worker `slots` / per-individual `rngs` pattern NSGA2::Run uses for its own
    // threaded local search and evaluation. Fixed (seed, executor worker count) stays reproducible
    // for both phases: Build's dedup race no longer determines which worker fits a given tree
    // (fitting now happens in the deterministic canonical-representative list, not from inside
    // TryInsert), but the fit/rank pass's own worker-RNG-stream assignment is still interleaving-
    // dependent, so results are not reproducible *across* different worker counts.
    void Run(tf::Executor& executor, Operon::RandomGenerator& rng, Operon::ReportCallback report = {});

    // Convenience overload: builds and owns a local tf::Executor with `threads` workers (0 =
    // std::thread::hardware_concurrency(), mirroring NSGA2::Run's analogous convenience overload) for the
    // duration of this call.
    void Run(Operon::RandomGenerator& rng, Operon::ReportCallback report = {}, std::size_t threads = 0);

    // StopRequested()/RequestStop() are inherited from StoppableAlgorithm.

    // Best-fitness canonical-class representatives found, sorted ascending by Score (lower = better),
    // ties broken by CanonicalKey, capped at EnumerationConfig::TopK.
    [[nodiscard]] auto BestTrees() const -> std::span<EnumerationResult const> { return best_; }

    [[nodiscard]] auto GetEngine() const -> EnumerationEngine const& { return engine_; }

private:
    void ConsiderBest(EnumerationResult result);

    EnumerationConfig config_;
    EnumerationEngine engine_;
    gsl::not_null<Operon::OptimizerBase const*> optimizer_;
    EnumerationScorer scorer_;
    std::vector<EnumerationResult> best_; // sorted ascending by (.Score, .CanonicalKey), size() <= config_.TopK
};

} // namespace Operon

#endif
