// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_OPERATORS_SHAPE_CONSTRAINED_EVALUATOR_HPP
#define OPERON_OPERATORS_SHAPE_CONSTRAINED_EVALUATOR_HPP

#include "operon/core/constraint.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/evaluator.hpp"
#include <optional>
#include <stdexcept>
#include <string>

namespace tf {
class Executor;
} // NOLINT(readability-identifier-naming) -- Taskflow's own namespace

namespace Operon {

struct ShapeConstraintMeasurement {
    bool Certified { false };
    std::optional<std::pair<Operon::Scalar, Operon::Scalar>> Bound {};
    Operon::Scalar Violation { 0 };
};

struct ShapeConstraintMeasurementSummary {
    bool Feasible { true };
    Operon::Scalar Violation { 0 };
    Operon::Vector<ShapeConstraintMeasurement> Measurements {};
};

enum class ShapeConstraintEnforcement : unsigned {
    None = 0U,
    HardReject = 1U << 0U,
    Penalty = 1U << 1U,
    ExtraObjective = 1U << 2U,
    FeasibilityFirst = 1U << 3U,
};

[[nodiscard]] constexpr auto operator|(ShapeConstraintEnforcement lhs, ShapeConstraintEnforcement rhs) noexcept
    -> ShapeConstraintEnforcement
{
    return static_cast<ShapeConstraintEnforcement>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

[[nodiscard]] constexpr auto operator&(ShapeConstraintEnforcement lhs, ShapeConstraintEnforcement rhs) noexcept
    -> ShapeConstraintEnforcement
{
    return static_cast<ShapeConstraintEnforcement>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

[[nodiscard]] constexpr auto HasFlag(ShapeConstraintEnforcement value, ShapeConstraintEnforcement flag) noexcept -> bool
{
    return (value & flag) != ShapeConstraintEnforcement::None;
}

struct ShapeConstraintPolicy {
    ShapeConstraintEnforcement Enforcement { ShapeConstraintEnforcement::HardReject };
    Operon::Scalar UnknownViolation { 1 };
    Operon::Scalar PenaltyWeight { 1 };
};

// Backend(s) TryAffineBound uses to bound a constraint. Interval (default) uses
// plain interval arithmetic only -- ties or beats Combined on the paper's full
// matrix (see RESULTS.md) at ~24% less work per Measure() call, with none of
// affine's ill-conditioned/uncertified failure modes. Combined intersects
// affine and interval; Affine isolates the affine backend alone (confirmed
// non-competitive -- kept only so run_bound_mode_ablation*.sh can still
// reproduce that published cell by name). Bisected recursively bisects the
// domain and unions per-sub-box results; currently only supported combined
// with Interval.
enum class ShapeBoundMode : unsigned {
    Combined = 0U,
    Interval = 1U << 0U,
    Affine = 1U << 1U,
    Bisected = 1U << 2U,
};

[[nodiscard]] constexpr auto operator|(ShapeBoundMode lhs, ShapeBoundMode rhs) noexcept -> ShapeBoundMode
{
    return static_cast<ShapeBoundMode>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

[[nodiscard]] constexpr auto operator&(ShapeBoundMode lhs, ShapeBoundMode rhs) noexcept -> ShapeBoundMode
{
    return static_cast<ShapeBoundMode>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

[[nodiscard]] constexpr auto HasFlag(ShapeBoundMode value, ShapeBoundMode flag) noexcept -> bool
{
    return (value & flag) != ShapeBoundMode::Combined;
}

// Tuning knobs for the affine/interval bound machinery. Defaults match this file's previous behavior exactly.
struct ShapeBoundOptions {
    // Interval-only bisection (ShapeBoundMode::Bisected): 2^BisectionDepth uniform sub-boxes along the widest
    // referenced axis (leaf count, not recursion levels, hence not tied to SIMD width). Multi-axis trees
    // bisect over a balanced grid evaluated scalar/unbatched, independently capped well below this value.
    int BisectionDepth { 3 };
    // Affine-mode fallback: max bisection depth when the direct
    // affine/interval intersection fails on the whole domain. 0 disables it.
    int AffineBisectionMaxDepth { 0 };
    // Flags an affine bound as uncertified when the float32 rounding-error
    // floor implied by the largest intermediate center exceeds this many
    // times the final radius.
    Operon::Scalar AffineIllConditionedThreshold { 4 };
    // Opt-in TightenRange rescue path for bounds the direct/bisection paths
    // couldn't certify.
    bool UseTightenRangeFallback { false };
};
inline void ValidateShapeBoundOptions(ShapeBoundOptions const& options)
{
    if (options.BisectionDepth < 0 || options.BisectionDepth > 20 || options.AffineBisectionMaxDepth < 0
        || options.AffineBisectionMaxDepth > 20) {
        throw std::invalid_argument("bisection depths must be in [0, 20]");
    }
}

[[nodiscard]] OPERON_EXPORT auto ValidatePolicy(ShapeConstraintPolicy const& policy, bool isNsga2)
    -> std::optional<std::string>;
[[nodiscard]] OPERON_EXPORT auto ParseShapeEnforcement(std::string const& str) -> ShapeConstraintEnforcement;
// Rejects Interval+Affine together, or Bisected without Interval. Shared by
// ParseShapeBoundMode and both SetBoundMode setters below so a
// programmatically-constructed mode is held to the same contract as a
// string-parsed one -- constructing ShapeBoundMode values directly (not
// through the parser) previously bypassed this check entirely.
[[nodiscard]] OPERON_EXPORT auto ValidateShapeBoundMode(ShapeBoundMode mode) -> std::optional<std::string>;
[[nodiscard]] OPERON_EXPORT auto ParseShapeBoundMode(std::string const& str) -> ShapeBoundMode;

// Wraps an inner EvaluatorBase (typically NMSE-with-linear-scaling) with the shape-constraint check from
// Kronberger et al. 2021 Algorithm 1: bound the model's output and requested partial derivatives over the
// constraint set's domain box via AffineEvaluator (tighter than the paper's plain interval arithmetic); if a
// bound proves a constraint can't hold everywhere in the box, every objective gets WorstValue() instead of
// calling the inner evaluator. A derivative constraint's bound uses BuildVariableGradientDag (tree_diff.hpp) to
// get a standalone derivative tree, evaluated the same way as the identity case (see the .cpp).
//
// Pessimistic in the paper's own sense (Sec. 3.1): only accepts a constraint the enclosure proves holds
// everywhere, so overestimation can reject an actually-feasible model. Documented tradeoff, not a bug.
class OPERON_EXPORT ShapeConstrainedEvaluator final : public EvaluatorBase {
public:
    // `constraints`' variable names are resolved against `evaluator`'s Problem/Dataset once, at construction --
    // throws std::invalid_argument if a referenced variable isn't a dataset column, or has no `constraints.Domains`
    // entry. `constraints.Domains` must cover every variable the tree can reference, not just constraint
    // variables: a derivative bound still walks the whole original tree internally (BuildVariableGradientDag's
    // dag carries the full tree as a prefix).
    ShapeConstrainedEvaluator(gsl::not_null<EvaluatorBase const*> evaluator,
        gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSet constraints);

    [[nodiscard]] auto WorstValue() const noexcept -> double { return worstValue_; }
    void SetWorstValue(double value) { worstValue_ = value; }

    [[nodiscard]] auto BoundMode() const noexcept -> ShapeBoundMode { return boundMode_; }
    // Throws std::invalid_argument if `mode` fails ValidateShapeBoundMode
    // (e.g. Bisected without Interval) -- see that function's comment.
    void SetBoundMode(ShapeBoundMode mode);

    [[nodiscard]] auto BoundOptions() const noexcept -> ShapeBoundOptions const& { return boundOptions_; }
    // Throws std::invalid_argument if either bisection depth fails ValidateShapeBoundOptions (negative, or deep
    // enough that the 2^depth leaf arithmetic stops being exact). Also clears the feasibility cache: its memo key
    // covers the bound mode but NOT the options, so entries computed under the previous depths would otherwise
    // keep answering Feasible() as if those depths were still set.
    void SetBoundOptions(ShapeBoundOptions options)
    {
        ValidateShapeBoundOptions(options);
        boundOptions_ = options;
        feasibleCache_.Clear();
    }

    // The tf::Executor Prepare() uses to parallelize its Feasible() pre-warm across `pop` -- normally the
    // caller's own GP/NSGA2 executor (see Reporter for the same reuse pattern), not a private one. Unset
    // (nullptr, default) means Prepare() runs sequentially.
    void SetExecutor(tf::Executor& executor) noexcept { taskExecutor_ = &executor; }

    // Individuals rejected by the constraint check so far (paper's Sec. 5.1 "constraint violations" figure).
    // Accumulates over this evaluator's lifetime; EvaluatorBase::Reset() does not clear it.
    [[nodiscard]] auto Violations() const noexcept -> std::size_t { return violations_.load(); }

    // Delegates to the wrapped evaluator's Evaluate, then fits (a,b) from those values
    // (or its own tree-overload pass, if the wrapped evaluator isn't value-based) and
    // populates feasibleCache_ for this tree.
    auto Evaluate(Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> override;

    // Feasible (per the cache Evaluate just populated) -> wrapped evaluator's Score;
    // infeasible -> WorstValue.
    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const ->
        typename EvaluatorBase::ReturnType override;

    auto ObjectiveCount() const -> std::size_t override { return evaluator_->ObjectiveCount(); }

    // Delegates to the inner evaluator's Prepare(), then bulk-computes and caches Feasible() for every individual
    // in `pop`, parallelized over `taskExecutor_` (see ParallelForPopulation in the .cpp for the corun()
    // rationale). Cleared and rebuilt each call, so it always reflects the most recent `pop`.
    auto Prepare(Operon::Span<Individual const> pop) const -> void override;

    auto Stats() const -> std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> override
    {
        return evaluator_->Stats();
    }
    auto BudgetExhausted() const -> bool override { return evaluator_->BudgetExhausted(); }

    // The same box-bounding check Evaluate() uses, exposed standalone so a caller can ask whether a tree
    // satisfies the constraints without scoring it (and without counting toward Violations()/CallCount). Not
    // the paper's separate Sec. 5.1 point-sampling violation-rate methodology.
    //
    // Checks the Prepare()-populated cache first; a miss computes and stores the result, safe concurrently.
    [[nodiscard]] auto Feasible(Operon::Tree const& tree) const -> bool;
    [[nodiscard]] auto Measure(Operon::Tree const& tree, Operon::Scalar unknownViolation = Operon::Scalar { 1 }) const
        -> ShapeConstraintMeasurementSummary;

private:
    gsl::not_null<EvaluatorBase const*> evaluator_;
    gsl::not_null<Operon::ScalarDispatch const*> dtable_;
    ShapeConstraintSet constraints_;
    // constraintVarHash_[i] is constraints_.Constraints[i].Variable's resolved
    // Dataset hash (default Operon::Hash{} for an Identity constraint, which
    // has no variable) — resolved once at construction so Feasible() never
    // does string lookups on the hot path.
    Operon::Vector<Operon::Hash> constraintVarHash_;
    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domainsByHash_;
    double worstValue_ { 1.0 };
    ShapeBoundMode boundMode_ { ShapeBoundMode::Interval };
    ShapeBoundOptions boundOptions_ {};
    tf::Executor* taskExecutor_ { nullptr };
    mutable std::atomic_size_t violations_ { 0 };

    struct FeasibleData {
        ShapeConstraintMeasurementSummary Value {};
    };
    mutable ZobristCache<CacheEntry<FeasibleData>> feasibleCache_;
};

// Computes shape-constraint violation as a standalone objective from a
// Problem plus ScalarDispatch; it does not wrap or delegate to another evaluator.
class OPERON_EXPORT ShapeViolationEvaluator final : public EvaluatorBase {
public:
    ShapeViolationEvaluator(gsl::not_null<Operon::Problem const*> problem,
        gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSet constraints,
        Operon::Scalar weight = Operon::Scalar { 1 }, Operon::Scalar unknownViolation = Operon::Scalar { 1 });

    [[nodiscard]] auto Weight() const noexcept -> Operon::Scalar { return weight_; }
    [[nodiscard]] auto UnknownViolation() const noexcept -> Operon::Scalar { return unknownViolation_; }
    [[nodiscard]] auto BoundMode() const noexcept -> ShapeBoundMode { return boundMode_; }
    // See ShapeConstrainedEvaluator::SetBoundMode -- same validation contract.
    void SetBoundMode(ShapeBoundMode mode);
    [[nodiscard]] auto BoundOptions() const noexcept -> ShapeBoundOptions const& { return boundOptions_; }
    // See ShapeConstrainedEvaluator::SetBoundOptions -- same validation contract, and Measure()'s memo key covers
    // the bound mode but not the options, so the measurement cache is cleared here too.
    void SetBoundOptions(ShapeBoundOptions options)
    {
        ValidateShapeBoundOptions(options);
        boundOptions_ = options;
        measurementCache_.Clear();
    }
    [[nodiscard]] auto RawViolation(Operon::Tree const& tree, Operon::Span<Operon::Scalar> scratch = {}) const
        -> Operon::Scalar;
    [[nodiscard]] auto Measure(Operon::Tree const& tree, Operon::Span<Operon::Scalar> scratch = {}) const
        -> ShapeConstraintMeasurementSummary;

    // See ShapeConstrainedEvaluator::SetExecutor — Prepare()'s population
    // Measure() pre-warm reuses the caller's executor the same way.
    void SetExecutor(tf::Executor& executor) noexcept { taskExecutor_ = &executor; }

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> /*evaluated*/) const ->
        typename EvaluatorBase::ReturnType override;
    auto ObjectiveCount() const -> std::size_t override { return 1; }
    auto Prepare(Operon::Span<Individual const> pop) const -> void override;

private:
    gsl::not_null<Operon::Problem const*> problem_;
    gsl::not_null<Operon::ScalarDispatch const*> dtable_;
    ShapeConstraintSet constraints_;
    Operon::Vector<Operon::Hash> constraintVarHash_;
    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domainsByHash_;
    Operon::Scalar weight_ { 1 };
    Operon::Scalar unknownViolation_ { 1 };
    ShapeBoundMode boundMode_ { ShapeBoundMode::Interval };
    ShapeBoundOptions boundOptions_ {};
    tf::Executor* taskExecutor_ { nullptr };

    struct MeasurementData {
        ShapeConstraintMeasurementSummary Value {};
    };
    mutable ZobristCache<CacheEntry<MeasurementData>> measurementCache_;
};

} // namespace Operon

#endif
