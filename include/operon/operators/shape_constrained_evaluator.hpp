// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_OPERATORS_SHAPE_CONSTRAINED_EVALUATOR_HPP
#define OPERON_OPERATORS_SHAPE_CONSTRAINED_EVALUATOR_HPP

#include "operon/core/constraint.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/operators/evaluator.hpp"
#include <optional>
#include <string>

namespace tf { class Executor; } // NOLINT(readability-identifier-naming) -- Taskflow's own namespace

namespace Operon {

struct ShapeConstraintMeasurement {
    bool Certified{false};
    std::optional<std::pair<Operon::Scalar, Operon::Scalar>> Bound{};
    Operon::Scalar Violation{0};
};

struct ShapeConstraintMeasurementSummary {
    bool Feasible{true};
    Operon::Scalar Violation{0};
    Operon::Vector<ShapeConstraintMeasurement> Measurements{};
};

enum class ShapeConstraintEnforcement : unsigned {
    None = 0U,
    HardReject = 1U << 0U,
    Penalty = 1U << 1U,
    ExtraObjective = 1U << 2U,
    FeasibilityFirst = 1U << 3U,
};

[[nodiscard]] constexpr auto operator|(ShapeConstraintEnforcement lhs, ShapeConstraintEnforcement rhs) noexcept -> ShapeConstraintEnforcement
{
    return static_cast<ShapeConstraintEnforcement>(static_cast<unsigned>(lhs) | static_cast<unsigned>(rhs));
}

[[nodiscard]] constexpr auto operator&(ShapeConstraintEnforcement lhs, ShapeConstraintEnforcement rhs) noexcept -> ShapeConstraintEnforcement
{
    return static_cast<ShapeConstraintEnforcement>(static_cast<unsigned>(lhs) & static_cast<unsigned>(rhs));
}

[[nodiscard]] constexpr auto HasFlag(ShapeConstraintEnforcement value, ShapeConstraintEnforcement flag) noexcept -> bool
{
    return (value & flag) != ShapeConstraintEnforcement::None;
}

struct ShapeConstraintPolicy {
    ShapeConstraintEnforcement Enforcement{ShapeConstraintEnforcement::HardReject};
    Operon::Scalar UnknownViolation{1};
    Operon::Scalar PenaltyWeight{1};
};

// Which arithmetic backend(s) TryAffineBound uses to bound a constraint's
// tree/derivative over its domain box. `Combined` is the default and the
// only mode with any published measurement behind it so far: affine,
// intersected with plain interval whenever both succeed (see
// TryAffineBoundDirect's comment for why that intersection is sound).
// `IntervalOnly`/`AffineOnly` exist for isolating each backend's actual
// contribution to feasibility/NMSE outcomes -- prompted by an earlier
// finding that affine rarely decides the `Combined` intersection in
// practice (see intersectLoFromAffine/intersectLoFromInterval under
// OPERON_SHAPE_BOUND_STATS=1), which raised the question of whether it's
// pulling its weight at all. `AffineOnly` still falls back to interval for
// cases affine structurally can't represent (a zero-crossing denominator,
// a non-integer-exponent domain check) -- those are representability
// gaps, not the tightness contribution the intersection step measures, so
// removing them would conflate two different questions.
enum class ShapeBoundMode : unsigned {
    Combined = 0U,
    IntervalOnly = 1U,
    AffineOnly = 2U,
};

[[nodiscard]] OPERON_EXPORT auto ValidatePolicy(ShapeConstraintPolicy const& policy, bool isNsga2) -> std::optional<std::string>;
[[nodiscard]] OPERON_EXPORT auto ParseShapeEnforcement(std::string const& str) -> ShapeConstraintEnforcement;
[[nodiscard]] OPERON_EXPORT auto ParseShapeBoundMode(std::string const& str) -> ShapeBoundMode;

// Wraps an inner EvaluatorBase (typically an NMSE-with-linear-scaling
// Evaluator, matching Kronberger et al. 2021's own fitness setup) with the
// shape-constraint check from that paper's Algorithm 1 `Evaluate`
// function: bound the model's output and the requested partial
// derivatives over the constraint set's domain box via AffineEvaluator
// (tighter than the paper's own plain interval arithmetic); if any bound
// proves a constraint can't hold everywhere in the box, every objective
// gets `WorstValue()` instead of calling the inner evaluator at all — the
// same "worst possible fitness for an infeasible candidate" rule the
// paper uses (NMSE's worst case is exactly 1.0 under its own linear
// scaling convention, hence WorstValue defaulting to 1.0 rather than
// EvaluatorBase::ErrMax, which is a different, evaluator-agnostic
// "non-finite" sentinel used elsewhere in this codebase).
//
// Bound computation for a derivative constraint uses
// BuildVariableGradientDag (tree_diff.hpp) to get a standalone derivative
// expression tree for the requested variable, then evaluates that tree's
// affine enclosure exactly like the "id" case — see the .cpp for the
// slice-and-wrap-in-a-Tree idiom this shares with the tree_diff tests.
//
// This is a pessimistic check in the paper's own sense (Sec. 3.1): a
// constraint is accepted only if the *enclosure* proves it holds
// everywhere in the box, so a real conservatism gap (affine/interval
// overestimation) can reject an actually-feasible model. That's the
// documented tradeoff, not a bug to work around here.
class OPERON_EXPORT ShapeConstrainedEvaluator final : public EvaluatorBase {
public:
    // `constraints`' variable names are resolved against `evaluator`'s own
    // Problem/Dataset once, at construction — throws std::invalid_argument
    // if a referenced variable name isn't a column in that dataset, or if
    // a variable used by a constraint has no entry in `constraints.Domains`.
    //
    // `constraints.Domains` must cover every variable that can appear
    // anywhere in a scored tree, not just the ones named by a constraint's
    // own `Variable` — even a derivative/second-derivative bound's affine
    // evaluation walks the *whole* original tree internally (see the .cpp:
    // BuildVariableGradientDag's dag always carries the full original tree
    // as a prefix, so bounding a one-variable derivative still needs
    // domain bounds for every other variable the tree references). This
    // matches how the paper's own problems specify one shared input-space
    // box per problem (see operon-publications' shape-constraints-
    // reproduction/problems.yml), not a per-constraint domain.
    ShapeConstrainedEvaluator(gsl::not_null<EvaluatorBase const*> evaluator,
        gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSet constraints);

    [[nodiscard]] auto WorstValue() const noexcept -> double { return worstValue_; }
    void SetWorstValue(double value) { worstValue_ = value; }

    [[nodiscard]] auto BoundMode() const noexcept -> ShapeBoundMode { return boundMode_; }
    void SetBoundMode(ShapeBoundMode mode) noexcept { boundMode_ = mode; }

    // The tf::Executor Prepare() uses to parallelize its population-wide
    // Feasible() pre-warm -- the SAME executor the caller's GP/NSGA2 loop
    // already created (see cli/source/operon_gp.cpp: `tf::Executor
    // executor(threads);` threaded into both `gp.Run(executor, ...)` and
    // `Reporter::operator()`), not a private one owned by this class. A
    // private/self-owned executor was tried first and measured to not
    // help (~3x higher CPU, no wall-clock change on a real run) while
    // needlessly doubling the machine's thread count; reusing the
    // caller's is both correct (one thread pool, sized once to
    // --threads) and matches this codebase's existing pattern (see
    // Reporter, which also takes a `tf::Executor&` and calls
    // `executor.corun(...)`, never owns one). Not set (nullptr) means
    // Prepare() runs its population loop sequentially -- the safe
    // default for any caller (e.g. a test) that constructs this evaluator
    // without wiring an executor. Named distinctly from `evaluator_`
    // (the wrapped inner EvaluatorBase) to avoid visual confusion.
    void SetExecutor(tf::Executor& executor) noexcept { taskExecutor_ = &executor; }

    // Number of individuals rejected by the constraint check so far
    // (paper's Sec. 5.1 "constraint violations" figure). Accumulates over
    // this evaluator's lifetime, not per-generation — EvaluatorBase::Reset()
    // is non-virtual and does NOT touch this counter, so a
    // Reset()-and-continue caller (e.g. --resume) will keep accumulating
    // across the reset rather than starting over.
    [[nodiscard]] auto Violations() const noexcept -> std::size_t { return violations_.load(); }

    auto Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;

    auto ObjectiveCount() const -> std::size_t override { return evaluator_->ObjectiveCount(); }

    // Delegates to the inner evaluator's own Prepare(), then bulk-computes
    // and caches Feasible() for every individual in `pop`. The outer GP/
    // NSGA2 loop still schedules this as one non-parallel taskflow task
    // (the same single-task context DiversityEvaluator::Prepare relies on
    // for its own non-thread-safe population snapshot, and Reporter also
    // runs from) — Prepare() parallelizes *within* that task by building
    // a local tf::Taskflow over the population and running it via
    // `taskExecutor_->corun(...)` on the caller's own executor (see
    // SetExecutor), exactly like Reporter does for its own nested
    // taskflow. corun() (not run().get()) is required here specifically
    // because Prepare() is itself already running as a task on that same
    // executor -- run().get() would block a worker thread waiting on a
    // taskflow that needs a free worker to make progress, a classic
    // nested-executor deadlock risk; corun() has the calling thread join
    // in as a worker on the nested graph instead of just blocking.
    // feasibleCache_ is already a shard-locked parallel map whose
    // LazyEmplace serializes same-hash callers, and Feasible()/Measure()/
    // FitLinearScaling only read shared immutable state (Problem/Dataset/
    // constraints), so concurrent Feasible() calls are safe. The cache is
    // cleared and rebuilt on every call, so it always reflects `pop` as
    // of the most recent Prepare(). Runs sequentially if no executor was
    // ever set.
    auto Prepare(Operon::Span<Individual const> pop) const -> void override;

    auto Stats() const -> std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> override { return evaluator_->Stats(); }
    auto BudgetExhausted() const -> bool override { return evaluator_->BudgetExhausted(); }

    // The same box-bounding check Evaluate() uses internally, exposed
    // standalone so a caller can ask "does this tree satisfy the
    // constraints" without going through the full scoring path (and
    // without it counting toward Violations()/CallCount). Note this is
    // NOT the paper's separate Sec. 5.1 post-hoc violation-rate
    // methodology, which samples concrete points numerically rather than
    // reasoning about the whole domain box at once — that's a different,
    // point-sampling check a caller would build on top of the ordinary
    // Interpreter, not this method.
    //
    // Checks the Prepare()-populated cache first (thread-safe, so this is
    // also safe to call concurrently from Evaluate() on freshly generated
    // offspring that Prepare() never saw this generation -- a cache miss
    // there just computes and stores the result under this tree's own
    // content hash, same as any other miss).
    [[nodiscard]] auto Feasible(Operon::Tree const& tree) const -> bool;
    [[nodiscard]] auto Measure(Operon::Tree const& tree, Operon::Scalar unknownViolation = Operon::Scalar{1}) const -> ShapeConstraintMeasurementSummary;

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
    double worstValue_{1.0};
    ShapeBoundMode boundMode_{ShapeBoundMode::Combined};
    // Non-owning; set via SetExecutor(). nullptr means Prepare() runs
    // sequentially -- see Prepare()'s doc comment.
    tf::Executor* taskExecutor_{nullptr};
    mutable std::atomic_size_t violations_{0};

    struct FeasibleData {
        ShapeConstraintMeasurementSummary Value{};
    };
    mutable ZobristCache<CacheEntry<FeasibleData>> feasibleCache_;
};

// Computes shape-constraint violation as a standalone objective from a
// Problem plus ScalarDispatch; it does not wrap or delegate to another evaluator.
class OPERON_EXPORT ShapeViolationEvaluator final : public EvaluatorBase {
public:
    ShapeViolationEvaluator(gsl::not_null<Operon::Problem const*> problem,
        gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSet constraints,
        Operon::Scalar weight = Operon::Scalar{1}, Operon::Scalar unknownViolation = Operon::Scalar{1});

    [[nodiscard]] auto Weight() const noexcept -> Operon::Scalar { return weight_; }
    [[nodiscard]] auto UnknownViolation() const noexcept -> Operon::Scalar { return unknownViolation_; }
    [[nodiscard]] auto BoundMode() const noexcept -> ShapeBoundMode { return boundMode_; }
    void SetBoundMode(ShapeBoundMode mode) noexcept { boundMode_ = mode; }
    [[nodiscard]] auto RawViolation(Operon::Tree const& tree) const -> Operon::Scalar;
    [[nodiscard]] auto Measure(Operon::Tree const& tree) const -> ShapeConstraintMeasurementSummary;

    // See ShapeConstrainedEvaluator::SetExecutor — Prepare()'s population
    // Measure() pre-warm reuses the caller's executor the same way.
    void SetExecutor(tf::Executor& executor) noexcept { taskExecutor_ = &executor; }

    auto Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;
    auto ObjectiveCount() const -> std::size_t override { return 1; }
    auto Prepare(Operon::Span<Individual const> pop) const -> void override;

private:
    gsl::not_null<Operon::Problem const*> problem_;
    gsl::not_null<Operon::ScalarDispatch const*> dtable_;
    ShapeConstraintSet constraints_;
    Operon::Vector<Operon::Hash> constraintVarHash_;
    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domainsByHash_;
    Operon::Scalar weight_{1};
    Operon::Scalar unknownViolation_{1};
    ShapeBoundMode boundMode_{ShapeBoundMode::Combined};
    tf::Executor* taskExecutor_{nullptr};

    struct MeasurementData {
        ShapeConstraintMeasurementSummary Value{};
    };
    mutable ZobristCache<CacheEntry<MeasurementData>> measurementCache_;
};

} // namespace Operon

#endif
