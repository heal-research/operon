// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_OPERATORS_SHAPE_CONSTRAINED_EVALUATOR_HPP
#define OPERON_OPERATORS_SHAPE_CONSTRAINED_EVALUATOR_HPP

#include <atomic>
#include <cstddef>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <tl/expected.hpp>
#include <tuple>
#include <utility>
#include <vector>

#include "operon/core/concepts.hpp"
#include "operon/core/constraint.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operon_export.hpp"

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

// HardReject: an infeasible individual receives the worst value on every objective.
// Penalty: primary objective += PenaltyWeight * violation.
// ExtraObjective: violation is appended as the last objective.
// PenaltyAndExtraObjective: both of the above (NSGA2 only).
enum class ShapeConstraintEnforcement : std::uint8_t {
    HardReject,
    Penalty,
    ExtraObjective,
    PenaltyAndExtraObjective,
};

struct ShapeConstraintPolicy {
    ShapeConstraintEnforcement Enforcement { ShapeConstraintEnforcement::HardReject };
    Operon::Scalar UnknownViolation { 1 };
    Operon::Scalar PenaltyWeight { 1 };
};

// Bound backend. Interval (default): plain interval arithmetic. Affine: affine arithmetic only. Combined:
// intersection of affine and interval. Bisected: interval arithmetic unioned over 2^BisectionDepth sub-boxes.
enum class ShapeBoundMode : std::uint8_t {
    Combined,
    Interval,
    Affine,
    Bisected,
};

struct ShapeBoundOptions {
    ShapeBoundMode Mode { ShapeBoundMode::Interval };
    // Bisected mode: 2^BisectionDepth uniform sub-boxes along the widest referenced axis (leaf count, not
    // recursion levels, hence not tied to SIMD width). Multi-axis trees bisect over a balanced grid evaluated
    // scalar/unbatched, independently capped well below this value.
    int BisectionDepth { 3 };
    // Opt-in TightenRange rescue path for bounds the direct/bisection paths couldn't certify.
    bool UseTightenRangeFallback { false };
};

[[nodiscard]] OPERON_EXPORT auto ValidatePolicy(ShapeConstraintPolicy const& policy, bool isNsga2)
    -> std::optional<std::string>;
[[nodiscard]] OPERON_EXPORT auto ParseShapeEnforcement(std::string const& str) -> ShapeConstraintEnforcement;
// Parses exactly one token: `combined`, `interval`, `affine`, `bisected` (uses `bisectionDepth`) or `bisected:N`
// (N in [0, 20] overrides `bisectionDepth`). Throws std::invalid_argument on anything else, or if
// `bisectionDepth` is outside [0, 20].
[[nodiscard]] OPERON_EXPORT auto ParseShapeBoundOptions(std::string const& str, int bisectionDepth)
    -> ShapeBoundOptions;

// Shape-constraint configuration (constraint set, domain boxes, bound backend and options). It is not an
// evaluator and never evaluates predictions on behalf of search: enforcement passes it
// the exact transform that the evaluated model applies (`Measure`). `MeasureCanonical` is a standalone,
// reporting-only entry point that fits the canonical transform itself (one prediction pass under
// Problem::LinearScalingOmitsNonFinite -- the policy every evaluator, the exporter and
// ShapeConstrainedValueEvaluator certification share).
//
// Bound semantics follow Kronberger et al. 2021 Algorithm 1: bound the model's output and requested partial
// derivatives over the constraint set's domain box; only a constraint the enclosure proves holds everywhere is
// accepted (Sec. 3.1), so overestimation can reject an actually-feasible model. A derivative constraint's bound
// uses BuildVariableGradientDag (tree_diff.hpp) to get a standalone derivative tree.
class OPERON_EXPORT ShapeCertifier {
public:
    // `constraints`' variable names are resolved against `problem`'s Dataset once, at construction -- throws
    // std::invalid_argument if a referenced variable isn't a dataset column, or has no `constraints.Domains`
    // entry. `constraints.Domains` must cover every variable the tree can reference, not just constraint
    // variables: the identity bound and the jet passes evaluate the whole tree. `unknownViolation` (finite,
    // non-negative) is the violation assigned to a constraint whose bound cannot be certified.
    // `bound.BisectionDepth` must be in [0, 20] (std::invalid_argument otherwise).
    ShapeCertifier(gsl::not_null<Operon::Problem const*> problem, gsl::not_null<Operon::ScalarDispatch const*> dtable,
        ShapeConstraintSet constraints, Operon::Scalar unknownViolation = Operon::Scalar { 1 },
        ShapeBoundOptions bound = {});

    // Measures `tree` under exactly `transform` (`nullopt`: scaling disabled); safe to call concurrently.
    [[nodiscard]] auto Measure(Operon::Tree const& tree, std::optional<Operon::LinearScaling> const& transform) const
        -> ShapeConstraintMeasurementSummary;

    // Reporting / standalone use only: fits the canonical transform of `tree` (one interpreter pass; `scratch`,
    // if large enough, is reused as the output buffer) and measures under it. Search never calls this.
    [[nodiscard]] auto MeasureCanonical(Operon::Tree const& tree, Operon::Span<Operon::Scalar> scratch = {}) const
        -> ShapeConstraintMeasurementSummary;

private:
    gsl::not_null<Operon::Problem const*> problem_;
    gsl::not_null<Operon::ScalarDispatch const*> dtable_;
    ShapeConstraintSet constraints_;
    // constraintVarHash_[i] is constraints_.Constraints[i].Variable's resolved
    // Dataset hash (default Operon::Hash{} for an Identity constraint, which
    // has no variable) — resolved once at construction so Measure() never
    // does string lookups on the hot path.
    Operon::Vector<Operon::Hash> constraintVarHash_;
    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domainsByHash_;
    Operon::Scalar unknownViolation_ { 1 };
    ShapeBoundOptions bound_ {};
};

// Everything a shape-constrained adapter needs besides its base evaluator.
struct ShapeConstraintSpec {
    ShapeConstraintSet Constraints;
    ShapeConstraintPolicy Policy {};
    // Structural post-objectives (e.g. tree length), appended after the primary score in this order. Restricted to
    // TreePropertyEvaluator, whose prediction phase is sealed, so a post-objective can never add a prediction pass.
    std::vector<gsl::not_null<TreePropertyEvaluator const*>> Structural {};
    // Fitness assigned to every objective of a hard-rejected individual.
    double WorstValue { 1.0 };
    ShapeBoundOptions Bound {};
};

namespace detail {

    // The evaluator-independent half of every ShapeConstrainedValueEvaluator: the certifier, the policy, the
    // structural post-objectives and all enforcement and bookkeeping that does not depend on the base evaluator
    // type, so the adapter template stays a thin shell. It is an implementation detail, here only because the
    // adapter template needs its complete type: it is not an evaluator and cannot be constructed or driven on its
    // own. Its public members are the read/configure access generic code (CLI, probes) reaches with
    // `dynamic_cast<detail::ShapeConstraintEnforcer const*>(evaluator)`; the enforcement steps are protected and
    // called only by the adapter.
    class OPERON_EXPORT ShapeConstraintEnforcer {
    public:
        ShapeConstraintEnforcer(ShapeConstraintEnforcer const&) = delete;
        auto operator=(ShapeConstraintEnforcer const&) -> ShapeConstraintEnforcer& = delete;
        ShapeConstraintEnforcer(ShapeConstraintEnforcer&&) = delete;
        auto operator=(ShapeConstraintEnforcer&&) -> ShapeConstraintEnforcer& = delete;

        [[nodiscard]] auto Certifier() const noexcept -> ShapeCertifier const& { return certifier_; }
        [[nodiscard]] auto Certifier() noexcept -> ShapeCertifier& { return certifier_; }
        [[nodiscard]] auto Policy() const noexcept -> ShapeConstraintPolicy const& { return policy_; }

        [[nodiscard]] auto WorstValue() const noexcept -> double { return worstValue_; }
        void SetWorstValue(double value) { worstValue_ = value; }

        // Individuals rejected by HardReject so far (paper's Sec. 5.1 "constraint violations" figure). Accumulates
        // over this evaluator's lifetime; EvaluatorBase::Reset() does not clear it.
        [[nodiscard]] auto Violations() const noexcept -> std::size_t { return violations_.load(); }

    protected:
        // Throws std::invalid_argument if a structural objective has more than one objective, if the policy is
        // invalid for both GP and NSGA2, or if the constraints fail ShapeCertifier validation.
        ShapeConstraintEnforcer(gsl::not_null<Operon::Problem const*> problem,
            gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSpec spec);
        ~ShapeConstraintEnforcer() = default;

        // Width of the constrained objective vector: primary, structural objectives, optional violation.
        [[nodiscard]] auto ShapeObjectiveCount() const noexcept -> std::size_t;

        // Prepares the structural objectives.
        void ShapePrepare(Operon::Span<Individual const> pop) const;

        // Residual evaluations, Jacobian evaluations and cost-function time accumulated by the structural
        // objectives.
        [[nodiscard]] auto ShapeStructuralStats() const -> std::tuple<std::size_t, std::size_t, std::size_t>;

        // Measures `tree` under exactly the transform the evaluated prediction carries, before the base
        // evaluator's score rewrites the buffer. Throws std::logic_error if `evaluated` is empty.
        [[nodiscard]] auto ShapeCertify(Operon::Tree const& tree, std::optional<EvaluatedBuffer> const& evaluated) const
            -> ShapeConstraintMeasurementSummary;

        // HardReject of an infeasible summary: counts it and returns WorstValue for the whole vector (structural
        // objectives are not evaluated). Empty when the summary does not reject.
        [[nodiscard]] auto ShapeReject(ShapeConstraintMeasurementSummary const& summary) const
            -> std::optional<EvaluatorBase::ReturnType>;

        // Assembles the constrained vector from the base evaluator's single-objective `primary` score: a failed or
        // non-finite score gives ErrMax for the whole vector; Penalty adds PenaltyWeight * violation to the
        // primary; structural objectives follow, through their operator(); and ExtraObjective appends the
        // violation last (NSGP order: error, length, violation).
        [[nodiscard]] auto ShapeAssemble(Operon::RandomGenerator& rng, Operon::Individual const& ind,
            Operon::Span<Operon::Scalar> buf, ShapeConstraintMeasurementSummary const& summary,
            EvaluatorBase::ReturnType const& primary) const -> EvaluatorBase::ReturnType;

    private:
        std::vector<gsl::not_null<TreePropertyEvaluator const*>> structural_;
        ShapeCertifier certifier_;
        ShapeConstraintPolicy policy_;
        double worstValue_ { 1.0 };
        mutable std::atomic_size_t violations_ { 0 };
    };

} // namespace detail

// The one-pass shape-constrained composition, as a derived typed adapter over a concrete value evaluator
// `BaseEvaluator` (an interpreter evaluator with any scoring metric, MDL/FBF/likelihood, or the JIT evaluator).
// The adapter *is* the value evaluator and owns the complete constrained objective vector
//
//     [ primary (+ penalty), structural post-objectives..., (violation) ]
//
// so that prediction evaluation, the fitted transform, shape measurement and enforcement all happen inside one
// operator() invocation of the one evaluator object:
//
//  * the base evaluator's prediction phase runs exactly once and yields the exact transform it will apply;
//  * the certifier measures the model under that transform before the base evaluator's score rewrites the buffer;
//  * HardReject returns WorstValue for the whole vector (structural objectives are not evaluated), a failed
//    prediction or score returns ErrMax for the whole vector, Penalty adds PenaltyWeight * violation to the
//    primary score, and ExtraObjective appends the measured violation last (NSGP order: error, length, violation).
//
// It exposes no execution method besides the inherited, final EvaluatorBase::operator(): its overrides of the
// protected phases call `BaseEvaluator`'s own phases on `this`, so no evaluator reaches into another and no
// runtime child is erased. Structural post-objectives are TreePropertyEvaluators, whose prediction phase is sealed,
// so they can never add a second prediction pass.
//
// Fitness describes the evaluated model, error and shape alike. Under non-Lamarckian local search the optimized
// coefficients are what gets scored and certified; the restored genotype is not re-certified (use
// ShapeCertifier::MeasureCanonical to certify an exported model). An adapter bypasses the structural
// transposition cache (StructuralFitnessReusable() is false) because a structural hit carries no coefficient
// identity.
//
// Adapters cannot nest: a constrained adapter is not a Concepts::ValueEvaluator (it sets `IsShapeConstrained`), and
// the class is final. Build one over the concrete evaluator before erasing it to EvaluatorBase (see
// MakeValueEvaluator).
template <Concepts::ValueEvaluator BaseEvaluator>
class ShapeConstrainedValueEvaluator final : public BaseEvaluator, public detail::ShapeConstraintEnforcer {
public:
    static constexpr bool IsShapeConstrained = true;

    // `baseArgs` construct the `BaseEvaluator`. `dtable` is the dispatch table the certifier fits canonical
    // transforms with (reporting-only `MeasureCanonical`).
    template <typename... BaseArgs>
    explicit ShapeConstrainedValueEvaluator(
        ShapeConstraintSpec spec, gsl::not_null<Operon::ScalarDispatch const*> dtable, BaseArgs&&... baseArgs)
        : BaseEvaluator(std::forward<BaseArgs>(baseArgs)...)
        , detail::ShapeConstraintEnforcer(this->GetProblem(), dtable, std::move(spec))
    {
    }

    [[nodiscard]] auto ObjectiveCount() const -> std::size_t override { return this->ShapeObjectiveCount(); }

    [[nodiscard]] auto StructuralFitnessReusable() const -> bool override { return false; }

    // Prepares the base evaluator and the structural objectives.
    auto Prepare(Operon::Span<Individual const> pop) const -> void override
    {
        BaseEvaluator::Prepare(pop);
        this->ShapePrepare(pop);
    }

    auto Stats() const -> std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> override
    {
        auto [residual, jacobian, calls, costFunctionTime] = BaseEvaluator::Stats();
        auto const [structuralResidual, structuralJacobian, structuralTime] = this->ShapeStructuralStats();
        return std::tuple { residual + structuralResidual, jacobian + structuralJacobian, calls,
            costFunctionTime + structuralTime };
    }

    auto BudgetExhausted() const -> bool override
    {
        auto const [residual, jacobian, calls, costFunctionTime] = Stats();
        return residual + jacobian >= this->Budget();
    }

protected:
    auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override
    {
        auto const summary = this->ShapeCertify(ind.Genotype, evaluated);
        if (auto rejected = this->ShapeReject(summary)) {
            return std::move(*rejected);
        }
        auto const primary = BaseEvaluator::Score(rng, ind, buf, std::move(evaluated));
        return this->ShapeAssemble(rng, ind, buf, summary, primary);
    }
};

// Builds `Leaf` -- a value evaluator constructed from `leafArgs` -- or, when `shape` is given, its
// ShapeConstrainedValueEvaluator, and returns it type-erased. The typed adapter is constructed before the erasure,
// so the set of supported branches stays the closed set of callers' `Leaf` types.
template <Concepts::ValueEvaluator Leaf, typename... LeafArgs>
[[nodiscard]] auto MakeValueEvaluator(std::optional<ShapeConstraintSpec> const& shape,
    gsl::not_null<Operon::ScalarDispatch const*> dtable, LeafArgs&&... leafArgs) -> std::unique_ptr<EvaluatorBase>
{
    if (shape) {
        return std::make_unique<ShapeConstrainedValueEvaluator<Leaf>>(
            ShapeConstraintSpec { *shape }, dtable, std::forward<LeafArgs>(leafArgs)...);
    }
    return std::make_unique<Leaf>(std::forward<LeafArgs>(leafArgs)...);
}

} // namespace Operon

#endif
