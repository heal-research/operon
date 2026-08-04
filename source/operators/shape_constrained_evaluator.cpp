// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/shape_constrained_evaluator.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include <fmt/format.h>
#include <tl/expected.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/core/tree_hash.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/operators/linear_scaling.hpp"

namespace Operon {

namespace {

constexpr std::size_t NoGrad = std::numeric_limits<std::size_t>::max();

using Interval = AffineEvaluator::Interval;
using BoundResult = tl::expected<Interval, std::string>;

// Slices a VariableGradientDag root into a standalone Tree (same Ref-node
// convention as JacobianDag/HessianDag — see the tree_diff tests for the
// same idiom). std::nullopt means the derivative is identically zero, not
// an error.
auto SliceToTree(VariableGradientDag const& dag, std::size_t root) -> std::optional<Tree>
{
    if (root == NoGrad) { return std::nullopt; }
    Operon::Vector<Node> sliced(dag.Nodes.begin(), dag.Nodes.begin() + static_cast<std::ptrdiff_t>(root) + 1);
    Tree t(std::move(sliced));
    t.UpdateNodes();
    return t;
}

auto VariableIndex(VariableGradientDag const& dag, Operon::Hash variable) -> std::optional<std::size_t>
{
    auto it = std::ranges::find(dag.Variables, variable);
    if (it == dag.Variables.end()) { return std::nullopt; }
    return static_cast<std::size_t>(std::distance(dag.Variables.begin(), it));
}

// The only point in this file that crosses into AffineEvaluator, which is
// pre-existing code that throws by design (domain violations, e.g.
// dividing by a zero-containing interval; ops with no registered affine
// rule) rather than returning an expected/empty value the way
// IntervalEvaluator does. Converting AffineEvaluator itself to
// tl::expected is a separate, larger change (deferred); this is the one
// place that adapts its exceptions to this file's own expected-based
// internals, so the rest of this file never needs a try/catch.
auto TryAffineBound(Tree const& tree, AffineEvaluator::DomainMap const& domains) -> BoundResult
{
    try {
        AffineEvaluator ae(&tree, domains);
        return ae.Evaluate(tree.GetCoefficients()).to_interval();
    } catch (std::exception const& e) {
        return tl::unexpected(std::string(e.what()));
    }
}

// The bound for one constraint's Op: the tree itself for Identity, or the
// (possibly twice-)differentiated tree for First-/SecondDerivative — an
// identically-zero derivative bounds to the degenerate interval [0, 0]
// rather than requiring a special case at every call site.
//
// VariableGradientDag::Certain[k] == false means the derivative dag hit an
// op with no rule on this variable's dependency path -- Roots[k] then isn't
// a trustworthy "derivative is zero" claim (see tree_diff.hpp). Reported as
// an error result here, same as any other can't-certify case.
auto BoundFor(ShapeConstraintOp op, Tree const& tree, Operon::Hash variable,
              AffineEvaluator::DomainMap const& domains) -> BoundResult
{
    if (op == ShapeConstraintOp::Identity) { return TryAffineBound(tree, domains); }

    auto const coeff = tree.GetCoefficients();
    auto dag1 = BuildVariableGradientDag(tree, coeff);
    auto const i1 = VariableIndex(dag1, variable);
    if (!i1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag1.Certain[*i1]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
    if (op == ShapeConstraintOp::FirstDerivative) {
        return d1 ? TryAffineBound(*d1, domains) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
    }

    // SecondDerivative: differentiate the materialized first-derivative
    // tree again, same variable both times — mixed partials aren't needed
    // by any constraint in this codebase's problem set.
    if (!d1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    auto const coeff1 = d1->GetCoefficients();
    auto dag2 = BuildVariableGradientDag(*d1, coeff1);
    auto const i2 = VariableIndex(dag2, variable);
    if (!i2) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag2.Certain[*i2]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d2 = SliceToTree(dag2, dag2.Roots[*i2]);
    return d2 ? TryAffineBound(*d2, domains) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
}

auto ResolveShapeConstraintContext(gsl::not_null<Operon::Problem const*> problem, ShapeConstraintSet const& constraints,
    Operon::Vector<Operon::Hash>& constraintVarHash,
    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>>& domainsByHash,
    std::string_view owner) -> void
{
    auto const* ds = problem->GetDataset();

    for (auto const& [name, bound] : constraints.Domains) {
        auto v = ds->GetVariable(name);
        if (!v) { throw std::invalid_argument(fmt::format("{}: domain references unknown variable '{}'", owner, name)); }
        domainsByHash.insert_or_assign(v->Hash, bound);
    }

    for (auto const& hash : problem->GetInputs()) {
        if (domainsByHash.contains(hash)) { continue; }
        auto v = ds->GetVariable(hash);
        throw std::invalid_argument(fmt::format(
            "{}: input variable '{}' has no entry in 'domains'", owner, v ? v->Name : fmt::format("<hash {}>", hash)));
    }

    constraintVarHash.reserve(constraints.Constraints.size());
    for (auto const& c : constraints.Constraints) {
        if (c.Sign.has_value() == c.Bound.has_value()) {
            throw std::invalid_argument(fmt::format("{}: constraint must set exactly one of Sign or Bound", owner));
        }
        if (c.Sign && *c.Sign != 1 && *c.Sign != -1) {
            throw std::invalid_argument(fmt::format("{}: constraint Sign {} must be 1 or -1", owner, *c.Sign));
        }
        if (c.Bound && c.Bound->first > c.Bound->second) {
            throw std::invalid_argument(fmt::format("{}: constraint Bound [{}, {}] has lo > hi", owner, c.Bound->first, c.Bound->second));
        }

        if (c.Op == ShapeConstraintOp::Identity) {
            constraintVarHash.push_back(Operon::Hash{});
            continue;
        }
        auto v = ds->GetVariable(c.Variable);
        if (!v) { throw std::invalid_argument(fmt::format("{}: constraint references unknown variable '{}'", owner, c.Variable)); }
        if (!domainsByHash.contains(v->Hash)) {
            throw std::invalid_argument(fmt::format("{}: constraint on '{}' has no matching entry in 'domains'", owner, c.Variable));
        }
        constraintVarHash.push_back(v->Hash);
    }
}

auto ConstraintViolation(ShapeConstraint const& c, Interval const& bound) -> Operon::Scalar
{
    if (c.Sign) {
        return (*c.Sign > 0) ? std::max(Operon::Scalar{0}, -bound.inf()) : std::max(Operon::Scalar{0}, bound.sup());
    }
    return std::max(Operon::Scalar{0}, c.Bound->first - bound.inf())
         + std::max(Operon::Scalar{0}, bound.sup() - c.Bound->second);
}

auto TransformBound(ShapeConstraintOp op, Interval const& bound, Operon::LinearScaling const& scaling) -> Interval
{
    auto const [lo, hi] = op == ShapeConstraintOp::Identity
        ? scaling.ApplyToValueInterval(bound.inf(), bound.sup())
        : scaling.ApplyToDerivativeInterval(bound.inf(), bound.sup());
    return Interval(lo, hi);
}

auto MeasureConstraints(ShapeConstraintSet const& constraints, Operon::Vector<Operon::Hash> const& constraintVarHash,
    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> const& domainsByHash,
    Operon::Tree const& tree, Operon::Scalar unknownViolation,
    std::optional<Operon::LinearScaling> scaling) -> ShapeConstraintMeasurementSummary
{
    ShapeConstraintMeasurementSummary summary;
    summary.Measurements.reserve(constraints.Constraints.size());
    for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
        auto const& c = constraints.Constraints[i];
        ShapeConstraintMeasurement m;
        auto const bound = BoundFor(c.Op, tree, constraintVarHash[i], domainsByHash);
        if (!bound) {
            m.Certified = false;
            m.Violation = unknownViolation;
        } else {
            auto const checkedBound = scaling ? TransformBound(c.Op, *bound, *scaling) : *bound;
            // A NaN endpoint (e.g. Scale == 0 times an unbounded raw-tree
            // interval, 0 * inf) must not reach ConstraintViolation:
            // std::max(0, NaN) returns 0 (NaN comparisons are always false),
            // which would silently certify an uncheckable tree as having zero
            // violation instead of flagging it as uncertified.
            if (!std::isfinite(checkedBound.inf()) || !std::isfinite(checkedBound.sup())) {
                m.Certified = false;
                m.Violation = unknownViolation;
            } else {
                m.Certified = true;
                m.Bound = std::pair{checkedBound.inf(), checkedBound.sup()};
                m.Violation = ConstraintViolation(c, checkedBound);
            }
        }
        if (!m.Certified || m.Violation != Operon::Scalar{0}) { summary.Feasible = false; }
        summary.Violation += m.Violation;
        summary.Measurements.push_back(m);
    }
    return summary;
}

} // namespace

ShapeConstrainedEvaluator::ShapeConstrainedEvaluator(gsl::not_null<EvaluatorBase const*> evaluator,
    gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSet constraints)
    : EvaluatorBase(evaluator->GetProblem())
    , evaluator_(evaluator)
    , dtable_(dtable)
    , constraints_(std::move(constraints))
{
    ResolveShapeConstraintContext(evaluator->GetProblem(), constraints_, constraintVarHash_, domainsByHash_, "ShapeConstrainedEvaluator");
}


auto ParseShapeEnforcement(std::string const& str) -> ShapeConstraintEnforcement
{
    auto result = ShapeConstraintEnforcement::None;
    std::size_t pos = 0;
    while (pos <= str.size()) {
        auto const next = str.find(',', pos);
        auto const token = str.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        if (token.empty()) { throw std::invalid_argument(fmt::format("unable to parse shape-enforcement argument '{}'", str)); }

        if (token == "hard-reject") {
            result = result | ShapeConstraintEnforcement::HardReject;
        } else if (token == "penalty") {
            result = result | ShapeConstraintEnforcement::Penalty;
        } else if (token == "extra-objective") {
            result = result | ShapeConstraintEnforcement::ExtraObjective;
        } else if (token == "feasibility-first") {
            result = result | ShapeConstraintEnforcement::FeasibilityFirst;
        } else {
            throw std::invalid_argument(fmt::format("unable to parse shape-enforcement argument '{}'", token));
        }

        if (next == std::string::npos) { break; }
        pos = next + 1;
    }
    return result;
}

auto ValidatePolicy(ShapeConstraintPolicy const& policy, bool isNsga2) -> std::optional<std::string>
{
    auto const modes = policy.Enforcement;
    auto const hard = HasFlag(modes, ShapeConstraintEnforcement::HardReject);
    auto const penalty = HasFlag(modes, ShapeConstraintEnforcement::Penalty);
    auto const extra = HasFlag(modes, ShapeConstraintEnforcement::ExtraObjective);
    auto const feasibilityFirst = HasFlag(modes, ShapeConstraintEnforcement::FeasibilityFirst);
    auto const raw = static_cast<unsigned>(modes);
    auto const known = static_cast<unsigned>(ShapeConstraintEnforcement::HardReject)
        | static_cast<unsigned>(ShapeConstraintEnforcement::Penalty)
        | static_cast<unsigned>(ShapeConstraintEnforcement::ExtraObjective)
        | static_cast<unsigned>(ShapeConstraintEnforcement::FeasibilityFirst);

    if ((raw & ~known) != 0U) { return "shape constraint policy contains unknown enforcement bits"; }
    if (modes == ShapeConstraintEnforcement::None) { return "shape constraint policy must select at least one enforcement mode"; }
    if (!std::isfinite(policy.UnknownViolation) || policy.UnknownViolation < Operon::Scalar{0}) { return "shape unknown violation must be finite and non-negative"; }
    if (!std::isfinite(policy.PenaltyWeight) || policy.PenaltyWeight < Operon::Scalar{0}) { return "shape penalty weight must be finite and non-negative"; }

    if (isNsga2) {
        if (feasibilityFirst) { return "shape constraint feasibility-first mode is not valid for NSGA2"; }
        if (hard && (penalty || extra)) { return "shape constraint hard-reject mode cannot be combined with penalty or extra-objective"; }
        return std::nullopt;
    }

    if (extra) { return "shape constraint extra-objective mode is only valid for NSGA2"; }
    if (hard && penalty) { return "shape constraint hard-reject mode cannot be combined with penalty"; }
    (void)feasibilityFirst;
    return std::nullopt;
}

auto ShapeConstrainedEvaluator::Measure(Operon::Tree const& tree, Operon::Scalar unknownViolation) const -> ShapeConstraintMeasurementSummary
{
    // Recompute instead of reusing a carried value: (a,b) is pure in tree/training data, and
    // non-Lamarckian local search may restore inherited coefficients after scoring optimized ones,
    // so scoring-path scaling could describe a different tree than the genotype certified here.
    auto const scaling = Operon::FitLinearScaling(tree, *GetProblem(), *dtable_, GetProblem()->TrainingRange());
    return MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, unknownViolation, scaling);
}

auto ShapeConstrainedEvaluator::Feasible(Operon::Tree const& tree) const -> bool
{
    auto const hash = Operon::detail::HashTreeForMemo(tree);
    ShapeConstraintMeasurementSummary result;
    // LazyEmplace holds this hash's shard lock across the miss branch, so
    // a concurrent caller hashing to the same key blocks on the first
    // computation rather than duplicating it.
    feasibleCache_.LazyEmplace(hash,
        [&](auto const& e) { result = e.Value; },
        [&](auto& e) {
            // Recompute instead of reusing a carried value: (a,b) is pure in tree/training data, and
            // non-Lamarckian local search may restore inherited coefficients after scoring optimized ones,
            // so scoring-path scaling could describe a different tree than the genotype certified here.
            auto const scaling = Operon::FitLinearScaling(tree, *GetProblem(), *dtable_, GetProblem()->TrainingRange());
            result = MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, Operon::Scalar{1}, scaling);
            e.Value = result;
        });
    return result.Feasible;
}

auto ShapeConstrainedEvaluator::Prepare(Operon::Span<Individual const> pop) const -> void
{
    evaluator_->Prepare(pop);
    feasibleCache_.Clear();
    for (auto const& ind : pop) {
        std::ignore = Feasible(ind.Genotype); // populates the cache as a side effect
    }
}

auto ShapeConstrainedEvaluator::Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType
{
    ++CallCount;
    if (!Feasible(ind.Genotype)) {
        ++violations_;
        return ReturnType(evaluator_->ObjectiveCount(), static_cast<Operon::Scalar>(worstValue_));
    }
    return (*evaluator_)(rng, ind, buf);
}

ShapeViolationEvaluator::ShapeViolationEvaluator(gsl::not_null<Operon::Problem const*> problem,
    gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSet constraints,
    Operon::Scalar weight, Operon::Scalar unknownViolation)
    : EvaluatorBase(problem)
    , problem_(problem)
    , dtable_(dtable)
    , constraints_(std::move(constraints))
    , weight_(weight)
    , unknownViolation_(unknownViolation)
{
    ResolveShapeConstraintContext(problem_, constraints_, constraintVarHash_, domainsByHash_, "ShapeViolationEvaluator");
}

auto ShapeViolationEvaluator::Measure(Operon::Tree const& tree) const -> ShapeConstraintMeasurementSummary
{
    auto const hash = Operon::detail::HashTreeForMemo(tree);
    ShapeConstraintMeasurementSummary result;
    // LazyEmplace holds this hash's shard lock across the miss branch, so
    // a concurrent caller hashing to the same key blocks on the first
    // computation rather than duplicating it.
    measurementCache_.LazyEmplace(hash,
        [&](auto const& e) { result = e.Value; },
        [&](auto& e) {
            // Recompute instead of reusing a carried value: (a,b) is pure in tree/training data, and
            // non-Lamarckian local search may restore inherited coefficients after scoring optimized ones,
            // so scoring-path scaling could describe a different tree than the genotype certified here.
            auto const scaling = Operon::FitLinearScaling(tree, *GetProblem(), *dtable_, GetProblem()->TrainingRange());
            result = MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, unknownViolation_, scaling);
            e.Value = result;
        });
    return result;
}

auto ShapeViolationEvaluator::Prepare(Operon::Span<Individual const> pop) const -> void
{
    measurementCache_.Clear();
    for (auto const& ind : pop) {
        std::ignore = Measure(ind.Genotype); // populates the cache as a side effect
    }
}

auto ShapeViolationEvaluator::RawViolation(Operon::Tree const& tree) const -> Operon::Scalar
{
    return Measure(tree).Violation;
}

auto ShapeViolationEvaluator::Evaluate(Operon::RandomGenerator& /*rng*/, Individual const& ind, Operon::Span<Operon::Scalar> /*buf*/) const -> typename EvaluatorBase::ReturnType
{
    ++CallCount;
    return ReturnType{static_cast<Operon::Scalar>(weight_ * RawViolation(ind.Genotype))};
}

} // namespace Operon
