// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/shape_constrained_evaluator.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <tuple>

#include <fmt/format.h>
#include <taskflow/algorithm/for_each.hpp>
#include <tl/expected.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/core/tree_hash.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/range_tightening.hpp"
#include "operon/operators/linear_scaling.hpp"

#include <fstream>
#include <mutex>

namespace Operon {

namespace {

constexpr std::size_t NoGrad = std::numeric_limits<std::size_t>::max();

using Interval = AffineEvaluator::Interval;
using BoundResult = tl::expected<Interval, std::string>;

// TEMPORARY diagnostic (not a permanent feature): OPERON_SHAPE_BOUND_STATS=1
// tallies, over the whole process lifetime, which of TryAffineBound's paths
// fired and -- on the intersection path -- which of affine/interval was the
// actually-binding (tighter) bound on each side. Answers "how often does
// affine actually help here vs. interval doing the real work", the question
// prompted by 2026-08-09's finding that HL (plain interval only) had a
// higher feasible fraction than pre-fix Operon (affine alone on its happy
// path) on this same problem.
struct BoundPathStats {
    std::atomic_size_t affineDirect{0};      // interval fallback failed or wasn't needed to check
    std::atomic_size_t intersectCount{0};    // both bounds available, intersected
    std::atomic_size_t intersectLoFromAffine{0};
    std::atomic_size_t intersectLoFromInterval{0};
    std::atomic_size_t intersectLoTied{0};
    std::atomic_size_t intersectHiFromAffine{0};
    std::atomic_size_t intersectHiFromInterval{0};
    std::atomic_size_t intersectHiTied{0};
    std::atomic_size_t nonOverlapping{0};    // intersection empty -- used affine alone
    std::atomic_size_t illConditionedFallback{0}; // affine distrusted outright -- interval alone
    std::atomic_size_t nonFiniteFallback{0};      // affine non-finite -- interval alone
    std::atomic_size_t exceptionFallback{0};      // affine threw -- interval alone
    std::atomic_size_t bisectionAttempted{0};     // direct path failed/non-finite, tried domain bisection
    std::atomic_size_t bisectionRescued{0};       // bisection found a finite sound bound the direct path missed
    std::atomic_size_t tightenRangeAttempted{0};  // direct path failed/non-finite, tried TightenRange
    std::atomic_size_t tightenRangeRescued{0};    // TightenRange found a finite sound bound the direct path missed
};

auto GlobalBoundPathStats() -> BoundPathStats&
{
    static BoundPathStats stats;
    return stats;
}

auto BoundPathStatsEnabled() -> bool
{
    static bool const enabled = std::getenv("OPERON_SHAPE_BOUND_STATS") != nullptr;
    if (enabled) {
        static std::once_flag registered;
        std::call_once(registered, [] {
            std::atexit([] {
                auto const& s = GlobalBoundPathStats();
                std::fprintf(stderr,
                    "[shape-bound-stats] affine-direct=%zu intersect=%zu (lo: affine=%zu interval=%zu tied=%zu; "
                    "hi: affine=%zu interval=%zu tied=%zu) non-overlapping=%zu ill-conditioned-fallback=%zu "
                    "non-finite-fallback=%zu exception-fallback=%zu bisection-attempted=%zu bisection-rescued=%zu "
                    "tightenrange-attempted=%zu tightenrange-rescued=%zu\n",
                    s.affineDirect.load(), s.intersectCount.load(),
                    s.intersectLoFromAffine.load(), s.intersectLoFromInterval.load(), s.intersectLoTied.load(),
                    s.intersectHiFromAffine.load(), s.intersectHiFromInterval.load(), s.intersectHiTied.load(),
                    s.nonOverlapping.load(), s.illConditionedFallback.load(),
                    s.nonFiniteFallback.load(), s.exceptionFallback.load(),
                    s.bisectionAttempted.load(), s.bisectionRescued.load(),
                    s.tightenRangeAttempted.load(), s.tightenRangeRescued.load());
            });
        });
    }
    return enabled;
}

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
// k: flag a bound as uncertified when the float32 rounding-error floor
// implied by the largest intermediate center exceeds k * the final radius.
// A pathological cancellation (Case D) reaches ~909 vs radius ~9 (~100x);
// k=4 leaves two orders of magnitude of headroom for benign large-then-small
// arithmetic while still catching a genuinely unsound enclosure.
constexpr Operon::Scalar IllConditionedThreshold = Operon::Scalar{4};

auto ShapeAffineIllConditionedThreshold() -> Operon::Scalar
{
    static auto const threshold = [] {
        auto const* env = std::getenv("OPERON_SHAPE_AFFINE_ILL_THRESHOLD");
        if (env == nullptr || *env == '\0') { return IllConditionedThreshold; }
        char* end = nullptr;
        auto const value = std::strtof(env, &end);
        if (end == env || *end != '\0' || !std::isfinite(value) || value < Operon::Scalar{0}) {
            return IllConditionedThreshold;
        }
        return value;
    }();
    return threshold;
}

// The affine+interval intersection path, unchanged from before -- extracted
// so TryAffineBound (below) can retry it over bisected sub-boxes when it
// fails on the whole domain.
auto TryAffineBoundDirect(Tree const& tree, AffineEvaluator& ae, ShapeBoundMode mode) -> BoundResult
{
    // TEMPORARY diagnostic (not a permanent feature): OPERON_SHAPE_DEBUG=1
    // traces which of the three paths (affine / ill-conditioned-fallback /
    // exception-fallback) produced the returned bound, for root-causing
    // disagreements against HL's plain-interval bound on the same tree.
    static bool const dbg = std::getenv("OPERON_SHAPE_DEBUG") != nullptr;
    bool const boundStats = BoundPathStatsEnabled();

    // Affine forms cannot represent every interval enclosure. In particular,
    // a zero-crossing denominator is unbounded and a variable exponent may
    // reject an otherwise valid constant integer power. Fall back to the
    // interval evaluator, which can conservatively represent those cases.
    auto const IntervalBound = [&]() -> BoundResult {
        try {
            IntervalEvaluator ie(&tree, IntervalEvaluator::DomainMap{ae.Domains()});
            return ie.Evaluate(tree.GetCoefficients());
        } catch (std::exception const& e) {
            return tl::unexpected(std::string(e.what()));
        }
    };

    if (mode == ShapeBoundMode::IntervalOnly) { return IntervalBound(); }

    try {
        ae.SetTree(&tree);
        auto affine = ae.Evaluate(tree.GetCoefficients());
        // Catastrophic cancellation can make this float32 enclosure unsound:
        // an intermediate center orders of magnitude larger than the result
        // implies a rounding-error floor exceeding the tracked radius, so the
        // true value may fall outside the certified interval. Treat as
        // uncertified (same path as a pow domain error or a NaN bound) rather
        // than trusting a possibly-wrong interval.
        constexpr auto eps = std::numeric_limits<Operon::Scalar>::epsilon();
        auto const impliedErrorFloor = ae.MaxAbsCenter() * eps;
        // A zero radius means the form is an exact constant: every noise symbol
        // cancelled (e.g. a linear model's derivative, or x - x). That is
        // structurally sound, not an underestimate -- comparing floor > k*0 is
        // degenerate (any nonzero floor fires), so only judge forms that track
        // real variable uncertainty.
        auto const r = affine.radius();
        if (dbg) {
            auto const direct = affine.to_interval();
            std::fprintf(stderr, "[shape-trybound] affine direct=[%.9g, %.9g] center-based-floor=%.9g radius=%.9g threshold*radius=%.9g\n",
                direct.inf(), direct.sup(), impliedErrorFloor, r, ShapeAffineIllConditionedThreshold() * r);
        }
        if (r > 0 && impliedErrorFloor > ShapeAffineIllConditionedThreshold() * r) {
            auto bound = IntervalBound();
            if (boundStats) { ++GlobalBoundPathStats().illConditionedFallback; }
            if (dbg) {
                std::fprintf(stderr, "[shape-trybound] path=ill-conditioned-fallback interval=%s\n",
                    bound ? fmt::format("[{:.9g}, {:.9g}]", bound->inf(), bound->sup()).c_str() : bound.error().c_str());
            }
            if (bound) { return bound; }
            return tl::unexpected(fmt::format(
                "ill-conditioned: intermediate magnitude implies rounding error {} exceeds result radius {}; interval fallback failed: {}",
                impliedErrorFloor, affine.radius(), bound.error()));
        }
        auto const bound = affine.to_interval();
        if (!std::isfinite(bound.inf()) || !std::isfinite(bound.sup())) {
            auto ibound = IntervalBound();
            if (boundStats) { ++GlobalBoundPathStats().nonFiniteFallback; }
            if (dbg) {
                std::fprintf(stderr, "[shape-trybound] path=non-finite-affine-fallback interval=%s\n",
                    ibound ? fmt::format("[{:.9g}, {:.9g}]", ibound->inf(), ibound->sup()).c_str() : ibound.error().c_str());
            }
            return ibound;
        }
        // Affine's linearization of nonlinear ops (each Mul of two affine
        // forms needs its own error term for the cross-product it can't
        // represent exactly; likewise exp/log) can make it looser than
        // plain interval arithmetic on the same tree, even though affine is
        // tighter in the common case (a shared noise symbol lets repeated
        // occurrences of the same variable partially cancel). Confirmed
        // 2026-08-09 (operon-publications shape-constraints-reproduction):
        // chains of correlated coeff*x*coeff*y multiplications gave affine
        // [-5.15, 7.85] vs plain interval [0, 7.85] for the identical tree
        // and domain box -- affine sound but needlessly rejecting an
        // actually-feasible model. Both bounds are sound enclosures of the
        // same quantity, so their intersection is also sound and at least
        // as tight as either alone; take it whenever the interval fallback
        // itself succeeds and doesn't contradict affine (a non-overlapping
        // result would mean one of the two is unsound, not that the
        // intersection is empty -- fall back to the affine bound alone
        // rather than construct an inverted interval).
        if (mode == ShapeBoundMode::AffineOnly) {
            if (boundStats) { ++GlobalBoundPathStats().affineDirect; }
            return bound;
        }
        if (auto ibound = IntervalBound(); ibound) {
            auto const lo = std::max(bound.inf(), ibound->inf());
            auto const hi = std::min(bound.sup(), ibound->sup());
            if (dbg) {
                std::fprintf(stderr, "[shape-trybound] path=affine-interval-intersect affine=[%.9g, %.9g] interval=[%.9g, %.9g] intersect=%s\n",
                    bound.inf(), bound.sup(), ibound->inf(), ibound->sup(),
                    lo <= hi ? fmt::format("[{:.9g}, {:.9g}]", lo, hi).c_str() : "non-overlapping, using affine alone");
            }
            if (lo <= hi) {
                if (boundStats) {
                    auto& s = GlobalBoundPathStats();
                    ++s.intersectCount;
                    if (bound.inf() > ibound->inf()) { ++s.intersectLoFromAffine; }
                    else if (ibound->inf() > bound.inf()) { ++s.intersectLoFromInterval; }
                    else { ++s.intersectLoTied; }
                    if (bound.sup() < ibound->sup()) { ++s.intersectHiFromAffine; }
                    else if (ibound->sup() < bound.sup()) { ++s.intersectHiFromInterval; }
                    else { ++s.intersectHiTied; }
                }
                return Interval(lo, hi);
            }
            if (boundStats) { ++GlobalBoundPathStats().nonOverlapping; }
        } else if (dbg) {
            std::fprintf(stderr, "[shape-trybound] path=affine-direct (interval fallback failed: %s) bound=[%.9g, %.9g]\n",
                ibound.error().c_str(), bound.inf(), bound.sup());
        }
        if (boundStats) { ++GlobalBoundPathStats().affineDirect; }
        return bound;
    } catch (std::exception const& e) {
        auto bound = IntervalBound();
        if (boundStats) { ++GlobalBoundPathStats().exceptionFallback; }
        if (dbg) {
            std::fprintf(stderr, "[shape-trybound] path=exception-fallback what=%s interval=%s\n", e.what(),
                bound ? fmt::format("[{:.9g}, {:.9g}]", bound->inf(), bound->sup()).c_str() : bound.error().c_str());
        }
        if (bound) { return bound; }
        return tl::unexpected(fmt::format("affine evaluation failed: {}; interval fallback failed: {}", e.what(), bound.error()));
    }
}

auto IsFiniteBound(BoundResult const& b) -> bool
{
    return b.has_value() && std::isfinite(b->inf()) && std::isfinite(b->sup());
}

// Bounded-depth domain bisection, used only as a last resort when
// TryAffineBoundDirect fails on the whole domain box (e.g. log(x) where x's
// full range straddles zero, but a narrower sub-box's range doesn't). Picks
// the widest axis, splits it at its midpoint, recurses on both halves, and
// takes the hull of the two sub-results -- sound by construction (a union
// of sound sub-box enclosures is itself a sound enclosure of the whole
// box), same reasoning as pappus's own evaluate_bisected. Deliberately NOT
// operon's TightenRange/TightenRangeBisected (the mean-value/Newton-style
// method) -- that failed a soundness gate on this exact shape-constraint
// derivative-slice tree class (see project memory, 2026-08-06 finding), and
// this is a different, unrelated mechanism (no gradient/mean-value math
// involved) not affected by that bug.
//
// Only fires on the (relatively rare) already-uncertified path, so the
// exponential blowup with depth is bounded to cases that were already
// failing outright, not a per-call tax on the common case. An earlier
// 2026-08-09 sweep measured this as a net wall-clock LOSS on both problems
// tested, but that run picked the widest split axis from the evaluator's
// full domain map instead of the tree's own variables (fixed 2026-08-20)
// -- it was almost always bisecting an axis the tree doesn't even
// reference, so that result doesn't actually say anything about whether
// bisection helps and needs to be re-measured against the fixed axis
// selection. Opt-in only (default off) until a problem sweep against the
// corrected logic shows a net win -- see OPERON_SHAPE_BOUND_STATS's
// bisection-attempted vs bisection-rescued counters to judge the trade-off
// on a given problem.
auto BisectionMaxDepth() -> int
{
    static int const depth = [] {
        auto const* env = std::getenv("OPERON_SHAPE_BISECTION_DEPTH");
        return env ? std::atoi(env) : 0;
    }();
    return depth;
}

auto BisectedDomainBound(Tree const& tree, AffineEvaluator::DomainMap const& domains, int depth, ShapeBoundMode mode) -> BoundResult
{
    AffineEvaluator subAe(&tree, domains);
    auto direct = TryAffineBoundDirect(tree, subAe, mode);
    if (depth <= 0 || IsFiniteBound(direct)) { return direct; }

    // `domains` is the evaluator's full domain map (every problem input),
    // not just the variables `tree` actually references -- widening the
    // search to unused axes would burn the depth budget splitting a box
    // dimension that can never affect this tree's bound. Restrict the
    // widest-axis pick to hashes tree actually contains.
    Operon::Hash widest{};
    Operon::Scalar widestDiam{-1};
    bool any = false;
    for (auto const& n : tree.Nodes()) {
        if (!n.IsVariable()) { continue; }
        auto const it = domains.find(n.HashValue);
        if (it == domains.end()) { continue; }
        auto const diam = it->second.second - it->second.first;
        if (diam > widestDiam) { widestDiam = diam; widest = n.HashValue; any = true; }
    }
    if (!any || widestDiam <= Operon::Scalar{0}) { return direct; }

    auto loDomains = domains;
    auto hiDomains = domains;
    auto const [lo, hi] = domains.at(widest);
    auto const mid = lo + (hi - lo) / Operon::Scalar{2};
    loDomains[widest].second = mid;
    hiDomains[widest].first = mid;

    auto left = BisectedDomainBound(tree, loDomains, depth - 1, mode);
    auto right = BisectedDomainBound(tree, hiDomains, depth - 1, mode);
    if (!IsFiniteBound(left) || !IsFiniteBound(right)) { return direct; }

    return Interval(std::min(left->inf(), right->inf()), std::max(left->sup(), right->sup()));
}

// Opt-in (default off), independent of bisection so each can be A/B tested
// on its own. TightenRange's own soundness gate now passes (2026-08-09,
// see project memory) after fixing the mean-value-form overflow bug, so
// this is safe to try -- but its value as a *rescue* mechanism here is
// separate from and untested against bisection's, hence the independent
// toggle.
auto UseTightenRangeFallback() -> bool
{
    static bool const enabled = std::getenv("OPERON_SHAPE_USE_TIGHTENRANGE") != nullptr;
    return enabled;
}

auto TryAffineBound(Tree const& tree, AffineEvaluator& ae, ShapeBoundMode mode) -> BoundResult
{
    auto direct = TryAffineBoundDirect(tree, ae, mode);
    if (IsFiniteBound(direct)) { return direct; }

    bool const boundStats = BoundPathStatsEnabled();

    if (UseTightenRangeFallback()) {
        if (boundStats) { ++GlobalBoundPathStats().tightenRangeAttempted; }
        // TightenRange runs IntervalEvaluator internally, which throws for an
        // op hash with no registered interval rule -- unlike every other path
        // in this file, it isn't pre-adapted to BoundResult's exception-free
        // contract, so wrap it here rather than let it escape into the
        // caller's worker-thread evaluation loop.
        try {
            auto tr = TightenRange(tree, ae.Domains(), tree.GetCoefficients());
            if (std::isfinite(tr.inf()) && std::isfinite(tr.sup())) {
                if (boundStats) { ++GlobalBoundPathStats().tightenRangeRescued; }
                return tr;
            }
        } catch (std::exception const&) {
            // fall through to the bisection fallback (or the uncertified direct bound)
        }
    }

    auto const depth = BisectionMaxDepth();
    if (depth <= 0) { return direct; }

    if (boundStats) { ++GlobalBoundPathStats().bisectionAttempted; }
    auto bisected = BisectedDomainBound(tree, ae.Domains(), depth, mode);
    if (IsFiniteBound(bisected)) {
        if (boundStats) { ++GlobalBoundPathStats().bisectionRescued; }
        return bisected;
    }
    return direct;
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
//
// dag1 is the first-order gradient-dag of `tree` (df/d(variables)), built
// once per bound set by the caller and shared across every derivative
// constraint in it: it's a pure function of (tree, coeff) and does not
// depend on which variable a given constraint is on. Hoisting it out of
// here removes a previously-per-constraint rebuild that was pure redundant
// work whenever a bound set has more than one derivative constraint (the
// common case). SecondDerivative still builds its own dag2 from the sliced
// first-derivative tree `d1`, which IS variable-specific.
auto BoundFor(ShapeConstraintOp op, Tree const& tree, Operon::Hash variable,
              AffineEvaluator& ae,
              VariableGradientDag const& dag1, ShapeBoundMode mode) -> BoundResult
{
    if (op == ShapeConstraintOp::Identity) { return TryAffineBound(tree, ae, mode); }

    auto const i1 = VariableIndex(dag1, variable);
    if (!i1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag1.Certain[*i1]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
    if (op == ShapeConstraintOp::FirstDerivative) {
        return d1 ? TryAffineBound(*d1, ae, mode) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
    }

    // SecondDerivative: differentiate the materialized first-derivative
    // tree again, same variable both times — mixed partials aren't needed
    // by any constraint in this codebase's problem set.
    if (!d1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    auto dag2 = BuildVariableGradientDag(*d1, d1->GetCoefficients());
    auto const i2 = VariableIndex(dag2, variable);
    if (!i2) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag2.Certain[*i2]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d2 = SliceToTree(dag2, dag2.Roots[*i2]);
    return d2 ? TryAffineBound(*d2, ae, mode) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
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
    std::optional<Operon::LinearScaling> scaling, ShapeBoundMode mode) -> ShapeConstraintMeasurementSummary
{
    static bool const dbg = std::getenv("OPERON_SHAPE_DEBUG") != nullptr;
    if (dbg && scaling) {
        std::fprintf(stderr, "[shape] scaling Scale=%.9g Offset=%.9g\n", scaling->Scale, scaling->Offset);
    } else if (dbg) {
        std::fprintf(stderr, "[shape] no scaling (nullopt)\n");
    }
    ShapeConstraintMeasurementSummary summary;
    summary.Measurements.reserve(constraints.Constraints.size());
    // One AffineEvaluator shared across every bound in this set: skip
    // re-copying the DomainMap and re-growing primal_ capacity for each
    // constraint (typical Friction config = identity + two first-derivative
    // constraints, so 3x savings on those costs per individual per cache
    // miss). SetTree() retargets it at each constraint's slice (the original
    // tree for identity, the sliced derivative trees for the derivatives);
    // ctx_ keeps a single monotonic noise-symbol counter, which is sound --
    // the bounds are consumed as intervals independently of each other.
    AffineEvaluator ae(&tree, domainsByHash);
    // Built on first use by BoundFor; shared across every derivative
    // constraint in this bound set (see BoundFor's comment). Identity
    // constraints never touch it, so it is lazily constructed only when a
    // bound set actually contains a derivative constraint.
    std::optional<VariableGradientDag> dag1;
    auto const SharedDag1 = [&]() -> VariableGradientDag const& {
        if (!dag1) { dag1.emplace(BuildVariableGradientDag(tree, tree.GetCoefficients())); }
        return *dag1;
    };
    for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
        auto const& c = constraints.Constraints[i];
        ShapeConstraintMeasurement m;
        auto const bound = c.Op == ShapeConstraintOp::Identity
            ? TryAffineBound(tree, ae, mode)
            : BoundFor(c.Op, tree, constraintVarHash[i], ae, SharedDag1(), mode);
        if (dbg) {
            std::fprintf(stderr, "[shape] c[%zu] op=%d var_hash=%llu raw=", i, static_cast<int>(c.Op),
                         static_cast<unsigned long long>(constraintVarHash[i]));
            if (bound) std::fprintf(stderr, "[%.9g, %.9g]", bound->inf(), bound->sup());
            else std::fprintf(stderr, "<error: %s>", bound.error().c_str());
            std::fprintf(stderr, "\n");
        }
        if (!bound) {
            m.Certified = false;
            m.Violation = unknownViolation;
        } else {
            auto const checkedBound = scaling ? TransformBound(c.Op, *bound, *scaling) : *bound;
            if (dbg) {
                std::fprintf(stderr, "[shape] c[%zu] checked=[%.9g, %.9g] finite=%d\n", i,
                             checkedBound.inf(), checkedBound.sup(),
                             static_cast<int>(std::isfinite(checkedBound.inf()) && std::isfinite(checkedBound.sup())));
            }
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
        if (dbg) {
            std::fprintf(stderr, "[shape] c[%zu] certified=%d violation=%.9g\n", i,
                         static_cast<int>(m.Certified), static_cast<double>(m.Violation));
        }
        if (!m.Certified || m.Violation != Operon::Scalar{0}) { summary.Feasible = false; }
        summary.Violation += m.Violation;
        summary.Measurements.push_back(m);
    }
    return summary;
}

// Runs `f(i)` for i in [0,pop.size()) on `executor` when one was set (via
// SetExecutor -- the caller's own, already-sized-to-`--threads` executor,
// e.g. the one cli/source/operon_gp.cpp threads into both gp.Run() and
// Reporter::operator()), else sequentially. Uses executor->corun(...), not
// run(...).get(): the only caller is Prepare(), which is itself already
// running as a task on that same executor (a single non-parallel "prepare
// evaluator" task, see gp.cpp/nsga2.cpp), so run().get() would risk a
// worker blocking on a taskflow that needs a free worker to progress --
// corun() has the calling thread join in as a worker on the nested graph
// instead, avoiding that deadlock (same reasoning as Reporter's own
// executor.corun(tf) call). A private per-instance Executor was tried
// first and measured to not help (~3x higher CPU, no wall-clock change on
// a real 200-generation run) while needlessly doubling the machine's
// thread count on top of the caller's own executor -- reusing the
// caller's is both correct and matches this codebase's existing pattern.
// NULL is never passed for f: the only caller is Prepare(), whose wrapped
// cache Emplace already serializes same-hash concurrent callers, so the
// only shared mutable state here is the cache shards the body writes
// through.
template<typename F>
auto ParallelForPopulation(tf::Executor* executor, Operon::Span<Operon::Individual const> pop, F&& f) -> void
{
    auto const n = pop.size();
    if (n == 0) { return; }
    if (executor == nullptr || n == 1) {
        for (std::size_t i = 0; i != n; ++i) { f(i); }
        return;
    }
    tf::Taskflow taskflow;
    taskflow.for_each_index(std::size_t{0}, n, std::size_t{1}, [&](std::size_t i) { f(i); });
    executor->corun(taskflow);
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

auto ParseShapeBoundMode(std::string const& str) -> ShapeBoundMode
{
    if (str == "combined") { return ShapeBoundMode::Combined; }
    if (str == "interval-only") { return ShapeBoundMode::IntervalOnly; }
    if (str == "affine-only") { return ShapeBoundMode::AffineOnly; }
    throw std::invalid_argument(fmt::format("unable to parse shape-bound-mode argument '{}'", str));
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
    return MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, unknownViolation, scaling, boundMode_);
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
            result = MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, Operon::Scalar{1}, scaling, boundMode_);
            e.Value = result;
        });
    return result.Feasible;
}

auto ShapeConstrainedEvaluator::Prepare(Operon::Span<Individual const> pop) const -> void
{
    evaluator_->Prepare(pop);
    feasibleCache_.Clear();

    // TEMPORARY diagnostic (not a permanent feature): OPERON_SHAPE_CERT_STATS=1
    // tallies the same three buckets HL's ShapeDynamicsAnalyzer reports per
    // generation (feasible / certified-infeasible / uncertified), to compare
    // Operon's affine-bound certification against HL's plain-interval one
    // directly instead of from source reading alone.
    static bool const certStats = std::getenv("OPERON_SHAPE_CERT_STATS") != nullptr;
    if (!certStats) {
        ParallelForPopulation(taskExecutor_, pop, [&](std::size_t i) {
            std::ignore = Feasible(pop[i].Genotype); // populates the cache as a side effect
        });
        return;
    }

    std::atomic_size_t feasible{0};
    std::atomic_size_t certifiedInfeasible{0};
    std::atomic_size_t uncertified{0};

    // TEMPORARY diagnostic (not a permanent feature): OPERON_SHAPE_DUMP_PATH=<path>
    // additionally dumps this generation's whole population as JSONL (infix
    // expression + per-constraint bound/violation/certified), overwriting the
    // file every call so the process's final write is the last generation's
    // population -- feeds heuristiclab-headless-runner's `--mode evalexpr`
    // for a direct, same-tree bound cross-comparison against HL's own
    // IntervalArithBoundsEstimator.
    static char const* const dumpPath = std::getenv("OPERON_SHAPE_DUMP_PATH");
    std::unique_ptr<std::ofstream> dump;
    std::mutex dumpMutex;
    if (dumpPath != nullptr) { dump = std::make_unique<std::ofstream>(dumpPath, std::ios::trunc); }

    auto const EscapeJson = [](std::string const& s) {
        std::string out;
        out.reserve(s.size());
        for (char c : s) {
            if (c == '"' || c == '\\') { out.push_back('\\'); }
            out.push_back(c);
        }
        return out;
    };

    ParallelForPopulation(taskExecutor_, pop, [&](std::size_t i) {
        auto const summary = Measure(pop[i].Genotype);
        bool const anyUncertified = std::ranges::any_of(summary.Measurements, [](auto const& m) { return !m.Certified; });
        if (anyUncertified) { ++uncertified; }
        else if (summary.Violation != Operon::Scalar{0}) { ++certifiedInfeasible; }
        else { ++feasible; }
        std::ignore = Feasible(pop[i].Genotype); // populates the ordinary cache as a side effect

        if (dump) {
            auto const infix = Operon::InfixFormatter::Format(pop[i].Genotype, *GetProblem()->GetDataset(), std::numeric_limits<Operon::Scalar>::max_digits10);
            // Constraints are checked against the SCALED model (Scale*f+Offset),
            // not the raw subtree -- TransformBound applies this same fit to the
            // bound rather than baking it into the tree (see MeasureConstraints).
            // A cross-engine comparison must apply the same scaling on the other
            // side (e.g. wrap the dumped infix as "offset + scale * (...)")
            // before checking it against HL, which bakes scaling into the tree
            // itself -- comparing raw-subtree bounds between engines silently
            // compares two different mathematical objects.
            auto const scaling = Operon::FitLinearScaling(pop[i].Genotype, *GetProblem(), *dtable_, GetProblem()->TrainingRange());
            auto const scale = scaling ? scaling->Scale : Operon::Scalar{1};
            auto const offset = scaling ? scaling->Offset : Operon::Scalar{0};
            std::string line = fmt::format(R"({{"id":"ind{}","infix":"{}","scale":{:.9g},"offset":{:.9g},"feasible":{},"constraints":[)",
                i, EscapeJson(infix), scale, offset, summary.Feasible ? "true" : "false");
            for (std::size_t c = 0; c < summary.Measurements.size(); ++c) {
                auto const& m = summary.Measurements[c];
                if (c > 0) { line += ","; }
                if (m.Certified && m.Bound) {
                    line += fmt::format(R"({{"certified":true,"lo":{:.9g},"hi":{:.9g},"violation":{:.9g}}})",
                        m.Bound->first, m.Bound->second, static_cast<double>(m.Violation));
                } else {
                    line += R"({"certified":false})";
                }
            }
            line += "]}\n";
            std::lock_guard<std::mutex> lock(dumpMutex);
            (*dump) << line;
        }
    });
    if (dump) { dump->flush(); }

    fmt::print(stderr, "[shape-cert-stats] total={} feasible={} certified_infeasible={} uncertified={}\n",
        pop.size(), feasible.load(), certifiedInfeasible.load(), uncertified.load());
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
            result = MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, unknownViolation_, scaling, boundMode_);
            e.Value = result;
        });
    return result;
}

auto ShapeViolationEvaluator::Prepare(Operon::Span<Individual const> pop) const -> void
{
    measurementCache_.Clear();
    ParallelForPopulation(taskExecutor_, pop, [&](std::size_t i) {
        std::ignore = Measure(pop[i].Genotype); // populates the cache as a side effect
    });
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
