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

namespace Operon {

namespace {

constexpr std::size_t NoGrad = std::numeric_limits<std::size_t>::max();

using Interval = AffineEvaluator<Operon::Scalar>::Interval;
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

// The only point in this file that crosses into AffineEvaluator. Domain
// violations return an `invalid()` NaN-poisoned form; the finiteness check
// below catches those. This try/catch adapts rare structural throws (e.g.
// forms from different affine_context instances) to these expected-based
// internals, so the rest of this file never needs a try/catch.

auto IsFiniteBound(BoundResult const& b) -> bool
{
    return b.has_value() && std::isfinite(b->inf()) && std::isfinite(b->sup());
}

// Upper bound for both bisection-depth knobs, enforced by
// ValidateShapeBoundOptions: the interval bisection's packed leaf
// endpoints are computed as lo + idx * fl(diam/2^depth) in
// Operon::Scalar, which stays exactly representable only while every leaf
// index up to 2^depth fits the scalar's mantissa -- 24 bits in the
// tightest supported precision (float). The affine knob's midpoint split
// is exact at any depth, but its sub-box count grows as 2^depth the same
// way, so one cap governs both.
constexpr int MaxBisectionDepth = 24;

// Shared by both SetBoundOptions setters (ShapeConstrainedEvaluator and
// ShapeViolationEvaluator), mirroring ValidateShapeBoundMode's role for
// SetBoundMode: a programmatically-constructed options struct is held to
// the same contract as the documented defaults instead of failing later
// deep inside the bound machinery.
auto ValidateShapeBoundOptions(ShapeBoundOptions const& opts) -> std::optional<std::string>
{
    if (opts.BisectionDepth < 0 || opts.BisectionDepth > MaxBisectionDepth) {
        return fmt::format("shape-bound-options: BisectionDepth must be in [0, {}]", MaxBisectionDepth);
    }
    if (opts.AffineBisectionMaxDepth < 0 || opts.AffineBisectionMaxDepth > MaxBisectionDepth) {
        return fmt::format("shape-bound-options: AffineBisectionMaxDepth must be in [0, {}]", MaxBisectionDepth);
    }
    return std::nullopt;
}

// The wide registry intentionally contains built-ins only. Check every
// structural failure that TryEvaluate can report before creating a wide
// object, so a composed or scalar-only user rule takes the direct path.
auto SupportsWideIntervalEvaluation(
    Tree const& tree, IntervalEvaluator<Operon::Scalar>::DomainMap const& dom
) -> bool
{
    using WScalar = eve::wide<Operon::Scalar>;
    RegisterIntervalBuiltins<WScalar>();

    if (tree.Nodes().empty()) { return false; }
    for (auto const& node : tree.Nodes()) {
        if (node.Type == NodeType::Variable && !dom.contains(node.HashValue)) { return false; }
        if (node.Type != NodeType::Function) { continue; }

        switch (node.HashValue) {
        case Operon::Hash(BuiltinOp::Add):
        case Operon::Hash(BuiltinOp::Mul):
        case Operon::Hash(BuiltinOp::Sub):
        case Operon::Hash(BuiltinOp::Div):
        case Operon::Hash(BuiltinOp::Fmin):
        case Operon::Hash(BuiltinOp::Fmax):
            continue;
        case Operon::Hash(BuiltinOp::Abs):
        case Operon::Hash(BuiltinOp::Acos):
        case Operon::Hash(BuiltinOp::Asin):
        case Operon::Hash(BuiltinOp::Atan):
        case Operon::Hash(BuiltinOp::Cbrt):
        case Operon::Hash(BuiltinOp::Ceil):
        case Operon::Hash(BuiltinOp::Cos):
        case Operon::Hash(BuiltinOp::Cosh):
        case Operon::Hash(BuiltinOp::Exp):
        case Operon::Hash(BuiltinOp::Floor):
        case Operon::Hash(BuiltinOp::Log):
        case Operon::Hash(BuiltinOp::Logabs):
        case Operon::Hash(BuiltinOp::Log1p):
        case Operon::Hash(BuiltinOp::Sin):
        case Operon::Hash(BuiltinOp::Sinh):
        case Operon::Hash(BuiltinOp::Sqrt):
        case Operon::Hash(BuiltinOp::Sqrtabs):
        case Operon::Hash(BuiltinOp::Tan):
        case Operon::Hash(BuiltinOp::Tanh):
        case Operon::Hash(BuiltinOp::Square):
            if (node.Arity == 1) { continue; }
            return false;
        case Operon::Hash(BuiltinOp::Aq):
        case Operon::Hash(BuiltinOp::Pow):
        case Operon::Hash(BuiltinOp::Powabs):
            if (node.Arity == 2) { continue; }
            return false;
        default:
            return false;
        }
    }
    return true;
}

// Interval-only domain bisection, SIMD-batched: picks the tree's single
// widest referenced axis, splits it into 2^depth uniform sub-intervals, and
// evaluates them eve::cardinal_v<wide<Operon::Scalar>> at a time through
// IntervalEvaluator<wide<Operon::Scalar>>::TryEvaluate() call per batch
// instead of one scalar evaluation per leaf. Mirrors pappus's
// batch_evaluate_ia: affine-in-i wide index arithmetic for the packed
// sub-interval endpoints, lane-wise union via interval<wide<T>>::operator|=,
// single horizontal reduce (eve::minimum/eve::maximum) at the end, scalar
// tail for any remainder below a full lane width.
//
// The packed sub-interval endpoints form an ENCLOSING partition: each is
// computed with Pappus's directed-rounding arithmetic, the first leaf's lower
// endpoint is clamped back up to the box's own inf (it involves no rounding),
// and the last leaf's upper endpoint is clamped up to the box's own sup.
// Round-to-nearest evaluation of lo + k*fl(diam/2^depth) can land strictly
// below the sup at large magnitudes (fl(hi-lo) rounds down), which would leave
// the top sliver of the domain covered by no leaf at all. The directed lower
// and upper results overlap at every split point, so the leaves cover
// [inf, sup].
//
// Falls back to the whole-box direct bound if the axis can't be split, if
// the domain box itself isn't finite, if any evaluated lane or tail leaf is
// empty (NaN bounds -- an out-of-domain sub-box, e.g. sqrt of an entirely
// negative slice) or nonfinite (an unbounded slice, e.g. 1/x straddling
// the split axis), or if preflight finds a tree that the built-in-only wide
// registry cannot evaluate. In particular, user-defined and composed rules
// stay on the scalar/direct path. In the empty/nonfinite cases the union
// over the surviving leaves would NOT be a sound enclosure of the whole box:
// interval<wide<T>>::operator|= drops empty lanes and the horizontal
// eve::minimum/eve::maximum reduce drops NaNs, so a poisoned slice would
// otherwise silently narrow the reported bound instead of widening it.
// Deliberately noexcept: all ordinary evaluation failures are returned as
// nullopt, so an SEH unwind can never cross this wide-local frame.
auto TryWideBisectedIntervalBound(
    Tree const& tree, IntervalEvaluator<Operon::Scalar>::DomainMap const& dom,
    Operon::Hash widest, Operon::Scalar widestDiam, int depth
) noexcept -> std::optional<Interval>
{
    using WScalar = eve::wide<Operon::Scalar>;
    constexpr int WSize = static_cast<int>(eve::cardinal_v<WScalar>);
    auto const domain = dom.find(widest);
    if (domain == dom.end()) { return std::nullopt; }
    auto const [lo, hi] = domain->second;
    if (!std::isfinite(lo) || !std::isfinite(hi)) { return std::nullopt; }

    int const nLeaves = 1 << depth;
    Operon::Scalar const h = widestDiam / Operon::Scalar(nLeaves);
    auto const coeff = tree.GetCoefficients();
    auto acc = IntervalEvaluator<WScalar>::Interval::empty();
    WScalar const hw(h);
    WScalar const infw(lo);
    WScalar const onew(Operon::Scalar{1});
    WScalar const lastw{Operon::Scalar(nLeaves)};
    int k = 0;
    for (; k + WSize <= nLeaves; k += WSize) {
        // Fresh map each batch (not mutated in place) -- avoids relying on
        // in-place-assignment semantics for a SIMD-typed hash map value.
        IntervalEvaluator<WScalar>::DomainMap wdom;
        wdom.reserve(dom.size());
        for (auto const& [hash, bound] : dom) {
            if (hash == widest) { continue; }
            wdom.emplace(hash, IntervalEvaluator<WScalar>::Domain{ WScalar(bound.first), WScalar(bound.second) });
        }
        WScalar const idx = eve::iota(eve::as<WScalar>()) + WScalar(Operon::Scalar(k));
        // Match Pappus's batch_evaluate_ia partition, but explicitly direct
        // both multiplication and addition before clamping the terminal
        // endpoint to the original domain's sup.
        auto const lowerOffset = pappus::fp::ropd<pappus::fp::op_mul>(idx, hw);
        auto const upperOffset = pappus::fp::ropu<pappus::fp::op_mul>(idx + onew, hw);
        auto leafLo = eve::max(pappus::fp::ropd<pappus::fp::op_add>(infw, lowerOffset), infw);
        auto leafHi = pappus::fp::ropu<pappus::fp::op_add>(infw, upperOffset);
        leafHi = eve::if_else(idx + onew == lastw, eve::max(leafHi, WScalar(hi)), leafHi);
        wdom.emplace(widest, IntervalEvaluator<WScalar>::Domain{ leafLo, leafHi });
        IntervalEvaluator<WScalar> wie(&tree, wdom);
        auto const batch = wie.TryEvaluate(coeff);
        if (!batch || !eve::all(eve::is_finite(batch->inf()) && eve::is_finite(batch->sup()))) {
            return std::nullopt;
        }
        acc |= *batch;
    }

    std::optional<Interval> result;
    if (k > 0) { result = Interval(eve::minimum(acc.inf()), eve::maximum(acc.sup())); }

    // Scalar tail for any leaves that didn't fill a full wide batch, using
    // the same directed arithmetic as the wide path.
    auto tailDom = dom;
    for (; k < nLeaves; ++k) {
        auto const lowerOffset = pappus::fp::ropd<pappus::fp::op_mul>(Operon::Scalar(k), h);
        auto const upperOffset = pappus::fp::ropu<pappus::fp::op_mul>(Operon::Scalar(k + 1), h);
        auto leafLo = std::max(pappus::fp::ropd<pappus::fp::op_add>(lo, lowerOffset), lo);
        auto leafHi = pappus::fp::ropu<pappus::fp::op_add>(lo, upperOffset);
        if (k + 1 == nLeaves) { leafHi = std::max(leafHi, hi); }
        tailDom[widest] = { leafLo, leafHi };
        IntervalEvaluator<Operon::Scalar> ie(&tree, tailDom);
        auto const seg = ie.TryEvaluate(coeff);
        if (!seg || !std::isfinite(seg->inf()) || !std::isfinite(seg->sup())) { return std::nullopt; }
        result = result ? Interval(std::min(result->inf(), seg->inf()), std::max(result->sup(), seg->sup())) : *seg;
    }

    if (!result || !std::isfinite(result->inf()) || !std::isfinite(result->sup())) { return std::nullopt; }
    return result;
}

// Interval-only domain bisection, SIMD-batched: picks the tree's single
// widest referenced axis, splits it into 2^depth uniform sub-intervals, and
// evaluates them eve::cardinal_v<wide<Operon::Scalar>> at a time. Unsupported
// scalar-only operations fall back before entering TryWideBisectedIntervalBound.
auto BisectedIntervalBound(Tree const& tree, IntervalEvaluator<Operon::Scalar>::DomainMap const& dom, int depth) -> BoundResult
{
    auto const directBound = [&]() -> BoundResult {
        IntervalEvaluator<Operon::Scalar> ie(&tree, dom);
        return ie.TryEvaluate(tree.GetCoefficients());
    };

    if (depth <= 0 || !SupportsWideIntervalEvaluation(tree, dom)) { return directBound(); }

    Operon::Hash widest{};
    Operon::Scalar widestDiam{-1};
    bool any = false;
    for (auto const& n : tree.Nodes()) {
        if (!n.IsVariable()) { continue; }
        auto const it = dom.find(n.HashValue);
        if (it == dom.end()) { continue; }
        auto const diam = it->second.second - it->second.first;
        if (diam > widestDiam) { widestDiam = diam; widest = n.HashValue; any = true; }
    }
    if (!any || widestDiam <= Operon::Scalar{0}) { return directBound(); }

    auto const bound = TryWideBisectedIntervalBound(tree, dom, widest, widestDiam, depth);
    return bound ? BoundResult(*bound) : directBound();
}

// The affine+interval intersection path, unchanged from before -- extracted
// so TryAffineBound (below) can retry it over bisected sub-boxes when it
// fails on the whole domain.
auto TryAffineBoundDirect(Tree const& tree, AffineEvaluator<Operon::Scalar>& ae, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
{
    // Affine forms cannot represent every interval enclosure. In particular,
    // a zero-crossing denominator is unbounded and a variable exponent may
    // reject an otherwise valid constant integer power. Fall back to the
    // interval evaluator, which can conservatively represent those cases.
    auto const IntervalBound = [&]() -> BoundResult {
        IntervalEvaluator<Operon::Scalar> ie(&tree, IntervalEvaluator<Operon::Scalar>::DomainMap{ae.Domains()});
        return ie.TryEvaluate(tree.GetCoefficients());
    };

    if (HasFlag(mode, ShapeBoundMode::Interval)) {
        if (HasFlag(mode, ShapeBoundMode::Bisected)) {
            return BisectedIntervalBound(tree, IntervalEvaluator<Operon::Scalar>::DomainMap{ae.Domains()}, opts.BisectionDepth);
        }
        return IntervalBound();
    }

    ae.SetTree(&tree);
    auto affine = ae.TryEvaluate(tree.GetCoefficients());
    if (!affine) {
        auto bound = IntervalBound();
        if (bound) { return bound; }
        return tl::unexpected(fmt::format(
            "affine evaluation failed: {}; interval fallback failed: {}",
            affine.error(), bound.error()));
    }
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
        auto const r = affine->radius();
        if (r > 0 && impliedErrorFloor > opts.AffineIllConditionedThreshold * r) {
            auto bound = IntervalBound();
            if (bound) { return bound; }
            return tl::unexpected(fmt::format(
                "ill-conditioned: intermediate magnitude implies rounding error {} exceeds result radius {}; interval fallback failed: {}",
                impliedErrorFloor, affine->radius(), bound.error()));
        }
        auto const bound = affine->to_interval();
        if (!std::isfinite(bound.inf()) || !std::isfinite(bound.sup())) {
            return IntervalBound();
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
        if (HasFlag(mode, ShapeBoundMode::Affine)) {
            return bound;
        }
        if (auto ibound = IntervalBound(); ibound) {
            auto const lo = std::max(bound.inf(), ibound->inf());
            auto const hi = std::min(bound.sup(), ibound->sup());
            if (lo <= hi) { return Interval(lo, hi); }
        }
        return bound;
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
// failing outright, not a per-call tax on the common case. Opt-in only
// (opts.AffineBisectionMaxDepth defaults to 0) until a problem sweep
// against the corrected logic shows a net win on a given problem.
auto BisectedDomainBound(Tree const& tree, AffineEvaluator<Operon::Scalar>::DomainMap const& domains, int depth, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
{
    // Each sub-box evaluator owns a separate noise counter. Combine sub-box
    // results only as intervals; never combine their affine forms directly.
    AffineEvaluator<Operon::Scalar> subAe(&tree, domains);
    auto direct = TryAffineBoundDirect(tree, subAe, mode, opts);
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

    auto left = BisectedDomainBound(tree, loDomains, depth - 1, mode, opts);
    auto right = BisectedDomainBound(tree, hiDomains, depth - 1, mode, opts);
    if (!IsFiniteBound(left) || !IsFiniteBound(right)) { return direct; }

    return Interval(std::min(left->inf(), right->inf()), std::max(left->sup(), right->sup()));
}

// Opt-in (default off, opts.UseTightenRangeFallback), independent of
// bisection. TightenRange's own soundness gate passes (2026-08-09 fix to
// the mean-value-form overflow bug), but a targeted 2026-09-12 probe (3
// problems x 5 reps, see operon-publications' shape-constraints-
// reproduction/TIGHTENRANGE_RESCUE_FINDING.md) measured its rescue rate
// here at 2 of 1,113,643 attempts (0.00018%), versus bisection's 26,455 of
// 839,666 (3.15%) on the same cells: TightenRange degrades to the
// already-failing naive bound on exactly the pathological derivative-slice
// trees this rescue role invokes it on. Kept (not removed) since it's
// real, tested infrastructure that costs nothing when unset, but do not
// expect it to help in this role -- bisection is the effective rescue
// mechanism here.
auto TryAffineBound(Tree const& tree, AffineEvaluator<Operon::Scalar>& ae, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
{
    auto direct = TryAffineBoundDirect(tree, ae, mode, opts);
    if (IsFiniteBound(direct)) { return direct; }

    if (opts.UseTightenRangeFallback) {
        // TightenRange runs IntervalEvaluator internally, which throws for an
        // op hash with no registered interval rule -- unlike every other path
        // in this file, it isn't pre-adapted to BoundResult's exception-free
        // contract, so wrap it here rather than let it escape into the
        // caller's worker-thread evaluation loop.
        try {
            auto tr = TightenRange(tree, ae.Domains(), tree.GetCoefficients());
            if (std::isfinite(tr.inf()) && std::isfinite(tr.sup())) { return tr; }
        } catch (std::exception const&) {
            // fall through to the bisection fallback (or the uncertified direct bound)
        }
    }

    if (opts.AffineBisectionMaxDepth <= 0) { return direct; }

    auto bisected = BisectedDomainBound(tree, ae.Domains(), opts.AffineBisectionMaxDepth, mode, opts);
    if (IsFiniteBound(bisected)) { return bisected; }
    return direct;
}

// Interval-only fast path: TryAffineBoundDirect's `HasFlag(mode, Interval)`
// branch never touches ae's affine capabilities (SetTree/Evaluate), only
// `ae.Domains()` -- so building a full AffineEvaluator (and copying its
// DomainMap) purely to discard the affine half was measured at ~70ns of
// fully wasted work per Measure() call (~24% of the whole call for a small
// tree). Takes the plain interval domain map directly instead.
//
// Drops the (opt-in, off-by-default, rarely-triggered) BisectedDomainBound
// rescue: that mechanism builds its own AffineEvaluator per sub-box and
// exists specifically to rescue affine-mode failures -- invoking it here
// would silently reintroduce the exact per-call AffineEvaluator cost this
// function exists to avoid, for a rescue that doesn't conceptually belong
// to interval-only mode anyway (BisectedIntervalBound already has its own
// interval-native fallback via directBound()). TightenRange's fallback is
// kept -- it only ever needed the domain map too.
auto TryIntervalBound(Tree const& tree, IntervalEvaluator<Operon::Scalar>::DomainMap const& dom, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
{
    auto const IntervalBound = [&]() -> BoundResult {
        IntervalEvaluator<Operon::Scalar> ie(&tree, dom);
        return ie.TryEvaluate(tree.GetCoefficients());
    };

    auto direct = HasFlag(mode, ShapeBoundMode::Bisected)
        ? BisectedIntervalBound(tree, dom, opts.BisectionDepth)
        : IntervalBound();
    if (IsFiniteBound(direct)) { return direct; }

    if (opts.UseTightenRangeFallback) {
        try {
            auto tr = TightenRange(tree, dom, tree.GetCoefficients());
            if (std::isfinite(tr.inf()) && std::isfinite(tr.sup())) { return tr; }
        } catch (std::exception const&) {
            // fall through to the uncertified direct bound
        }
    }
    return direct;
}

// Mirrors BoundFor exactly, but for TryIntervalBound's lighter domain map
// instead of AffineEvaluator&. See BoundFor's comment for the derivative
// slicing rationale (identical here).
auto BoundForInterval(ShapeConstraintOp op, Tree const& tree, Operon::Hash variable,
                       IntervalEvaluator<Operon::Scalar>::DomainMap const& dom,
                       VariableGradientDag const& dag1, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
{
    if (op == ShapeConstraintOp::Identity) { return TryIntervalBound(tree, dom, mode, opts); }

    auto const i1 = VariableIndex(dag1, variable);
    if (!i1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag1.Certain[*i1]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
    if (op == ShapeConstraintOp::FirstDerivative) {
        return d1 ? TryIntervalBound(*d1, dom, mode, opts) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
    }

    if (!d1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    auto dag2 = BuildVariableGradientDag(*d1, d1->GetCoefficients());
    auto const i2 = VariableIndex(dag2, variable);
    if (!i2) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag2.Certain[*i2]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d2 = SliceToTree(dag2, dag2.Roots[*i2]);
    return d2 ? TryIntervalBound(*d2, dom, mode, opts) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
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
              AffineEvaluator<Operon::Scalar>& ae,
              VariableGradientDag const& dag1, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
{
    if (op == ShapeConstraintOp::Identity) { return TryAffineBound(tree, ae, mode, opts); }

    auto const i1 = VariableIndex(dag1, variable);
    if (!i1) { return BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0})); }
    if (!dag1.Certain[*i1]) { return tl::unexpected("variable derivative involves an op with no differentiation rule"); }
    auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
    if (op == ShapeConstraintOp::FirstDerivative) {
        return d1 ? TryAffineBound(*d1, ae, mode, opts) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
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
    return d2 ? TryAffineBound(*d2, ae, mode, opts) : BoundResult(Interval(Operon::Scalar{0}, Operon::Scalar{0}));
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
    std::optional<Operon::LinearScaling> scaling, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> ShapeConstraintMeasurementSummary
{
    ShapeConstraintMeasurementSummary summary;
    summary.Measurements.reserve(constraints.Constraints.size());

    // Built on first use by BoundFor/BoundForInterval; shared across every
    // derivative constraint in this bound set (see BoundFor's comment).
    // Identity constraints never touch it, so it is lazily constructed only
    // when a bound set actually contains a derivative constraint.
    std::optional<VariableGradientDag> dag1;
    auto const SharedDag1 = [&]() -> VariableGradientDag const& {
        if (!dag1) { dag1.emplace(BuildVariableGradientDag(tree, tree.GetCoefficients())); }
        return *dag1;
    };

    // Applies one constraint's raw bound to `summary`, shared by both the
    // interval-only and affine/combined loops below.
    auto const Apply = [&](std::size_t i, BoundResult const& bound) {
        auto const& c = constraints.Constraints[i];
        ShapeConstraintMeasurement m;
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
    };

    // Interval-only mode (with or without Bisected) never touches
    // AffineEvaluator's actual affine machinery -- skip constructing it and
    // the DomainMap copy it costs (see TryIntervalBound's comment), sharing
    // the lighter interval domain map across every constraint in this set
    // instead (same amortization AffineEvaluator gave affine mode).
    if (HasFlag(mode, ShapeBoundMode::Interval)) {
        IntervalEvaluator<Operon::Scalar>::DomainMap const dom{domainsByHash};
        for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
            auto const& c = constraints.Constraints[i];
            auto const bound = c.Op == ShapeConstraintOp::Identity
                ? TryIntervalBound(tree, dom, mode, opts)
                : BoundForInterval(c.Op, tree, constraintVarHash[i], dom, SharedDag1(), mode, opts);
            Apply(i, bound);
        }
        return summary;
    }

    // One AffineEvaluator shared across every bound in this set: skip
    // re-copying the DomainMap and re-growing primal_ capacity for each
    // constraint (typical Friction config = identity + two first-derivative
    // constraints, so 3x savings on those costs per individual per cache
    // miss). SetTree() retargets it at each constraint's slice (the original
    // tree for identity, the sliced derivative trees for the derivatives);
    // ctx_ keeps a single monotonic noise-symbol counter, which is sound --
    // the bounds are consumed as intervals independently of each other.
    AffineEvaluator<Operon::Scalar> ae(&tree, domainsByHash);
    for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
        auto const& c = constraints.Constraints[i];
        auto const bound = c.Op == ShapeConstraintOp::Identity
            ? TryAffineBound(tree, ae, mode, opts)
            : BoundFor(c.Op, tree, constraintVarHash[i], ae, SharedDag1(), mode, opts);
        Apply(i, bound);
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

auto ValidateShapeBoundMode(ShapeBoundMode mode) -> std::optional<std::string>
{
    auto const raw = static_cast<unsigned>(mode);
    auto const known = static_cast<unsigned>(ShapeBoundMode::Interval)
        | static_cast<unsigned>(ShapeBoundMode::Affine)
        | static_cast<unsigned>(ShapeBoundMode::Bisected);
    if ((raw & ~known) != 0U) { return "shape-bound-mode contains unknown bits"; }
    if (HasFlag(mode, ShapeBoundMode::Interval) && HasFlag(mode, ShapeBoundMode::Affine)) {
        return "shape-bound-mode: interval and affine are mutually exclusive";
    }
    if (HasFlag(mode, ShapeBoundMode::Bisected) && !HasFlag(mode, ShapeBoundMode::Interval)) {
        return "shape-bound-mode: bisected is only supported combined with interval";
    }
    return std::nullopt;
}

auto ParseShapeBoundMode(std::string const& str) -> ShapeBoundMode
{
    auto result = ShapeBoundMode::Combined;
    std::size_t pos = 0;
    while (pos <= str.size()) {
        auto const next = str.find(',', pos);
        auto const token = str.substr(pos, next == std::string::npos ? std::string::npos : next - pos);
        if (token == "combined") {
            // no bits to set
        } else if (token == "interval") {
            result = result | ShapeBoundMode::Interval;
        } else if (token == "affine") {
            result = result | ShapeBoundMode::Affine;
        } else if (token == "bisected") {
            result = result | ShapeBoundMode::Bisected;
        } else {
            throw std::invalid_argument(fmt::format("unable to parse shape-bound-mode argument '{}'", token));
        }
        if (next == std::string::npos) { break; }
        pos = next + 1;
    }
    if (auto err = ValidateShapeBoundMode(result)) {
        throw std::invalid_argument(*err);
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
    return MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, unknownViolation, scaling, boundMode_, boundOptions_);
}

auto ShapeConstrainedEvaluator::Feasible(Operon::Tree const& tree) const -> bool
{
    auto const hash = Operon::detail::HashTreeForMemo(tree, static_cast<Operon::Hash>(boundMode_));
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
            result = MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, Operon::Scalar{1}, scaling, boundMode_, boundOptions_);
            e.Value = result;
        });
    return result.Feasible;
}

auto ShapeConstrainedEvaluator::Prepare(Operon::Span<Individual const> pop) const -> void
{
    evaluator_->Prepare(pop);
    feasibleCache_.Clear();
    ParallelForPopulation(taskExecutor_, pop, [&](std::size_t i) {
        std::ignore = Feasible(pop[i].Genotype); // populates the cache as a side effect
    });
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

auto ShapeConstrainedEvaluator::SetBoundMode(ShapeBoundMode mode) -> void
{
    if (auto err = ValidateShapeBoundMode(mode)) { throw std::invalid_argument(*err); }
    boundMode_ = mode;
}

auto ShapeConstrainedEvaluator::SetBoundOptions(ShapeBoundOptions options) -> void
{
    if (auto err = ValidateShapeBoundOptions(options)) { throw std::invalid_argument(*err); }
    boundOptions_ = options;
    // The memo key Feasible() hashes covers the bound mode but NOT the
    // options, so entries computed under the previous depths would keep
    // answering as if those depths were still set.
    feasibleCache_.Clear();
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
    auto const hash = Operon::detail::HashTreeForMemo(tree, static_cast<Operon::Hash>(boundMode_));
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
            result = MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, unknownViolation_, scaling, boundMode_, boundOptions_);
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

auto ShapeViolationEvaluator::SetBoundMode(ShapeBoundMode mode) -> void
{
    if (auto err = ValidateShapeBoundMode(mode)) { throw std::invalid_argument(*err); }
    boundMode_ = mode;
}

auto ShapeViolationEvaluator::SetBoundOptions(ShapeBoundOptions options) -> void
{
    if (auto err = ValidateShapeBoundOptions(options)) { throw std::invalid_argument(*err); }
    boundOptions_ = options;
    // See ShapeConstrainedEvaluator::SetBoundOptions -- Measure()'s memo key
    // covers the bound mode but not the options.
    measurementCache_.Clear();
}

} // namespace Operon
