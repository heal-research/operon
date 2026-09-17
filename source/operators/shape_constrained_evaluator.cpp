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
#include <vector>

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
        if (root == NoGrad) {
            return std::nullopt;
        }
        Operon::Vector<Node> sliced(dag.Nodes.begin(), dag.Nodes.begin() + static_cast<std::ptrdiff_t>(root) + 1);
        Tree t(std::move(sliced));
        t.UpdateNodes();
        return t;
    }

    auto VariableIndex(VariableGradientDag const& dag, Operon::Hash variable) -> std::optional<std::size_t>
    {
        auto it = std::ranges::find(dag.Variables, variable);
        if (it == dag.Variables.end()) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::distance(dag.Variables.begin(), it));
    }

    // True if `b` holds a finite interval -- used to detect NaN/inf enclosures from domain errors or
    // degenerate affine forms without a try/catch at each call site.

    auto IsFiniteBound(BoundResult const& b) -> bool
    {
        return b.has_value() && std::isfinite(b->inf()) && std::isfinite(b->sup());
    }

struct IntervalSubdivisionPlan {
    using DomainMap = IntervalEvaluator<Operon::Scalar>::DomainMap;

    Operon::Vector<Operon::Hash> Axes;
    Operon::Vector<std::size_t> Schedule;
    Operon::Vector<int> Splits;
    int Depth{};

    [[nodiscard]] auto SingleAxis() const noexcept -> bool { return Axes.size() == 1; }

    [[nodiscard]] static auto Make(Tree const& tree, DomainMap const& domains, int depth) -> std::optional<IntervalSubdivisionPlan>
    {
        if (depth <= 0 || depth > 20) { return std::nullopt; }

        IntervalSubdivisionPlan plan;
        Operon::Vector<Operon::Scalar> widths;
        for (auto const& node : tree.Nodes()) {
            if (!node.IsVariable() || std::ranges::find(plan.Axes, node.HashValue) != plan.Axes.end()) { continue; }
            auto const it = domains.find(node.HashValue);
            if (it == domains.end()) { continue; }
            auto const width = it->second.second - it->second.first;
            if (width <= Operon::Scalar { 0 }) { continue; }
            plan.Axes.push_back(node.HashValue);
            widths.push_back(width);
        }
        if (plan.Axes.empty()) { return std::nullopt; }

        constexpr int MaxScalarDepth = 12;
        plan.Depth = plan.SingleAxis() ? depth : std::min(depth, MaxScalarDepth);
        plan.Schedule.reserve(static_cast<std::size_t>(plan.Depth));
        plan.Splits.assign(plan.Axes.size(), 0);
        for (int level = 0; level < plan.Depth; ++level) {
            auto selected = std::size_t { 0 };
            for (std::size_t axis = 1; axis < plan.Axes.size(); ++axis) {
                if (widths[axis] > widths[selected]) { selected = axis; }
            }
            plan.Schedule.push_back(selected);
            ++plan.Splits[selected];
            widths[selected] /= Operon::Scalar { 2 };
        }
        return plan;
    }

    [[nodiscard]] auto LeafDomains(DomainMap const& domains, std::size_t leaf) const -> DomainMap
    {
        auto result = domains;
        Operon::Vector<int> cells(Axes.size());
        Operon::Vector<int> bits(Axes.size());
        for (std::size_t bit = 0; bit < Schedule.size(); ++bit) {
            auto const axis = Schedule[bit];
            cells[axis] |= (static_cast<int>((leaf >> bit) & std::size_t { 1 }) << bits[axis]++);
        }
        for (std::size_t axis = 0; axis < Axes.size(); ++axis) {
            auto const [lo, hi] = domains.at(Axes[axis]);
            auto const step = (hi - lo) / Operon::Scalar(std::size_t { 1 } << Splits[axis]);
            auto const cell = Operon::Scalar(cells[axis]);
            result[Axes[axis]] = {
                pappus::fp::ropd<pappus::fp::op_add>(lo, cell * step),
                pappus::fp::ropu<pappus::fp::op_add>(lo, (cell + Operon::Scalar { 1 }) * step)
            };
        }
        return result;
    }
};

    auto BisectedIntervalBound(Tree const& tree, IntervalEvaluator<Operon::Scalar>::DomainMap const& dom, int depth) -> BoundResult
    {
        using WScalar = eve::wide<Operon::Scalar>;
        constexpr int WSize = static_cast<int>(eve::cardinal_v<WScalar>);

        auto const directBound = [&]() -> BoundResult {
            try {
                IntervalEvaluator<Operon::Scalar> ie(&tree, dom);
                return ie.Evaluate(tree.GetCoefficients());
            } catch (std::exception const& e) {
                return tl::unexpected(std::string(e.what()));
            }
        };

        auto const plan = IntervalSubdivisionPlan::Make(tree, dom, depth);
        if (!plan) { return directBound(); }

        if (!plan->SingleAxis()) {
            try {
                using Pack = pappus::packed_subdomains<Operon::Scalar, WScalar>;
                auto const coeff = tree.GetCoefficients();
                auto const nLeaves = std::size_t { 1 } << plan->Depth;
                pappus::box<Operon::Scalar> domain;
                domain.reserve(plan->Axes.size());
                for (auto const axis : plan->Axes) {
                    auto const [lo, hi] = dom.at(axis);
                    domain.emplace_back(lo, hi);
                }
                pappus::subdivision_plan subdivision(std::move(domain), plan->Schedule);
                IntervalEvaluator<WScalar> evaluator(&tree, dom);
                auto acc = IntervalEvaluator<WScalar>::Interval::empty();
                std::size_t first = 0;
                Operon::Vector<typename IntervalEvaluator<WScalar>::LaneOverride> overrides(plan->Axes.size());
                for (; first + Pack::width <= nLeaves; first += Pack::width) {
                    Pack pack(subdivision, first);
                    for (std::size_t axis = 0; axis < plan->Axes.size(); ++axis) {
                        overrides[axis] = { plan->Axes[axis], pack.lower_data(axis), pack.upper_data(axis) };
                    }
                    auto const batch = evaluator.TryEvaluate(coeff, overrides);
                    if (!batch || eve::any(batch->is_empty())
                        || !eve::all(eve::is_finite(batch->inf()) && eve::is_finite(batch->sup()))) {
                        return directBound();
                    }
                    acc |= *batch;
                }
                std::optional<Interval> result;
                if (first != 0) {
                    result = Interval(eve::reduce(acc.inf(), eve::min), eve::reduce(acc.sup(), eve::max));
                }
                for (; first < nLeaves; ++first) {
                    IntervalEvaluator<Operon::Scalar> tail(&tree, plan->LeafDomains(dom, first));
                    auto const bound = tail.TryEvaluate(coeff);
                    if (!bound || bound->is_empty() || !std::isfinite(bound->inf()) || !std::isfinite(bound->sup())) {
                        return directBound();
                    }
                    result = result ? Interval(std::min(result->inf(), bound->inf()), std::max(result->sup(), bound->sup())) : *bound;
                }
                return result ? BoundResult { *result } : directBound();
            } catch (std::exception const&) {
                return directBound();
            }
        }

        try {
            auto const nLeaves = std::size_t { 1 } << plan->Depth;
            auto const widest = plan->Axes.front();
            auto const widestDiam = dom.at(widest).second - dom.at(widest).first;
            auto const lo = dom.at(widest).first;
            auto const hi = dom.at(widest).second;
            auto const h = widestDiam / Operon::Scalar(nLeaves);
            auto const coeff = tree.GetCoefficients();
            auto acc = IntervalEvaluator<WScalar>::Interval::empty();
            WScalar const hw(h);
            WScalar const infw(lo);
            WScalar const onew(Operon::Scalar { 1 });
            WScalar const lastw { Operon::Scalar(nLeaves) };
            // `eve::wide<T>` doesn't reliably keep its own alignment nested inside `std::pair`/hash-map storage
            // on this toolchain, so the bisected axis's per-lane bound is passed straight into TryEvaluate per
            // batch rather than boxed into `dom`.
            IntervalEvaluator<WScalar> wie(&tree, dom);
            std::size_t k = 0;
            for (; k + static_cast<std::size_t>(WSize) <= nLeaves; k += static_cast<std::size_t>(WSize)) {
                WScalar const idx = eve::iota(eve::as<WScalar>()) + WScalar(Operon::Scalar(k));
                // Matches Pappus's batch_evaluate_ia partition with explicit directed rounding; a poisoned lane
                // would otherwise be silently dropped by the union/reduction below, so any nonfinite batch
                // falls back to the sound direct bound instead.
                auto const lowerOffset = pappus::fp::ropd<pappus::fp::op_mul>(idx, hw);
                auto const upperOffset = pappus::fp::ropu<pappus::fp::op_mul>(idx + onew, hw);
                auto leafLo = eve::max(pappus::fp::ropd<pappus::fp::op_add>(infw, lowerOffset), infw);
                auto leafHi = pappus::fp::ropu<pappus::fp::op_add>(infw, upperOffset);
                leafHi = eve::if_else(idx + onew == lastw, eve::max(leafHi, WScalar(hi)), leafHi);
                auto const batch = wie.TryEvaluate(coeff, widest, leafLo, leafHi);
                if (!batch || !eve::all(eve::is_finite(batch->inf()) && eve::is_finite(batch->sup()))) {
                    return directBound();
                }
                acc |= *batch;
            }

            std::optional<Interval> result;
            if (k > 0) {
                result = Interval(eve::minimum(acc.inf()), eve::maximum(acc.sup()));
            }
            auto tailDom = dom;
            // Scalar tail for any leaves that didn't fill a full wide batch, using the same directed arithmetic and
            // endpoint clamping as the wide path above.
            for (; k < nLeaves; ++k) {
                auto const lowerOffset = pappus::fp::ropd<pappus::fp::op_mul>(Operon::Scalar(k), h);
                auto const upperOffset = pappus::fp::ropu<pappus::fp::op_mul>(Operon::Scalar(k + 1), h);
                auto leafLo = std::max(pappus::fp::ropd<pappus::fp::op_add>(lo, lowerOffset), lo);
                auto leafHi = pappus::fp::ropu<pappus::fp::op_add>(lo, upperOffset);
                if (k + 1 == nLeaves) { leafHi = std::max(leafHi, hi); }
                tailDom[widest] = { leafLo, leafHi };
                IntervalEvaluator<Operon::Scalar> ie(&tree, tailDom);
                auto const seg = ie.Evaluate(coeff);
                if (seg.is_empty() || !std::isfinite(seg.inf()) || !std::isfinite(seg.sup())) {
                    return directBound();
                }
                result = result ? Interval(std::min(result->inf(), seg.inf()), std::max(result->sup(), seg.sup())) : seg;
            }
            if (!result || !std::isfinite(result->inf()) || !std::isfinite(result->sup())) {
                return directBound();
            }
            return *result;
        } catch (std::exception const&) {
            return directBound();
        }
    }

    // The affine+interval intersection path, unchanged from before -- extracted so TryAffineBound (below) can
    // retry it over bisected sub-boxes when it fails on the whole domain.
    auto TryAffineBoundDirect(Tree const& tree, AffineEvaluator<Operon::Scalar>& ae, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
    {
        // Affine forms cannot represent every interval enclosure. In particular, a zero-crossing denominator is
        // unbounded and a variable exponent may reject an otherwise valid constant integer power. Fall back to
        // the interval evaluator, which can conservatively represent those cases.
        auto const IntervalBound = [&]() -> BoundResult {
            try {
                IntervalEvaluator<Operon::Scalar> ie(&tree, IntervalEvaluator<Operon::Scalar>::DomainMap { ae.Domains() });
                return ie.Evaluate(tree.GetCoefficients());
            } catch (std::exception const& e) {
                return tl::unexpected(std::string(e.what()));
            }
        };

        if (HasFlag(mode, ShapeBoundMode::Interval)) {
            if (HasFlag(mode, ShapeBoundMode::Bisected)) {
                return BisectedIntervalBound(tree, IntervalEvaluator<Operon::Scalar>::DomainMap { ae.Domains() }, opts.BisectionDepth);
            }
            return IntervalBound();
        }

        try {
            ae.SetTree(&tree);
            auto affine = ae.Evaluate(tree.GetCoefficients());
            // Catastrophic cancellation can make this float32 enclosure unsound: an intermediate center orders
            // of magnitude larger than the result implies a rounding-error floor exceeding the tracked radius.
            // Treat as uncertified rather than trusting a possibly-wrong interval.
            constexpr auto eps = std::numeric_limits<Operon::Scalar>::epsilon();
            auto const impliedErrorFloor = ae.MaxAbsCenter() * eps;
            // A zero radius means every noise symbol cancelled (e.g. x - x): structurally sound, not an
            // underestimate, so only judge forms that track real variable uncertainty.
            auto const r = affine.radius();
            if (r > 0 && impliedErrorFloor > opts.AffineIllConditionedThreshold * r) {
                auto bound = IntervalBound();
                if (bound) {
                    return bound;
                }
                return tl::unexpected(fmt::format(
                    "ill-conditioned: intermediate magnitude implies rounding error {} exceeds result radius {}; interval fallback failed: {}",
                    impliedErrorFloor, affine.radius(), bound.error()));
            }
            auto const bound = affine.to_interval();
            if (!std::isfinite(bound.inf()) || !std::isfinite(bound.sup())) {
                return IntervalBound();
            }
            // Affine's linearization of nonlinear ops (each Mul needs its own error term for the cross-product
            // it can't represent exactly; likewise exp/log) can make it looser than plain interval arithmetic on
            // the same tree, even though affine is tighter in the common case. Empirically confirmed on
            // correlated coeff*x*coeff*y chains (operon-publications shape-constraints-reproduction). Both
            // bounds are sound enclosures of the same quantity, so their intersection is sound and at least as
            // tight as either alone -- take it whenever the interval fallback succeeds and doesn't contradict
            // affine (a non-overlapping result means one bound is unsound, not that the intersection is empty --
            // fall back to the affine bound alone).
            if (HasFlag(mode, ShapeBoundMode::Affine)) {
                return bound;
            }
            if (auto ibound = IntervalBound(); ibound) {
                auto const lo = std::max(bound.inf(), ibound->inf());
                auto const hi = std::min(bound.sup(), ibound->sup());
                if (lo <= hi) {
                    return Interval(lo, hi);
                }
            }
            return bound;
        } catch (std::exception const& e) {
            auto bound = IntervalBound();
            if (bound) {
                return bound;
            }
            return tl::unexpected(fmt::format("affine evaluation failed: {}; interval fallback failed: {}", e.what(), bound.error()));
        }
    }

    // Bounded-depth domain bisection, used only as a last resort when TryAffineBoundDirect fails on the whole
    // domain box (e.g. log(x) straddling zero, but a narrower sub-box doesn't). Picks the widest axis, splits it
    // at its midpoint, recurses on both halves, and takes the hull -- sound by construction, same reasoning as
    // pappus's own evaluate_bisected. Deliberately NOT TightenRange/TightenRangeBisected: that method failed a
    // soundness gate on this exact derivative-slice tree class (see project memory). Only fires on the
    // already-uncertified path, so the exponential blowup with depth only taxes cases that were already
    // failing outright. Opt-in (opts.AffineBisectionMaxDepth) until a problem sweep shows a net win.
    auto BisectedDomainBound(Tree const& tree, AffineEvaluator<Operon::Scalar>::DomainMap const& domains, int depth, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
    {
        // Each sub-box evaluator owns a separate noise counter. Combine sub-box results only as intervals; never
        // combine their affine forms directly.
        AffineEvaluator<Operon::Scalar> subAe(&tree, domains);
        auto direct = TryAffineBoundDirect(tree, subAe, mode, opts);
        if (depth <= 0 || IsFiniteBound(direct)) {
            return direct;
        }

        // `domains` is the evaluator's full domain map, not just the variables `tree` references -- restrict
        // the widest-axis pick to hashes the tree actually contains, so the depth budget isn't burned on unused axes.
        Operon::Hash widest {};
        Operon::Scalar widestDiam { -1 };
        bool any = false;
        for (auto const& n : tree.Nodes()) {
            if (!n.IsVariable()) {
                continue;
            }
            auto const it = domains.find(n.HashValue);
            if (it == domains.end()) {
                continue;
            }
            auto const diam = it->second.second - it->second.first;
            if (diam > widestDiam) {
                widestDiam = diam;
                widest = n.HashValue;
                any = true;
            }
        }
        if (!any || widestDiam <= Operon::Scalar { 0 }) {
            return direct;
        }

        auto loDomains = domains;
        auto hiDomains = domains;
        auto const [lo, hi] = domains.at(widest);
        auto const mid = lo + (hi - lo) / Operon::Scalar { 2 };
        loDomains[widest].second = mid;
        hiDomains[widest].first = mid;

        auto left = BisectedDomainBound(tree, loDomains, depth - 1, mode, opts);
        auto right = BisectedDomainBound(tree, hiDomains, depth - 1, mode, opts);
        if (!IsFiniteBound(left) || !IsFiniteBound(right)) {
            return direct;
        }

        return Interval(std::min(left->inf(), right->inf()), std::max(left->sup(), right->sup()));
    }

    // Opt-in (default off, opts.UseTightenRangeFallback), independent of bisection. A targeted probe (see
    // operon-publications shape-constraints-reproduction/TIGHTENRANGE_RESCUE_FINDING.md) measured its rescue
    // rate here near zero versus bisection's ~3%: TightenRange degrades to the already-failing naive bound on
    // exactly the pathological derivative-slice trees this rescue role invokes it on. Kept opt-in since it's
    // real, tested infrastructure that costs nothing when unset, but bisection is the effective mechanism here.
    auto TryAffineBound(Tree const& tree, AffineEvaluator<Operon::Scalar>& ae, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
    {
        auto direct = TryAffineBoundDirect(tree, ae, mode, opts);
        if (IsFiniteBound(direct)) {
            return direct;
        }

        if (opts.UseTightenRangeFallback) {
            // TightenRange runs IntervalEvaluator internally, which throws for an op hash with no registered
            // interval rule -- unlike every other path in this file, it isn't pre-adapted to BoundResult's
            // exception-free contract, so wrap it here rather than let it escape into the caller's worker-thread
            // evaluation loop.
            try {
                auto tr = TightenRange(tree, ae.Domains(), tree.GetCoefficients());
                if (std::isfinite(tr.inf()) && std::isfinite(tr.sup())) {
                    return tr;
                }
            } catch (std::exception const&) {
                // fall through to the bisection fallback (or the uncertified direct bound)
            }
        }

        if (opts.AffineBisectionMaxDepth <= 0) {
            return direct;
        }

        auto bisected = BisectedDomainBound(tree, ae.Domains(), opts.AffineBisectionMaxDepth, mode, opts);
        return IsFiniteBound(bisected) ? bisected : direct;
    }

    // Interval-only fast path: TryAffineBoundDirect's `HasFlag(mode, Interval)` branch never touches ae's affine
    // capabilities, only `ae.Domains()` -- building a full AffineEvaluator purely to discard the affine half was
    // measured at ~24% wasted work per Measure() call on a small tree. Takes the plain interval domain map
    // directly instead, and skips the (opt-in, rarely-triggered) BisectedDomainBound rescue: that mechanism
    // builds its own AffineEvaluator per sub-box specifically to rescue affine-mode failures, and would
    // reintroduce the exact cost this function avoids (BisectedIntervalBound already has its own fallback via
    // directBound()). TightenRange's fallback is kept -- it only ever needed the domain map too.
    auto TryIntervalBound(Tree const& tree, IntervalEvaluator<Operon::Scalar>::DomainMap const& dom, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
    {
        auto const IntervalBound = [&]() -> BoundResult {
            try {
                IntervalEvaluator<Operon::Scalar> ie(&tree, dom);
                return ie.Evaluate(tree.GetCoefficients());
            } catch (std::exception const& e) {
                return tl::unexpected(std::string(e.what()));
            }
        };

        auto direct = HasFlag(mode, ShapeBoundMode::Bisected)
            ? BisectedIntervalBound(tree, dom, opts.BisectionDepth)
            : IntervalBound();
        if (IsFiniteBound(direct)) {
            return direct;
        }

        if (opts.UseTightenRangeFallback) {
            try {
                auto tr = TightenRange(tree, dom, tree.GetCoefficients());
                if (std::isfinite(tr.inf()) && std::isfinite(tr.sup())) {
                    return tr;
                }
            } catch (std::exception const&) {
                // fall through to the uncertified direct bound
            }
        }
        return direct;
    }

    // Mirrors BoundFor exactly, but for TryIntervalBound's lighter domain map instead of AffineEvaluator&. See
    // BoundFor's comment for the derivative slicing rationale (identical here).
    auto BoundForInterval(ShapeConstraintOp op, Tree const& tree, Operon::Hash variable,
        IntervalEvaluator<Operon::Scalar>::DomainMap const& dom,
        VariableGradientDag const& dag1, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
    {
        if (op == ShapeConstraintOp::Identity) {
            return TryIntervalBound(tree, dom, mode, opts);
        }

        auto const i1 = VariableIndex(dag1, variable);
        if (!i1) {
            return BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
        }
        if (!dag1.Certain[*i1]) {
            return tl::unexpected("variable derivative involves an op with no differentiation rule");
        }
        auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
        if (op == ShapeConstraintOp::FirstDerivative) {
            return d1 ? TryIntervalBound(*d1, dom, mode, opts) : BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
        }

        if (!d1) {
            return BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
        }
        auto dag2 = BuildVariableGradientDag(*d1, d1->GetCoefficients());
        auto const i2 = VariableIndex(dag2, variable);
        if (!i2) {
            return BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
        }
        if (!dag2.Certain[*i2]) {
            return tl::unexpected("variable derivative involves an op with no differentiation rule");
        }
        auto d2 = SliceToTree(dag2, dag2.Roots[*i2]);
        return d2 ? TryIntervalBound(*d2, dom, mode, opts) : BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
    }

    // The bound for one constraint's Op: the tree itself for Identity, or the (possibly twice-)differentiated tree
    // for First-/SecondDerivative -- an identically-zero derivative bounds to the degenerate interval [0, 0].
    // `Certain[k] == false` means the derivative dag hit an op with no rule on this variable's path (see
    // tree_diff.hpp); reported as an error result here, same as any other can't-certify case.
    //
    // dag1 is the first-order gradient-dag of `tree`, built once per bound set by the caller and shared across
    // every derivative constraint in it (pure function of tree/coeff, independent of which variable). dag2
    // (SecondDerivative) is still built per-call from the sliced first-derivative tree `d1`, which IS
    // variable-specific.
    auto BoundFor(ShapeConstraintOp op, Tree const& tree, Operon::Hash variable,
        AffineEvaluator<Operon::Scalar>& ae,
        VariableGradientDag const& dag1, ShapeBoundMode mode, ShapeBoundOptions const& opts) -> BoundResult
    {
        if (op == ShapeConstraintOp::Identity) {
            return TryAffineBound(tree, ae, mode, opts);
        }

        auto const i1 = VariableIndex(dag1, variable);
        if (!i1) {
            return BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
        }
        if (!dag1.Certain[*i1]) {
            return tl::unexpected("variable derivative involves an op with no differentiation rule");
        }
        auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
        if (op == ShapeConstraintOp::FirstDerivative) {
            return d1 ? TryAffineBound(*d1, ae, mode, opts) : BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
        }

        // SecondDerivative: differentiate the materialized first-derivative
        // tree again, same variable both times — mixed partials aren't needed
        // by any constraint in this codebase's problem set.
        if (!d1) {
            return BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
        }
        auto dag2 = BuildVariableGradientDag(*d1, d1->GetCoefficients());
        auto const i2 = VariableIndex(dag2, variable);
        if (!i2) {
            return BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
        }
        if (!dag2.Certain[*i2]) {
            return tl::unexpected("variable derivative involves an op with no differentiation rule");
        }
        auto d2 = SliceToTree(dag2, dag2.Roots[*i2]);
        return d2 ? TryAffineBound(*d2, ae, mode, opts) : BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 }));
    }

    auto ResolveShapeConstraintContext(gsl::not_null<Operon::Problem const*> problem, ShapeConstraintSet const& constraints,
        Operon::Vector<Operon::Hash>& constraintVarHash,
        Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>>& domainsByHash,
        std::string_view owner) -> void
    {
        auto const* ds = problem->GetDataset();

        for (auto const& [name, bound] : constraints.Domains) {
            auto v = ds->GetVariable(name);
            if (!v) {
                throw std::invalid_argument(fmt::format("{}: domain references unknown variable '{}'", owner, name));
            }
            domainsByHash.insert_or_assign(v->Hash, bound);
        }

        for (auto const& hash : problem->GetInputs()) {
            if (domainsByHash.contains(hash)) {
                continue;
            }
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
                constraintVarHash.push_back(Operon::Hash {});
                continue;
            }
            auto v = ds->GetVariable(c.Variable);
            if (!v) {
                throw std::invalid_argument(fmt::format("{}: constraint references unknown variable '{}'", owner, c.Variable));
            }
            if (!domainsByHash.contains(v->Hash)) {
                throw std::invalid_argument(fmt::format("{}: constraint on '{}' has no matching entry in 'domains'", owner, c.Variable));
            }
            constraintVarHash.push_back(v->Hash);
        }
    }

    auto ConstraintViolation(ShapeConstraint const& c, Interval const& bound) -> Operon::Scalar
    {
        if (c.Sign) {
            return (*c.Sign > 0) ? std::max(Operon::Scalar { 0 }, -bound.inf()) : std::max(Operon::Scalar { 0 }, bound.sup());
        }
        return std::max(Operon::Scalar { 0 }, c.Bound->first - bound.inf())
            + std::max(Operon::Scalar { 0 }, bound.sup() - c.Bound->second);
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

        // Built on first use by BoundFor/BoundForInterval; shared across every derivative constraint in this
        // bound set (see BoundFor's comment). Identity constraints never touch it, so it is lazily constructed
        // only when a bound set actually contains a derivative constraint.
        std::optional<VariableGradientDag> dag1;
        auto const SharedDag1 = [&]() -> VariableGradientDag const& {
            if (!dag1) {
                dag1.emplace(BuildVariableGradientDag(tree, tree.GetCoefficients()));
            }
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
                // A NaN endpoint (e.g. Scale == 0 times an unbounded interval) must not reach ConstraintViolation:
                // std::max(0, NaN) returns 0, which would silently certify an uncheckable tree instead of flagging it.
                if (!std::isfinite(checkedBound.inf()) || !std::isfinite(checkedBound.sup())) {
                    m.Certified = false;
                    m.Violation = unknownViolation;
                } else {
                    m.Certified = true;
                    m.Bound = std::pair { checkedBound.inf(), checkedBound.sup() };
                    m.Violation = ConstraintViolation(c, checkedBound);
                }
            }
            if (!m.Certified || m.Violation != Operon::Scalar { 0 }) {
                summary.Feasible = false;
            }
            summary.Violation += m.Violation;
            summary.Measurements.push_back(m);
        };

        // Interval-only mode (with or without Bisected) never touches AffineEvaluator's affine machinery -- skip
        // constructing it, sharing the lighter interval domain map across every constraint in this set instead.
        if (HasFlag(mode, ShapeBoundMode::Interval)) {
            IntervalEvaluator<Operon::Scalar>::DomainMap const dom { domainsByHash };
            for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
                auto const& c = constraints.Constraints[i];
                auto const bound = c.Op == ShapeConstraintOp::Identity
                    ? TryIntervalBound(tree, dom, mode, opts)
                    : BoundForInterval(c.Op, tree, constraintVarHash[i], dom, SharedDag1(), mode, opts);
                Apply(i, bound);
            }
            return summary;
        }

        // One AffineEvaluator shared across every bound in this set, skipping a per-constraint DomainMap copy
        // and primal_ regrowth. SetTree() retargets it at each constraint's slice; ctx_'s monotonic noise-symbol
        // counter stays sound since the bounds are consumed as intervals independently of each other.
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

    // Runs `f(i)` for i in [0,pop.size()) on `executor` when one was set (the caller's own, already-sized
    // executor -- see SetExecutor), else sequentially. Uses `executor->corun(...)`, not `run(...).get()`: the
    // only caller, Prepare(), is itself already running as a task on that same executor, so `run().get()` would
    // risk a worker blocking on a taskflow that needs a free worker to progress; `corun()` joins the calling
    // thread in as a worker instead, avoiding that deadlock. A private per-instance Executor was tried and
    // measured to not help (~3x higher CPU, no wall-clock change) while doubling the thread count.
    template <typename F>
    auto ParallelForPopulation(tf::Executor* executor, Operon::Span<Operon::Individual const> pop, F&& f) -> void
    {
        auto const n = pop.size();
        if (n == 0) {
            return;
        }
        if (executor == nullptr || n == 1) {
            for (std::size_t i = 0; i != n; ++i) {
                f(i);
            }
            return;
        }
        tf::Taskflow taskflow;
        taskflow.for_each_index(std::size_t { 0 }, n, std::size_t { 1 }, [&](std::size_t i) { f(i); });
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
        if (token.empty()) {
            throw std::invalid_argument(fmt::format("unable to parse shape-enforcement argument '{}'", str));
        }

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

        if (next == std::string::npos) {
            break;
        }
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
    if ((raw & ~known) != 0U) {
        return "shape-bound-mode contains unknown bits";
    }
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
        if (next == std::string::npos) {
            break;
        }
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

    if ((raw & ~known) != 0U) {
        return "shape constraint policy contains unknown enforcement bits";
    }
    if (modes == ShapeConstraintEnforcement::None) {
        return "shape constraint policy must select at least one enforcement mode";
    }
    if (!std::isfinite(policy.UnknownViolation) || policy.UnknownViolation < Operon::Scalar { 0 }) {
        return "shape unknown violation must be finite and non-negative";
    }
    if (!std::isfinite(policy.PenaltyWeight) || policy.PenaltyWeight < Operon::Scalar { 0 }) {
        return "shape penalty weight must be finite and non-negative";
    }

    if (isNsga2) {
        if (feasibilityFirst) {
            return "shape constraint feasibility-first mode is not valid for NSGA2";
        }
        if (hard && (penalty || extra)) {
            return "shape constraint hard-reject mode cannot be combined with penalty or extra-objective";
        }
        return std::nullopt;
    }

    if (extra) {
        return "shape constraint extra-objective mode is only valid for NSGA2";
    }
    if (hard && penalty) {
        return "shape constraint hard-reject mode cannot be combined with penalty";
    }
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
    feasibleCache_.LazyEmplace(hash, [&](auto const& e) { result = e.Value; }, [&](auto& e) {
            // Recompute instead of reusing a carried value: (a,b) is pure in tree/training data, and
            // non-Lamarckian local search may restore inherited coefficients after scoring optimized ones,
            // so scoring-path scaling could describe a different tree than the genotype certified here.
            auto const scaling = Operon::FitLinearScaling(tree, *GetProblem(), *dtable_, GetProblem()->TrainingRange());
            result = MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, Operon::Scalar{1}, scaling, boundMode_, boundOptions_);
            e.Value = result; });
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
    if (auto err = ValidateShapeBoundMode(mode)) {
        throw std::invalid_argument(*err);
    }
    boundMode_ = mode;
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
    measurementCache_.LazyEmplace(hash, [&](auto const& e) { result = e.Value; }, [&](auto& e) {
            // Recompute instead of reusing a carried value: (a,b) is pure in tree/training data, and
            // non-Lamarckian local search may restore inherited coefficients after scoring optimized ones,
            // so scoring-path scaling could describe a different tree than the genotype certified here.
            auto const scaling = Operon::FitLinearScaling(tree, *GetProblem(), *dtable_, GetProblem()->TrainingRange());
            result = MeasureConstraints(constraints_, constraintVarHash_, domainsByHash_, tree, unknownViolation_, scaling, boundMode_, boundOptions_);
            e.Value = result; });
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
    return ReturnType { static_cast<Operon::Scalar>(weight_ * RawViolation(ind.Genotype)) };
}

auto ShapeViolationEvaluator::SetBoundMode(ShapeBoundMode mode) -> void
{
    if (auto err = ValidateShapeBoundMode(mode)) {
        throw std::invalid_argument(*err);
    }
    boundMode_ = mode;
}

} // namespace Operon
