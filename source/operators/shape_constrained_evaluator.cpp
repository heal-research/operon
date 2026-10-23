// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/shape_constrained_evaluator.hpp"

#include <algorithm>
#include <charconv>
#include <cmath>
#include <cstring>
#include <iterator>
#include <limits>
#include <numbers>
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <tuple>
#include <type_traits>
#include <vector>

#include <fmt/format.h>
#include <tl/expected.hpp>

#include "operon/core/dataset.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/interval_jet_evaluator.hpp"
#include "operon/interpreter/interval_second_order_jet_evaluator.hpp"
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
        return SliceDagRoot(dag.Nodes, root);
    }

    auto VariableIndex(VariableGradientDag const& dag, Operon::Hash variable) -> std::optional<std::size_t>
    {
        auto it = std::ranges::find(dag.Variables, variable);
        if (it == dag.Variables.end()) {
            return std::nullopt;
        }
        return static_cast<std::size_t>(std::distance(dag.Variables.begin(), it));
    }

    auto EvaluateIntervalBound(Tree const& tree, IntervalEvaluator<Operon::Scalar>::DomainMap const& domains)
        -> BoundResult
    {
        IntervalEvaluator<Operon::Scalar> evaluator(&tree, domains);
        return evaluator(tree.GetCoefficients());
    }

    // How many times each variable is evaluated when the expression rooted at the last node is expanded (a Ref slot
    // counts as the subtree it points at, so a variable reached through shared structure counts once per use).
    // Nodes the root does not reach are not counted. `Axes` lists the variable hashes of interest; the result holds
    // one count per axis. Sharing can double a count at every level of a deep DAG, so counts saturate at 2^500
    // (still far beyond any difference that matters, and small enough that scores computed from them stay finite).
    inline auto ExpandedOccurrences(Operon::Span<Node const> nodes, Operon::Span<Operon::Hash const> axes)
        -> Operon::Vector<double>
    {
        constexpr double Saturation = 0x1.0p500;
        Operon::Vector<double> occurrences(axes.size(), 0.0);
        if (nodes.empty()) {
            return occurrences;
        }
        Operon::Vector<double> uses(nodes.size(), 0.0);
        uses.back() = 1.0;
        for (auto i = nodes.size(); i-- > 0;) {
            auto const count = uses[i];
            if (count == 0.0) {
                continue;
            }
            auto const& node = nodes[i];
            if (node.IsRef()) {
                uses[node.RefTo] = std::min(uses[node.RefTo] + count, Saturation);
            } else if (node.IsVariable()) {
                if (auto const it = std::ranges::find(axes, node.HashValue); it != axes.end()) {
                    auto& occurrence = occurrences[static_cast<std::size_t>(it - axes.begin())];
                    occurrence = std::min(occurrence + count, Saturation);
                }
            } else if (!node.IsLeaf()) {
                for (auto const child : Subtree<Node const> { nodes, i }.Indices()) {
                    uses[child] = std::min(uses[child] + count, Saturation);
                }
            }
        }
        return occurrences;
    }

    struct IntervalSubdivisionPlan {
        using DomainMap = IntervalEvaluator<Operon::Scalar>::DomainMap;

        Operon::Vector<Operon::Hash> Axes;
        Operon::Vector<std::size_t> Schedule;
        Operon::Vector<int> Splits;
        int Depth {};

        [[nodiscard]] auto SingleAxis() const noexcept -> bool { return Axes.size() == 1; }

        // Depth cap of the routes that evaluate one scalar evaluator per leaf (jets, affine, combined). The multi-axis
        // plans are capped at it as well; a single-axis plan may go deeper only for the SIMD interval identity route.
        static constexpr int MaxScalarDepth = 12;

        // The plan of the first `maxDepth` bisections of this one (the schedule is built greedily, so that is the plan
        // `Make` builds for depth `maxDepth`).
        [[nodiscard]] auto Capped(int maxDepth) const -> IntervalSubdivisionPlan
        {
            auto result = *this;
            if (maxDepth >= Depth) {
                return result;
            }
            result.Depth = maxDepth;
            result.Schedule.resize(static_cast<std::size_t>(maxDepth));
            result.Splits.assign(Axes.size(), 0);
            for (auto const axis : result.Schedule) {
                ++result.Splits[axis];
            }
            return result;
        }

        // The 2^depth leaves are split along the axes by a fixed schedule of `depth` bisections. A variable that
        // occurs once in the expression is enclosed exactly by interval arithmetic and gains nothing from being
        // split; a variable that occurs several times loses the correlation between its occurrences (the dependency
        // problem), which splitting recovers. Each level therefore bisects the axis with the highest
        //     width * (occurrences - 1)^2
        // (occurrences counted in the expanded expression), halving that axis' score once it is split. When no
        // variable repeats the score is the plain width. Equal scores go to the smaller variable hash, so the
        // schedule does not depend on how the tree is laid out. Any schedule gives a sound bound; only its
        // tightness differs.
        [[nodiscard]] static auto Make(Tree const& tree, DomainMap const& domains, int depth)
            -> std::optional<IntervalSubdivisionPlan>
        {
            if (depth <= 0 || depth > 20) {
                return std::nullopt;
            }

            Operon::Vector<Operon::Hash> candidates;
            Operon::Vector<Operon::Scalar> candidateWidths;
            for (auto const& node : tree.Nodes()) {
                if (!node.IsVariable() || std::ranges::find(candidates, node.HashValue) != candidates.end()) {
                    continue;
                }
                auto const it = domains.find(node.HashValue);
                if (it == domains.end()) {
                    continue;
                }
                auto const width = it->second.second - it->second.first;
                if (width <= Operon::Scalar { 0 }) {
                    continue;
                }
                candidates.push_back(node.HashValue);
                candidateWidths.push_back(width);
            }

            // A variable the expanded expression never evaluates cannot affect the bound: it takes no axis.
            auto const counts = ExpandedOccurrences(tree.Nodes(), candidates);
            IntervalSubdivisionPlan plan;
            // Scores are kept as logarithms: log(width) + 2 log(occurrences - 1), minus log 2 per halving. An axis
            // that does not repeat scores -inf whenever another axis does.
            Operon::Vector<double> occurrences;
            Operon::Vector<double> score;
            for (std::size_t k = 0; k < candidates.size(); ++k) {
                if (counts[k] > 0.0) {
                    plan.Axes.push_back(candidates[k]);
                    occurrences.push_back(counts[k]);
                    score.push_back(std::log(static_cast<double>(candidateWidths[k])));
                }
            }
            if (plan.Axes.empty()) {
                return std::nullopt;
            }
            if (std::ranges::any_of(occurrences, [](double n) { return n > 1.0; })) {
                for (std::size_t k = 0; k < score.size(); ++k) {
                    score[k] += 2.0 * std::log(std::max(occurrences[k] - 1.0, 0.0));
                }
            }

            plan.Depth = plan.SingleAxis() ? depth : std::min(depth, MaxScalarDepth);
            plan.Schedule.reserve(static_cast<std::size_t>(plan.Depth));
            plan.Splits.assign(plan.Axes.size(), 0);
            for (int level = 0; level < plan.Depth; ++level) {
                auto selected = std::size_t { 0 };
                for (std::size_t axis = 1; axis < plan.Axes.size(); ++axis) {
                    if (score[axis] > score[selected]
                        || (score[axis] == score[selected] && plan.Axes[axis] < plan.Axes[selected])) {
                        selected = axis;
                    }
                }
                plan.Schedule.push_back(selected);
                ++plan.Splits[selected];
                score[selected] -= std::numbers::ln2;
            }
            return plan;
        }

        // The cell index of `leaf` on every axis: bit b of `leaf` selects the lower/upper half of the axis the b-th
        // bisection of the schedule splits.
        [[nodiscard]] auto Cells(std::size_t leaf) const -> Operon::Vector<int>
        {
            Operon::Vector<int> cells(Axes.size());
            Operon::Vector<int> bits(Axes.size());
            for (std::size_t bit = 0; bit < Schedule.size(); ++bit) {
                auto const axis = Schedule[bit];
                cells[axis] |= (static_cast<int>((leaf >> bit) & std::size_t { 1 }) << bits[axis]++);
            }
            return cells;
        }

        // Whether `cell` is the last cell of axis `axis`.
        [[nodiscard]] auto IsLastCell(std::size_t axis, int cell) const -> bool
        {
            return cell + 1 == (1 << Splits[axis]);
        }

        // The box of leaf `leaf`. The outermost endpoints of every axis are clamped to the domain itself, so that the
        // union of all leaf boxes provably covers the whole box despite the rounding of (hi - lo) / 2^splits (the step
        // is rounded to nearest, and underflows to zero for a subnormal width, so the top of the last cell can
        // otherwise fall short of hi).
        [[nodiscard]] auto LeafDomains(DomainMap const& domains, std::size_t leaf) const -> DomainMap
        {
            auto result = domains;
            auto const cells = Cells(leaf);
            for (std::size_t axis = 0; axis < Axes.size(); ++axis) {
                auto const [lo, hi] = domains.at(Axes[axis]);
                auto const step = (hi - lo) / Operon::Scalar(std::size_t { 1 } << Splits[axis]);
                auto const cell = Operon::Scalar(cells[axis]);
                auto cellLo = pappus::fp::ropd<pappus::fp::op_add>(lo, cell * step);
                auto cellHi = pappus::fp::ropu<pappus::fp::op_add>(lo, (cell + Operon::Scalar { 1 }) * step);
                if (cells[axis] == 0) {
                    cellLo = lo;
                }
                if (IsLastCell(axis, cells[axis])) {
                    cellHi = std::max(cellHi, hi);
                }
                result[Axes[axis]] = { cellLo, cellHi };
            }
            return result;
        }
    };

    // The interval bound of `tree` over the hull of the leaf boxes of `plan` (SIMD over the leaves for a single axis
    // and for packed multi-axis leaves), intersected with the whole-box bound so that subdividing can never weaken
    // it; the whole-box bound when `plan` is null or a leaf has no finite bound.
    auto BisectedIntervalBound(Tree const& tree, IntervalEvaluator<Operon::Scalar>::DomainMap const& dom,
        IntervalSubdivisionPlan const* plan) -> BoundResult
    {
        using WScalar = eve::wide<Operon::Scalar>;
        constexpr int WSize = static_cast<int>(eve::cardinal_v<WScalar>);

        auto const directBound = [&]() -> BoundResult {
            IntervalEvaluator<Operon::Scalar> ie(&tree, dom);
            return ie(tree.GetCoefficients());
        };

        // Both the hull of the leaf bounds and the whole-box bound enclose the range over the box, so their
        // intersection does too, and it is never wider than either. (The packed leaf evaluation is not guaranteed to
        // be inclusion-monotone against the scalar whole-box evaluation, e.g. across divisors that touch zero.) A
        // whole-box evaluation that fails leaves the hull as the bound; disjoint bounds mean one of them is unsound, so
        // no bound is reported.
        auto const intersectWithWholeBox = [&](Interval const& hull) -> BoundResult {
            auto const whole = directBound();
            if (!whole) {
                return hull;
            }
            auto const intersection = hull & *whole;
            if (whole->is_empty() || intersection.is_empty()) {
                return tl::unexpected(std::string { "subdivided and whole-box interval bounds are disjoint" });
            }
            return intersection;
        };

        if (plan == nullptr) {
            return directBound();
        }

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
                // The segment width of pappus is (hi - lo) / 2^splits, which underflows to zero for a subnormal
                // width: the top of the last cell of an axis is pinned to hi, as in LeafDomains.
                Pack pack(subdivision, 0);
                constexpr auto lanes = WScalar::size();
                Operon::Vector<std::array<Operon::Scalar, lanes>> upper(plan->Axes.size());
                for (; first + Pack::width <= nLeaves; first += Pack::width) {
                    pack.refill(subdivision, first);
                    for (std::size_t axis = 0; axis < plan->Axes.size(); ++axis) {
                        std::copy_n(pack.upper_data(axis), lanes, upper[axis].data());
                    }
                    for (std::size_t lane = 0; lane < lanes; ++lane) {
                        auto const cells = plan->Cells(first + lane);
                        for (std::size_t axis = 0; axis < plan->Axes.size(); ++axis) {
                            if (plan->IsLastCell(axis, cells[axis])) {
                                upper[axis][lane] = std::max(upper[axis][lane], dom.at(plan->Axes[axis]).second);
                            }
                        }
                    }
                    for (std::size_t axis = 0; axis < plan->Axes.size(); ++axis) {
                        overrides[axis] = { plan->Axes[axis], pack.lower_data(axis), upper[axis].data() };
                    }
                    auto const batch = evaluator(coeff, overrides);
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
                    auto const bound = tail(coeff);
                    if (!bound || bound->is_empty() || !std::isfinite(bound->inf()) || !std::isfinite(bound->sup())) {
                        return directBound();
                    }
                    result = result
                        ? Interval(std::min(result->inf(), bound->inf()), std::max(result->sup(), bound->sup()))
                        : *bound;
                }
                return result ? intersectWithWholeBox(*result) : directBound();
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
            // on this toolchain, so the bisected axis's per-lane bound is passed straight into operator() per
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
                typename IntervalEvaluator<WScalar>::LaneOverride const laneOverride { widest,
                    reinterpret_cast<Operon::Scalar const*>(&leafLo),
                    reinterpret_cast<Operon::Scalar const*>(&leafHi) };
                auto const batch = wie(coeff, std::span { &laneOverride, 1 });
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
                if (k + 1 == nLeaves) {
                    leafHi = std::max(leafHi, hi);
                }
                tailDom[widest] = { leafLo, leafHi };
                IntervalEvaluator<Operon::Scalar> ie(&tree, tailDom);
                auto const seg = ie(coeff);
                if (!seg || seg->is_empty() || !std::isfinite(seg->inf()) || !std::isfinite(seg->sup())) {
                    return directBound();
                }
                result = result ? Interval(std::min(result->inf(), seg->inf()), std::max(result->sup(), seg->sup()))
                                : *seg;
            }
            if (!result || !std::isfinite(result->inf()) || !std::isfinite(result->sup())) {
                return directBound();
            }
            return intersectWithWholeBox(*result);
        } catch (std::exception const&) {
            return directBound();
        }
    }

    // The whole-box affine+interval intersection bound (Combined or Affine mode).
    auto WholeBoxBound(Tree const& tree, AffineEvaluator<Operon::Scalar>& ae, ShapeBoundOptions const& opts)
        -> BoundResult
    {
        // Affine forms cannot represent every interval enclosure. In particular, a zero-crossing denominator is
        // unbounded and a variable exponent may reject an otherwise valid constant integer power. Fall back to
        // the interval evaluator, which can conservatively represent those cases.
        auto const IntervalBound = [&]() {
            return EvaluateIntervalBound(tree, IntervalEvaluator<Operon::Scalar>::DomainMap { ae.Domains() });
        };

        ae.SetTree(&tree);
        auto affineResult = ae(tree.GetCoefficients());
        if (!affineResult) {
            auto bound = IntervalBound();
            if (bound) {
                return bound;
            }
            return tl::unexpected(fmt::format(
                "affine evaluation failed: {}; interval fallback failed: {}", affineResult.error(), bound.error()));
        }
        auto const& affine = *affineResult;
        // Catastrophic cancellation can make this float32 enclosure unsound: an intermediate center orders
        // of magnitude larger than the result implies a rounding-error floor exceeding the tracked radius.
        // Treat as uncertified rather than trusting a possibly-wrong interval.
        constexpr auto illConditionedThreshold = Operon::Scalar { 4 };
        constexpr auto eps = std::numeric_limits<Operon::Scalar>::epsilon();
        auto const impliedErrorFloor = ae.MaxAbsCenter() * eps;
        // A zero radius means every noise symbol cancelled (e.g. x - x): structurally sound, not an
        // underestimate, so only judge forms that track real variable uncertainty.
        auto const r = affine.radius();
        if (r > 0 && impliedErrorFloor > illConditionedThreshold * r) {
            auto bound = IntervalBound();
            if (bound) {
                return bound;
            }
            return tl::unexpected(fmt::format("ill-conditioned: intermediate magnitude implies rounding error {} "
                                              "exceeds result radius {}; interval fallback failed: {}",
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
        if (opts.Mode == ShapeBoundMode::Affine) {
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
    }

    // The bound of a mode subdivided by `plan`: the hull, over the leaf boxes of the plan, of the mode's own bound on
    // the leaf (`leafBound` maps a leaf domain map to that bound), intersected with the whole-box bound `wholeBox`.
    // The leaf boxes cover the domain box (LeafDomains pins the outer endpoints), so the hull is a sound
    // enclosure; the whole-box bound is one too, hence so is their intersection, and it is never wider than the
    // whole-box bound -- which matters for affine arithmetic, whose enclosures are not inclusion-isotone: a
    // sub-box can come out wider than the whole box. A leaf without a finite bound (or an exception) gives up on
    // subdividing: the result is the whole-box bound. Disjoint hull and whole-box bound mean one of them is
    // unsound; the whole-box bound is returned.
    template <typename LeafBound>
    auto SubdividedBound(IntervalSubdivisionPlan const& plan, IntervalSubdivisionPlan::DomainMap const& domains,
        BoundResult wholeBox, LeafBound&& leafBound) -> BoundResult
    {
        std::optional<Interval> hull;
        try {
            auto const nLeaves = std::size_t { 1 } << plan.Depth;
            for (std::size_t leaf = 0; leaf < nLeaves; ++leaf) {
                auto const bound = leafBound(plan.LeafDomains(domains, leaf));
                if (!bound || bound->is_empty() || !std::isfinite(bound->inf()) || !std::isfinite(bound->sup())) {
                    return wholeBox;
                }
                hull = hull ? Interval(std::min(hull->inf(), bound->inf()), std::max(hull->sup(), bound->sup()))
                            : *bound;
            }
        } catch (std::exception const&) {
            return wholeBox;
        }
        if (!hull) {
            return wholeBox;
        }
        if (!wholeBox) {
            return *hull;
        }
        auto const lo = std::max(hull->inf(), wholeBox->inf());
        auto const hi = std::min(hull->sup(), wholeBox->sup());
        return lo <= hi ? BoundResult { Interval(lo, hi) } : wholeBox;
    }

    // The Combined/Affine bound of `tree`: the whole-box bound, or with a `plan` the bound subdivided over it. `ae`
    // is shared across the constraints of a bound set (SetTree retargets it at `tree`); each leaf box gets an
    // evaluator of its own.
    auto ComputeBound(Tree const& tree, AffineEvaluator<Operon::Scalar>& ae, ShapeBoundOptions const& opts,
        IntervalSubdivisionPlan const* plan) -> BoundResult
    {
        auto wholeBox = WholeBoxBound(tree, ae, opts);
        if (plan == nullptr) {
            return wholeBox;
        }
        return SubdividedBound(*plan, ae.Domains(), std::move(wholeBox), [&](IntervalSubdivisionPlan::DomainMap leaf) {
            AffineEvaluator<Operon::Scalar> leafEvaluator(&tree, std::move(leaf));
            return WholeBoxBound(tree, leafEvaluator, opts);
        });
    }

    // Interval-only bound: no AffineEvaluator is built (that was ~24% wasted work per Measure() call).
    // BisectedIntervalBound has its own fallback via directBound(). `ie` is shared across the
    // constraints of a bound set; SetTree retargets it at `tree` (the identity tree or a derivative slice)
    // instead of constructing a fresh evaluator per call. A `plan` subdivides the bound. The options are unused: the
    // signature matches the affine overload so that BoundFor serves both backends.
    auto ComputeBound(Tree const& tree, IntervalEvaluator<Operon::Scalar>& ie, ShapeBoundOptions const& /*opts*/,
        IntervalSubdivisionPlan const* plan) -> BoundResult
    {
        if (plan != nullptr) {
            return BisectedIntervalBound(tree, ie.Domains(), plan);
        }
        ie.SetTree(&tree);
        return ie(tree.GetCoefficients());
    }

    // The bound for one derivative constraint: the (possibly twice-)differentiated tree -- an identically-zero
    // derivative bounds to the degenerate interval [0, 0]. `op` must be FirstDerivative or SecondDerivative;
    // Identity bounds `tree` itself through ComputeBound. `Certain[k] == false` means the derivative dag hit an op
    // with no rule on this variable's path (see tree_diff.hpp); reported as an error result here, same as any other
    // can't-certify case.
    //
    // dag1 is the first-order gradient-dag of `tree`, built once per bound set by the caller and shared across
    // every derivative constraint in it (pure function of tree/coeff, independent of which variable). dag2
    // (SecondDerivative) is still built per-call from the first-derivative tree `d1` (SliceDagRoot of dag1's root for
    // `variable`, so it holds only the nodes that root depends on), which IS variable-specific.
    template <typename Evaluator>
    auto BoundFor(ShapeConstraintOp op, Operon::Hash variable, Evaluator& evaluator, VariableGradientDag const& dag1,
        ShapeBoundOptions const& opts, IntervalSubdivisionPlan const* plan) -> BoundResult
    {
        auto const zero = [] { return BoundResult(Interval(Operon::Scalar { 0 }, Operon::Scalar { 0 })); };

        auto const i1 = VariableIndex(dag1, variable);
        if (!i1) {
            return zero();
        }
        if (!dag1.Certain[*i1]) {
            return tl::unexpected("variable derivative involves an op with no differentiation rule");
        }
        auto d1 = SliceToTree(dag1, dag1.Roots[*i1]);
        if (op == ShapeConstraintOp::FirstDerivative) {
            return d1 ? ComputeBound(*d1, evaluator, opts, plan) : zero();
        }

        // SecondDerivative (symbolic path: Affine, Combined, and whatever the Interval-mode second-order
        // jet pass -- SecondDerivativeJetBounds -- does not decide): differentiate the
        // materialized first-derivative tree again, same variable both times — mixed partials aren't needed
        // by any constraint in this codebase's problem set.
        if (!d1) {
            return zero();
        }
        // Only `variable`'s derivative is used, so only it is differentiated (an unfiltered dag2 would also
        // differentiate d1 w.r.t. every other variable of the tree and leave their nodes in front of this root). No
        // coefficient span: d1 is a standalone tree whose Optimize nodes already hold the live coefficient values.
        auto dag2 = BuildVariableGradientDag(*d1, {}, Operon::Span<Operon::Hash const> { &variable, 1 });
        auto const i2 = VariableIndex(dag2, variable);
        if (!i2) {
            return zero();
        }
        if (!dag2.Certain[*i2]) {
            return tl::unexpected("variable derivative involves an op with no differentiation rule");
        }
        auto d2 = SliceToTree(dag2, dag2.Roots[*i2]);
        return d2 ? ComputeBound(*d2, evaluator, opts, plan) : zero();
    }

    auto ResolveShapeConstraintContext(gsl::not_null<Operon::Problem const*> problem,
        ShapeConstraintSet const& constraints, Operon::Vector<Operon::Hash>& constraintVarHash,
        Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>>& domainsByHash, std::string_view owner)
        -> void
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
            throw std::invalid_argument(fmt::format("{}: input variable '{}' has no entry in 'domains'", owner,
                v ? v->Name : fmt::format("<hash {}>", hash)));
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
                throw std::invalid_argument(
                    fmt::format("{}: constraint Bound [{}, {}] has lo > hi", owner, c.Bound->first, c.Bound->second));
            }

            if (c.Op == ShapeConstraintOp::Identity) {
                constraintVarHash.push_back(Operon::Hash {});
                continue;
            }
            auto v = ds->GetVariable(c.Variable);
            if (!v) {
                throw std::invalid_argument(
                    fmt::format("{}: constraint references unknown variable '{}'", owner, c.Variable));
            }
            if (!domainsByHash.contains(v->Hash)) {
                throw std::invalid_argument(
                    fmt::format("{}: constraint on '{}' has no matching entry in 'domains'", owner, c.Variable));
            }
            constraintVarHash.push_back(v->Hash);
        }
    }

    auto ConstraintViolation(ShapeConstraint const& c, Interval const& bound) -> Operon::Scalar
    {
        if (c.Sign) {
            return (*c.Sign > 0) ? std::max(Operon::Scalar { 0 }, -bound.inf())
                                 : std::max(Operon::Scalar { 0 }, bound.sup());
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

    // Runs one jet pass of `Evaluator` (differentiating `variables`) and returns its Bounds; `Field` selects the
    // per-direction enclosure the caller consumes (Bounds::Derivative or Bounds::Second).
    //
    // Without a `plan` this is the single whole-box pass (ShapeBoundMode::Interval, depth 0). With a plan it is the
    // subdivided route (Interval, depth > 0): one pass per leaf box of the plan (scalar, one evaluator per leaf),
    // and the enclosure of a direction is the hull of its per-leaf enclosures. A direction is decided by its leaves
    // only if every leaf reports Certain with a finite enclosure; otherwise (a leaf failed, overflowed, or hit an op
    // with no rule) the direction takes the whole-box pass's status and enclosure instead. That fallback is exactly
    // what depth 0 computes, so it is sound. A decided direction is not clamped to the whole-box enclosure: leaf
    // boxes lie inside the box, so interval inclusion keeps the hull inside the whole-box enclosure, except for the
    // rounding of the pinned outer endpoints (a leaf may stick out of the box by one ulp at the top, see
    // LeafDomains). (The alternative, bisecting the symbolic derivative DAG, is also sound but may be wider than
    // the whole-box jet bound.)
    // Each leaf is a subset of the whole box, so a leaf failure normally implies the whole-box pass fails too and
    // the constraint then falls through to the symbolic path, as in Interval mode.
    template <typename Evaluator, auto Field>
    auto EvaluateJetLeaves(Tree const& tree, IntervalSubdivisionPlan::DomainMap const& domains,
        IntervalSubdivisionPlan const* plan, Operon::Span<Operon::Hash const> variables,
        Operon::Span<Operon::Scalar const> coefficients) -> std::optional<typename Evaluator::Bounds>
    {
        using Bounds = typename Evaluator::Bounds;
        auto const wholeBox = [&]() -> std::optional<Bounds> {
            Evaluator evaluator(&tree, domains, variables);
            if (auto result = evaluator(coefficients)) {
                return *result;
            }
            return std::nullopt;
        };
        if (plan == nullptr) {
            return wholeBox();
        }

        Bounds hull;
        hull.Directions = variables.size();
        Operon::Vector<bool> decided(variables.size(), true);
        auto undecided = std::size_t { 0 };
        auto const nLeaves = std::size_t { 1 } << plan->Depth;
        for (std::size_t leaf = 0; leaf < nLeaves && undecided < variables.size(); ++leaf) {
            auto const leafDomains = plan->LeafDomains(domains, leaf);
            Evaluator evaluator(&tree, leafDomains, variables);
            auto const result = evaluator(coefficients);
            for (std::size_t k = 0; k < variables.size(); ++k) {
                if (!decided[k]) {
                    continue;
                }
                auto const ok = result && result->Status[k] == IntervalJetStatus::Certain
                    && std::isfinite((*result.*Field)[k].inf()) && std::isfinite((*result.*Field)[k].sup());
                if (!ok) {
                    decided[k] = false;
                    ++undecided;
                    continue;
                }
                auto const& enclosure = (*result.*Field)[k];
                (hull.*Field)[k] = leaf == 0 ? enclosure
                                             : Interval(std::min((hull.*Field)[k].inf(), enclosure.inf()),
                                                   std::max((hull.*Field)[k].sup(), enclosure.sup()));
                hull.Status[k] = IntervalJetStatus::Certain;
            }
        }
        if (undecided == 0) {
            return hull;
        }
        auto const whole = wholeBox();
        for (std::size_t k = 0; k < variables.size(); ++k) {
            if (decided[k]) {
                continue;
            }
            hull.Status[k] = whole ? whole->Status[k] : IntervalJetStatus::Failed;
            if (whole) {
                (hull.*Field)[k] = (*whole.*Field)[k];
            }
        }
        return hull;
    }

    // Interval-mode bounds of FirstDerivative constraints through IntervalJetEvaluator. The variable selection and
    // lazy chunk evaluation deliberately mirror SecondDerivativeJetBounds below: duplicate constraints reuse one
    // direction, Identity never consumes one, and each pass has at most four directions. Failed/non-finite results
    // fall back to the materialized symbolic derivative tree; Uncertain remains the symbolic no-rule result.
    class FirstDerivativeJetBounds {
    public:
        using Evaluator = IntervalJetEvaluator<Operon::Scalar>;
        static_assert(std::is_same_v<Evaluator::Interval, Interval>);
        static constexpr std::size_t MaxDirections = Evaluator::MaxDirections;

        FirstDerivativeJetBounds(ShapeConstraintSet const& constraints,
            Operon::Vector<Operon::Hash> const& constraintVarHash, IntervalSubdivisionPlan const* plan = nullptr)
            : direction_(constraints.Constraints.size(), NoDirection)
            , plan_(plan)
        {
            for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
                auto const hash = constraintVarHash[i];
                if (constraints.Constraints[i].Op != ShapeConstraintOp::FirstDerivative || hash == Operon::Hash {}) {
                    continue;
                }
                auto it = std::ranges::find(variables_, hash);
                if (it == variables_.end()) {
                    it = variables_.insert(variables_.end(), hash);
                }
                direction_[i] = static_cast<std::size_t>(it - variables_.begin());
            }
            chunks_.resize((variables_.size() + MaxDirections - 1) / MaxDirections);
        }

        [[nodiscard]] auto Empty() const noexcept -> bool { return variables_.empty(); }

        [[nodiscard]] auto Bound(std::size_t i, Operon::Tree const& tree,
            Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> const& domains,
            Operon::Span<Operon::Scalar const> coefficients) -> std::optional<BoundResult>
        {
            if (direction_[i] == NoDirection) {
                return std::nullopt;
            }
            auto const chunkIndex = direction_[i] / MaxDirections;
            auto const direction = direction_[i] % MaxDirections;
            auto& chunk = chunks_[chunkIndex];
            if (!chunk.Evaluated) {
                chunk.Evaluated = true;
                auto const first = chunkIndex * MaxDirections;
                auto const count = std::min(MaxDirections, variables_.size() - first);
                if (auto result = EvaluateJetLeaves<Evaluator, &Evaluator::Bounds::Derivative>(tree, domains, plan_,
                        Operon::Span<Operon::Hash const> { variables_.data() + first, count }, coefficients)) {
                    chunk.Result = *result;
                }
            }
            if (!chunk.Result) {
                return std::nullopt;
            }
            switch (chunk.Result->Status[direction]) {
            case IntervalJetStatus::Uncertain:
                return BoundResult { tl::unexpected(
                    "variable derivative involves an op with no differentiation rule") };
            case IntervalJetStatus::Certain: {
                auto const& derivative = chunk.Result->Derivative[direction];
                if (std::isfinite(derivative.inf()) && std::isfinite(derivative.sup())) {
                    return BoundResult { derivative };
                }
                return std::nullopt;
            }
            case IntervalJetStatus::Failed:
                break;
            }
            return std::nullopt;
        }

    private:
        static constexpr auto NoDirection = std::numeric_limits<std::size_t>::max();

        struct Chunk {
            bool Evaluated { false };
            std::optional<Evaluator::Bounds> Result;
        };

        Operon::Vector<Operon::Hash> variables_;
        Operon::Vector<std::size_t> direction_;
        Operon::Vector<Chunk> chunks_;
        IntervalSubdivisionPlan const* plan_ { nullptr }; // null: whole-box pass (Interval mode)
    };

    // Interval-mode bounds of the SecondDerivative constraints through IntervalSecondOrderJetEvaluator: one
    // post-order pass over the original tree yields d^2 f / d x^2 for up to MaxDirections variables at once, instead
    // of materializing the first-derivative tree and differentiating it again (BoundFor's symbolic path).
    //
    // Direction selections are the distinct, nonzero variable hashes of the SecondDerivative constraints, in order of
    // first appearance: constraints that repeat a variable share one direction, and constraints without a variable
    // (Identity, whose hash is the zero marker) never take a direction. Selections beyond MaxDirections run in
    // further passes (chunks), each evaluated lazily on first use.
    class SecondDerivativeJetBounds {
    public:
        using Evaluator = IntervalSecondOrderJetEvaluator<Operon::Scalar>;
        static_assert(std::is_same_v<Evaluator::Interval, Interval>);
        static constexpr std::size_t MaxDirections = Evaluator::MaxDirections;

        SecondDerivativeJetBounds(ShapeConstraintSet const& constraints,
            Operon::Vector<Operon::Hash> const& constraintVarHash, IntervalSubdivisionPlan const* plan = nullptr)
            : direction_(constraints.Constraints.size(), NoDirection)
            , plan_(plan)
        {
            for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
                auto const hash = constraintVarHash[i];
                if (constraints.Constraints[i].Op != ShapeConstraintOp::SecondDerivative || hash == Operon::Hash {}) {
                    continue;
                }
                auto it = std::ranges::find(variables_, hash);
                if (it == variables_.end()) {
                    it = variables_.insert(variables_.end(), hash);
                }
                direction_[i] = static_cast<std::size_t>(it - variables_.begin());
            }
            chunks_.resize((variables_.size() + MaxDirections - 1) / MaxDirections);
        }

        [[nodiscard]] auto Empty() const noexcept -> bool { return variables_.empty(); }

        // The bound of constraint `i` from the jet pass, or nullopt when the jet path does not decide it and the
        // symbolic path must (not a SecondDerivative constraint, an unsupported node kind such as a user-registered
        // operation, a domain error or a non-finite enclosure). An operation without a differentiation rule on the
        // variable's path is decided here, as an error, exactly as the symbolic path reports it.
        [[nodiscard]] auto Bound(std::size_t i, Operon::Tree const& tree,
            Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> const& domains,
            Operon::Span<Operon::Scalar const> coefficients) -> std::optional<BoundResult>
        {
            if (direction_[i] == NoDirection) {
                return std::nullopt;
            }
            auto const chunkIndex = direction_[i] / MaxDirections;
            auto const direction = direction_[i] % MaxDirections;
            auto& chunk = chunks_[chunkIndex];
            if (!chunk.Evaluated) {
                chunk.Evaluated = true;
                auto const first = chunkIndex * MaxDirections;
                auto const count = std::min(MaxDirections, variables_.size() - first);
                if (auto result = EvaluateJetLeaves<Evaluator, &Evaluator::Bounds::Second>(tree, domains, plan_,
                        Operon::Span<Operon::Hash const> { variables_.data() + first, count }, coefficients)) {
                    chunk.Result = *result;
                }
            }
            if (!chunk.Result) {
                return std::nullopt;
            }
            switch (chunk.Result->Status[direction]) {
            case IntervalJetStatus::Uncertain:
                return BoundResult { tl::unexpected(
                    "variable derivative involves an op with no differentiation rule") };
            case IntervalJetStatus::Certain: {
                auto const& second = chunk.Result->Second[direction];
                if (std::isfinite(second.inf()) && std::isfinite(second.sup())) {
                    return BoundResult { second };
                }
                return std::nullopt;
            }
            case IntervalJetStatus::Failed:
                break;
            }
            return std::nullopt;
        }

    private:
        static constexpr auto NoDirection = std::numeric_limits<std::size_t>::max();

        struct Chunk {
            bool Evaluated { false };
            std::optional<Evaluator::Bounds> Result;
        };

        // distinct, nonzero; direction d of chunk c is variables_[c * MaxDirections + d]
        Operon::Vector<Operon::Hash> variables_;
        Operon::Vector<std::size_t> direction_; // per constraint: index into variables_, or NoDirection
        Operon::Vector<Chunk> chunks_;
        IntervalSubdivisionPlan const* plan_ { nullptr }; // null: whole-box pass (Interval mode)
    };

    auto MeasureConstraints(ShapeConstraintSet const& constraints,
        Operon::Vector<Operon::Hash> const& constraintVarHash,
        Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> const& domainsByHash,
        Operon::Tree const& tree, Operon::Scalar unknownViolation, std::optional<Operon::LinearScaling> scaling,
        ShapeBoundOptions const& opts) -> ShapeConstraintMeasurementSummary
    {
        ShapeConstraintMeasurementSummary summary;
        summary.Measurements.reserve(constraints.Constraints.size());

        // Built on first use by BoundFor; shared across every derivative constraint in this
        // bound set (see BoundFor's comment). Identity constraints never touch it, so it is lazily constructed
        // only when a bound set actually contains a derivative constraint.
        std::optional<Operon::Vector<Operon::Scalar>> coefficients;
        auto const Coefficients = [&]() -> Operon::Span<Operon::Scalar const> {
            if (!coefficients) {
                coefficients.emplace(tree.GetCoefficients());
            }
            return *coefficients;
        };
        std::optional<VariableGradientDag> dag1;
        auto const SharedDag1 = [&]() -> VariableGradientDag const& {
            if (!dag1) {
                dag1.emplace(BuildVariableGradientDag(tree, Coefficients(), constraintVarHash));
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

        // Every mode bounds the model over the 2^BisectionDepth leaf boxes of one subdivision plan, made from the
        // original tree and the full domain map: all derivative directions of a chunk share one jet pass per leaf,
        // so one leaf schedule serves every constraint. No plan (depth 0, or no splittable axis) is the whole-box
        // bound.
        auto const plan = IntervalSubdivisionPlan::Make(tree, domainsByHash, opts.BisectionDepth);
        auto const* planPtr = plan ? &*plan : nullptr;
        // Every route that evaluates one scalar evaluator per leaf (jet passes, affine, combined) works on the plan
        // capped at MaxScalarDepth; only the interval Identity route, which is SIMD over the leaves, takes the full
        // depth.
        auto const cappedPlan = plan && plan->Depth > IntervalSubdivisionPlan::MaxScalarDepth
            ? std::optional { plan->Capped(IntervalSubdivisionPlan::MaxScalarDepth) }
            : std::nullopt;
        auto const* scalarPlan = cappedPlan ? &*cappedPlan : planPtr;
        auto const* identityPlan = opts.Mode == ShapeBoundMode::Interval ? planPtr : scalarPlan;
        // Interval mode: derivative constraints are bounded by the corresponding batched jet pass over the original
        // tree (whole box, or the hull over the leaf boxes of `plan`). Every other mode, and every constraint a jet
        // pass does not decide, uses BoundFor. A derivative constraint the jet pass does not decide goes to the
        // symbolic whole-box bound in Interval mode (only the Identity constraint is subdivided there); Combined and
        // Affine subdivide the derivative tree itself.
        auto const jetMode = opts.Mode == ShapeBoundMode::Interval;
        auto const* derivativePlan = jetMode ? nullptr : scalarPlan;
        std::optional<FirstDerivativeJetBounds> firstDerivativeJet;
        std::optional<SecondDerivativeJetBounds> secondDerivativeJet;
        if (jetMode) {
            FirstDerivativeJetBounds firstCandidate(constraints, constraintVarHash, scalarPlan);
            if (!firstCandidate.Empty()) {
                firstDerivativeJet.emplace(std::move(firstCandidate));
            }
            SecondDerivativeJetBounds secondCandidate(constraints, constraintVarHash, scalarPlan);
            if (!secondCandidate.Empty()) {
                secondDerivativeJet.emplace(std::move(secondCandidate));
            }
        }
        // The per-constraint loop shared by both backends. SharedDag1() is only built for a derivative constraint
        // that the jet path did not decide.
        auto const BoundEach = [&](auto& evaluator) {
            for (std::size_t i = 0; i < constraints.Constraints.size(); ++i) {
                auto const& c = constraints.Constraints[i];
                if (c.Op == ShapeConstraintOp::Identity) {
                    Apply(i, ComputeBound(tree, evaluator, opts, identityPlan));
                    continue;
                }
                if (firstDerivativeJet && c.Op == ShapeConstraintOp::FirstDerivative) {
                    if (auto jetBound = firstDerivativeJet->Bound(i, tree, domainsByHash, Coefficients())) {
                        Apply(i, *jetBound);
                        continue;
                    }
                }
                if (secondDerivativeJet && c.Op == ShapeConstraintOp::SecondDerivative) {
                    if (auto jetBound = secondDerivativeJet->Bound(i, tree, domainsByHash, Coefficients())) {
                        Apply(i, *jetBound);
                        continue;
                    }
                }
                Apply(i, BoundFor(c.Op, constraintVarHash[i], evaluator, SharedDag1(), opts, derivativePlan));
            }
        };

        // Interval mode never touches AffineEvaluator's affine machinery -- skip
        // constructing it, sharing one IntervalEvaluator across every constraint in this set instead (same
        // reasoning as the AffineEvaluator sharing just below: skips a per-constraint domainSlots_/primal_
        // regrowth that profiling showed as real allocation churn).
        if (opts.Mode == ShapeBoundMode::Interval) {
            IntervalEvaluator<Operon::Scalar> ie(&tree, domainsByHash);

            BoundEach(ie);
            return summary;
        }

        // One AffineEvaluator shared across every bound in this set, skipping a per-constraint DomainMap copy
        // and primal_ regrowth. SetTree() retargets it at each constraint's slice; ctx_'s monotonic noise-symbol
        // counter stays sound since the bounds are consumed as intervals independently of each other.
        AffineEvaluator<Operon::Scalar> ae(&tree, domainsByHash);
        BoundEach(ae);
        return summary;
    }

} // namespace

namespace detail {
    auto BisectionSplits(Tree const& tree,
        Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> const& domains, int depth)
        -> std::vector<std::pair<Operon::Hash, int>>
    {
        std::vector<std::pair<Operon::Hash, int>> result;
        if (auto const plan = IntervalSubdivisionPlan::Make(tree, domains, depth)) {
            for (std::size_t axis = 0; axis < plan->Axes.size(); ++axis) {
                if (plan->Splits[axis] > 0) {
                    result.emplace_back(plan->Axes[axis], plan->Splits[axis]);
                }
            }
        }
        return result;
    }
} // namespace detail

auto ParseShapeEnforcement(std::string const& str) -> ShapeConstraintEnforcement
{
    using E = ShapeConstraintEnforcement;
    if (str == "hard-reject") {
        return E::HardReject;
    }
    if (str == "penalty") {
        return E::Penalty;
    }
    if (str == "extra-objective") {
        return E::ExtraObjective;
    }
    if (str == "penalty,extra-objective" || str == "extra-objective,penalty") {
        return E::PenaltyAndExtraObjective;
    }
    throw std::invalid_argument(fmt::format("unable to parse shape-enforcement argument '{}'", str));
}

auto ParseShapeBoundOptions(std::string const& str) -> ShapeBoundOptions
{
    auto const colon = str.find(':');
    auto const name = str.substr(0, colon);
    ShapeBoundOptions options {};
    if (name == "combined") {
        options.Mode = ShapeBoundMode::Combined;
    } else if (name == "interval") {
        options.Mode = ShapeBoundMode::Interval;
    } else if (name == "affine") {
        options.Mode = ShapeBoundMode::Affine;
    } else {
        throw std::invalid_argument(fmt::format("unable to parse shape-bound-mode argument '{}': expected combined, "
                                                "interval or affine, optionally followed by :N with N in [0, 20] "
                                                "(for example interval, interval:4, affine:2)",
            str));
    }
    if (colon != std::string::npos) {
        auto const depthText = std::string_view { str }.substr(colon + 1);
        int depth = 0;
        auto const [end, ec] = std::from_chars(depthText.data(), depthText.data() + depthText.size(), depth);
        if (ec != std::errc {} || end != depthText.data() + depthText.size()) {
            throw std::invalid_argument(fmt::format("unable to parse shape-bound-mode depth '{}' in '{}': expected "
                                                    "an integer in [0, 20]",
                depthText, str));
        }
        if (depth < 0 || depth > 20) {
            throw std::invalid_argument(
                fmt::format("shape-bound-mode depth must be in [0, 20] (got '{}' in '{}')", depthText, str));
        }
        options.BisectionDepth = depth;
    }
    return options;
}

auto ValidatePolicy(ShapeConstraintPolicy const& policy, bool isNsga2) -> std::optional<std::string>
{
    if (!std::isfinite(policy.UnknownViolation) || policy.UnknownViolation < Operon::Scalar { 0 }) {
        return "shape unknown violation must be finite and non-negative";
    }
    if (!std::isfinite(policy.PenaltyWeight) || policy.PenaltyWeight < Operon::Scalar { 0 }) {
        return "shape penalty weight must be finite and non-negative";
    }

    // NSGA2 accepts every enforcement mode; GP has a single objective and so no extra-objective modes.
    if (!isNsga2
        && (policy.Enforcement == ShapeConstraintEnforcement::ExtraObjective
            || policy.Enforcement == ShapeConstraintEnforcement::PenaltyAndExtraObjective)) {
        return "shape constraint extra-objective mode is only valid for NSGA2";
    }
    return std::nullopt;
}

ShapeCertifier::ShapeCertifier(gsl::not_null<Operon::Problem const*> problem,
    gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSet constraints,
    Operon::Scalar unknownViolation, ShapeBoundOptions bound)
    : problem_(problem)
    , dtable_(dtable)
    , constraints_(std::move(constraints))
    , unknownViolation_(unknownViolation)
    , bound_(bound)
{
    if (!std::isfinite(unknownViolation_) || unknownViolation_ < Operon::Scalar { 0 }) {
        throw std::invalid_argument("ShapeCertifier: unknown violation must be finite and non-negative");
    }
    if (bound_.BisectionDepth < 0 || bound_.BisectionDepth > 20) {
        throw std::invalid_argument("bisection depth must be in [0, 20]");
    }
    ResolveShapeConstraintContext(problem_, constraints_, constraintVarHash_, domainsByHash_, "ShapeCertifier");
}

auto ShapeCertifier::Measure(Operon::Tree const& tree, std::optional<Operon::LinearScaling> const& transform) const
    -> ShapeConstraintMeasurementSummary
{
    return MeasureConstraints(
        constraints_, constraintVarHash_, domainsByHash_, tree, unknownViolation_, transform, bound_);
}

auto ShapeCertifier::MeasureCanonical(Operon::Tree const& tree, Operon::Span<Operon::Scalar> scratch) const
    -> ShapeConstraintMeasurementSummary
{
    auto const transform = Operon::FitLinearScaling(tree, *problem_, *dtable_, problem_->TrainingRange(), scratch);
    return Measure(tree, transform);
}

namespace {
    auto Rejects(ShapeConstraintEnforcement e) noexcept -> bool { return e == ShapeConstraintEnforcement::HardReject; }

    auto Penalizes(ShapeConstraintEnforcement e) noexcept -> bool
    {
        return e == ShapeConstraintEnforcement::Penalty || e == ShapeConstraintEnforcement::PenaltyAndExtraObjective;
    }

    auto AppendsViolation(ShapeConstraintEnforcement e) noexcept -> bool
    {
        return e == ShapeConstraintEnforcement::ExtraObjective
            || e == ShapeConstraintEnforcement::PenaltyAndExtraObjective;
    }

    auto PenalizedScore(Operon::Scalar primary, Operon::Scalar weight, Operon::Scalar violation) -> Operon::Scalar
    {
        auto const value
            = static_cast<double>(primary) + (static_cast<double>(weight) * static_cast<double>(violation));
        if (!std::isfinite(value) || value >= static_cast<double>(EvaluatorBase::ErrMax)) {
            return EvaluatorBase::ErrMax;
        }
        return static_cast<Operon::Scalar>(value);
    }
} // namespace

detail::ShapeConstraintEnforcer::ShapeConstraintEnforcer(gsl::not_null<Operon::Problem const*> problem,
    gsl::not_null<Operon::ScalarDispatch const*> dtable, ShapeConstraintSpec spec)
    : structural_(std::move(spec.Structural))
    , certifier_(problem, dtable, std::move(spec.Constraints), spec.Policy.UnknownViolation, spec.Bound)
    , policy_(spec.Policy)
    , worstValue_(spec.WorstValue)
{
    for (auto const& objective : structural_) {
        if (objective->ObjectiveCount() != 1) {
            throw std::invalid_argument(
                "ShapeConstrainedValueEvaluator: structural post-objectives must be single-objective");
        }
    }
    // The adapter serves both GP and NSGA2 compositions; reject only policies that no algorithm family accepts.
    auto const nsga2 = ValidatePolicy(policy_, /*isNsga2=*/true);
    auto const gp = ValidatePolicy(policy_, /*isNsga2=*/false);
    if (nsga2 && gp) {
        throw std::invalid_argument("ShapeConstrainedValueEvaluator: " + *gp);
    }
}

auto detail::ShapeConstraintEnforcer::ShapeObjectiveCount() const noexcept -> std::size_t
{
    auto const extra = AppendsViolation(policy_.Enforcement);
    return 1UL + structural_.size() + (extra ? 1UL : 0UL);
}

auto detail::ShapeConstraintEnforcer::ShapePrepare(Operon::Span<Individual const> pop) const -> void
{
    for (auto const& objective : structural_) {
        objective->Prepare(pop);
    }
}

auto detail::ShapeConstraintEnforcer::ShapeStructuralStats() const -> std::tuple<std::size_t, std::size_t, std::size_t>
{
    std::size_t residual = 0;
    std::size_t jacobian = 0;
    std::size_t costFunctionTime = 0;
    for (auto const& objective : structural_) {
        auto const [re, je, cc, ct] = objective->Stats();
        residual += re;
        jacobian += je;
        costFunctionTime += ct;
    }
    return std::tuple { residual, jacobian, costFunctionTime };
}

auto detail::ShapeConstraintEnforcer::ShapeCertify(Operon::Tree const& tree,
    std::optional<EvaluatedBuffer> const& evaluated) const -> ShapeConstraintMeasurementSummary
{
    if (!evaluated) {
        throw std::logic_error("ShapeConstrainedValueEvaluator: the base evaluator produced no prediction");
    }
    return certifier_.Measure(tree, evaluated->Scaling());
}

auto detail::ShapeConstraintEnforcer::ShapeReject(ShapeConstraintMeasurementSummary const& summary) const
    -> std::optional<EvaluatorBase::ReturnType>
{
    if (!Rejects(policy_.Enforcement) || summary.Feasible) {
        return std::nullopt;
    }
    ++violations_;
    return EvaluatorBase::ReturnType(ShapeObjectiveCount(), static_cast<Operon::Scalar>(worstValue_));
}

auto detail::ShapeConstraintEnforcer::ShapeAssemble(Operon::RandomGenerator& rng, Operon::Individual const& ind,
    Operon::Span<Operon::Scalar> buf, ShapeConstraintMeasurementSummary const& summary,
    EvaluatorBase::ReturnType const& primary) const -> EvaluatorBase::ReturnType
{
    if (primary.size() != 1) {
        throw std::logic_error("ShapeConstrainedValueEvaluator: the base evaluator must return exactly one objective");
    }
    auto const raw = primary.front();
    if (!std::isfinite(raw) || raw >= EvaluatorBase::ErrMax) {
        return EvaluatorBase::ReturnType(ShapeObjectiveCount(), EvaluatorBase::ErrMax);
    }

    EvaluatorBase::ReturnType fit;
    fit.reserve(ShapeObjectiveCount());
    fit.push_back(Penalizes(policy_.Enforcement) ? PenalizedScore(raw, policy_.PenaltyWeight, summary.Violation) : raw);
    for (auto const& objective : structural_) {
        auto const f = (*objective)(rng, ind, buf);
        std::ranges::copy(f, std::back_inserter(fit));
    }
    if (AppendsViolation(policy_.Enforcement)) {
        fit.push_back(summary.Violation);
    }
    return fit;
}

} // namespace Operon
