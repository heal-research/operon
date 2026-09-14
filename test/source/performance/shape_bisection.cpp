// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <fmt/format.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/constraint.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/parser/infix.hpp"

namespace nb = ankerl::nanobench;

namespace {
// Scalar (non-SIMD) reference: same widest-axis-pick + uniform-split
// algorithm as the production wide<T>-batched BisectedIntervalBound, but
// every leaf goes through IntervalEvaluator<Operon::Scalar> in a plain
// loop -- no wide<T> batching at all. This is the fair "SIMD vs no-SIMD,
// same n_leaves" baseline: the point of the SIMD rewrite was to make
// exactly this loop faster, not to make bisection itself faster than not
// bisecting (depth=0 already is the no-bisection case and is not a
// meaningful "speedup" baseline for that reason).
auto ScalarBisectedBound(Operon::Tree const& tree, Operon::IntervalEvaluator<Operon::Scalar>::DomainMap const& dom, int depth)
    -> std::pair<Operon::Scalar, Operon::Scalar>
{
    using IE = Operon::IntervalEvaluator<Operon::Scalar>;
    auto const directBound = [&]() -> std::pair<Operon::Scalar, Operon::Scalar> {
        IE ie(&tree, dom);
        auto const iv = ie.Evaluate(tree.GetCoefficients());
        return {iv.inf(), iv.sup()};
    };
    if (depth <= 0) { return directBound(); }

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

    int const nLeaves = 1 << depth;
    auto const lo0 = dom.at(widest).first;
    auto const h = widestDiam / Operon::Scalar(nLeaves);
    auto const coeff = tree.GetCoefficients();

    Operon::Scalar resLo = std::numeric_limits<Operon::Scalar>::infinity();
    Operon::Scalar resHi = -std::numeric_limits<Operon::Scalar>::infinity();
    auto leafDom = dom;
    for (int k = 0; k < nLeaves; ++k) {
        leafDom[widest] = { lo0 + Operon::Scalar(k) * h, lo0 + Operon::Scalar(k + 1) * h };
        IE ie(&tree, leafDom);
        auto const seg = ie.Evaluate(coeff);
        resLo = std::min(resLo, seg.inf());
        resHi = std::max(resHi, seg.sup());
    }
    return {resLo, resHi};
}
} // namespace

// Sweeps bisection depth and compares the production wide<T>-batched SIMD
// path against the scalar (non-SIMD) reference above, at the SAME n_leaves
// -- an apples-to-apples SIMD-vs-scalar comparison, not "bisected vs
// unbisected" (depth=0's cost is a different question and not a fair
// speedup baseline: of course evaluating 1 box is cheaper than evaluating
// N>1 sub-boxes). Also reports bound tightness per depth, using the SIMD
// path's own domain construction so the reported bound matches production
// behavior exactly (the scalar reference's bound is checked for exact
// agreement, not separately reported, since both paths implement the same
// mathematical bisection and must produce identical sound bounds).
// Tree: (X1-1)*(X1-1) over [0,10] -- a dependency-problem case with a known
// true range ([0, 81]) and a known naive interval-overestimate ([-9, 81],
// width 90), so tightening progress is directly measurable.
TEST_CASE("Shape bisection depth sweep: SIMD vs scalar, and bound tightness", "[performance][shape-constraints][bisection]")
{
    constexpr auto nrow = std::size_t{5};
    constexpr auto ncol = std::size_t{2};
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = data(static_cast<Eigen::Index>(i), 0);
    }
    Operon::Dataset ds(gsl::not_null{data.data()}, nrow, ncol);
    auto tree = Operon::InfixParser::Parse("(X1 - 1) * (X1 - 1)", ds);
    auto const x1Hash = ds.GetVariable("X1").value().Hash;

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({0, nrow});
    problem.SetTestRange({0, nrow});
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(false);
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Evaluator<Operon::DispatchTable<Operon::Scalar>> nmse(&problem, &dtable, Operon::NMSE{});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{0}, Operon::Scalar{10}});
    cs.Constraints.push_back({.Op = Operon::ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-1000}, Operon::Scalar{1000}}});

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);
    shapeEval.SetBoundMode(Operon::ShapeBoundMode::Interval | Operon::ShapeBoundMode::Bisected);

    Operon::IntervalEvaluator<Operon::Scalar>::DomainMap const dom{ {x1Hash, {Operon::Scalar{0}, Operon::Scalar{10}}} };

    constexpr double trueWidth = 81.0;
    constexpr double naiveWidth = 90.0;

    nb::Bench bench;
    bench.title("shape-bisection-depth").batch(1);

    constexpr int maxDepth = 10;
    for (int depth = 0; depth <= maxDepth; ++depth) {
        bench.run(fmt::format("scalar depth={:02d}", depth), [&] {
            auto const b = ScalarBisectedBound(tree, dom, depth);
            nb::doNotOptimizeAway(b.first);
            nb::doNotOptimizeAway(b.second);
        });

        shapeEval.SetBoundOptions({.BisectionDepth = depth});
        bench.run(fmt::format("SIMD   depth={:02d}", depth), [&] {
            auto r = shapeEval.Measure(tree);
            nb::doNotOptimizeAway(r.Measurements.size());
        });

        auto const [scalarLo, scalarHi] = ScalarBisectedBound(tree, dom, depth);
        auto const r = shapeEval.Measure(tree);
        auto const [simdLo, simdHi] = *r.Measurements[0].Bound;
        REQUIRE(static_cast<double>(scalarLo) == Catch::Approx(static_cast<double>(simdLo)).margin(1e-4));
        REQUIRE(static_cast<double>(scalarHi) == Catch::Approx(static_cast<double>(simdHi)).margin(1e-4));

        double const width = static_cast<double>(simdHi) - static_cast<double>(simdLo);
        auto const& results = bench.results();
        double const scalarNs = results[results.size() - 2].average(nb::Result::Measure::elapsed) * 1e9;
        double const simdNs = results[results.size() - 1].average(nb::Result::Measure::elapsed) * 1e9;
        fmt::print("depth={:2d}  n_leaves={:5d}  bound=[{:10.6f},{:10.6f}]  width={:9.6f}  excess_over_true={:8.4f}%  pct_of_naive={:8.4f}%  scalar={:10.1f}ns  SIMD={:10.1f}ns  SIMD_speedup={:6.2f}x\n",
            depth, 1 << depth, static_cast<double>(simdLo), static_cast<double>(simdHi), width,
            100.0 * (width - trueWidth) / trueWidth, 100.0 * width / naiveWidth,
            scalarNs, simdNs, scalarNs / simdNs);
    }

    bench.render(nb::templates::csv(), std::cout);
}
