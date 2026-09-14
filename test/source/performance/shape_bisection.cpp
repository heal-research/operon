// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/catch_approx.hpp>

#include <fmt/format.h>
#include <algorithm>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

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

struct ExprCase {
    std::string name;
    std::string expr;
    // (variable name, domain) pairs; first variable's data column doubles
    // as the (unused, linear-scaling-disabled) target placeholder.
    std::vector<std::pair<std::string, std::pair<Operon::Scalar, Operon::Scalar>>> domains;
};

// Diverse op mix: pure dependency-problem arithmetic, trig+pow, division
// (rational), exp/log -- covers every wide<T> body added this session, not
// just the one tree the original scalar-vs-SIMD claim happened to use.
auto const kExprCases = std::vector<ExprCase>{
    { "dependency", "(X1 - 1) * (X1 - 1)", { {"X1", {Operon::Scalar{0}, Operon::Scalar{10}}} } },
    { "trig_pow",   "sin(X1) + cos(X1) * X1 ^ 2 + X2", { {"X1", {Operon::Scalar{-5}, Operon::Scalar{5}}}, {"X2", {Operon::Scalar{-5}, Operon::Scalar{5}}} } },
    { "division",   "X1 / (X2 + 3)", { {"X1", {Operon::Scalar{-2}, Operon::Scalar{2}}}, {"X2", {Operon::Scalar{-2}, Operon::Scalar{2}}} } },
    { "exp_log",    "exp(X1) - log(X2 + 1)", { {"X1", {Operon::Scalar{-2}, Operon::Scalar{2}}}, {"X2", {Operon::Scalar{0.1F}, Operon::Scalar{5}}} } },
};

void RunExprCase(nb::Bench& bench, ExprCase const& ec)
{
    auto const nvars = ec.domains.size();
    auto const nrow = std::size_t{5};
    auto const ncol = nvars + 1;
    Eigen::Array<Operon::Scalar, -1, -1> data(static_cast<Eigen::Index>(nrow), static_cast<Eigen::Index>(ncol));
    for (std::size_t i = 0; i < nrow; ++i) {
        for (std::size_t v = 0; v < nvars; ++v) {
            auto const [lo, hi] = ec.domains[v].second;
            data(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(v)) = lo + (hi - lo) * static_cast<Operon::Scalar>(i) / static_cast<Operon::Scalar>(nrow - 1);
        }
        data(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(ncol - 1)) = Operon::Scalar{0};
    }
    Operon::Dataset ds(gsl::not_null{data.data()}, nrow, ncol);
    auto tree = Operon::InfixParser::Parse(ec.expr, ds);

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({0, nrow});
    problem.SetTestRange({0, nrow});
    problem.SetTarget(fmt::format("X{}", ncol));
    problem.SetLinearScalingEnabled(false);
    Operon::DispatchTable<Operon::Scalar> dtable;
    Operon::Evaluator<Operon::DispatchTable<Operon::Scalar>> nmse(&problem, &dtable, Operon::NMSE{});

    Operon::ShapeConstraintSet cs;
    Operon::IntervalEvaluator<Operon::Scalar>::DomainMap dom;
    for (auto const& [name, bound] : ec.domains) {
        cs.Domains.insert_or_assign(name, bound);
        dom.emplace(ds.GetVariable(name).value().Hash, bound);
    }
    cs.Constraints.push_back({.Op = Operon::ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-1e6}, Operon::Scalar{1e6}}});

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);
    shapeEval.SetBoundMode(Operon::ShapeBoundMode::Interval | Operon::ShapeBoundMode::Bisected);

    constexpr int maxDepth = 8;
    double width0 = -1.0;
    for (int depth = 0; depth <= maxDepth; ++depth) {
        bench.run(fmt::format("{} scalar depth={:02d}", ec.name, depth), [&] {
            auto const b = ScalarBisectedBound(tree, dom, depth);
            nb::doNotOptimizeAway(b.first);
            nb::doNotOptimizeAway(b.second);
        });

        shapeEval.SetBoundOptions({.BisectionDepth = depth});
        bench.run(fmt::format("{} SIMD   depth={:02d}", ec.name, depth), [&] {
            auto r = shapeEval.Measure(tree);
            nb::doNotOptimizeAway(r.Measurements.size());
        });

        auto const [scalarLo, scalarHi] = ScalarBisectedBound(tree, dom, depth);
        auto const r = shapeEval.Measure(tree);
        auto const [simdLo, simdHi] = *r.Measurements[0].Bound;
        REQUIRE(static_cast<double>(scalarLo) == Catch::Approx(static_cast<double>(simdLo)).margin(1e-3));
        REQUIRE(static_cast<double>(scalarHi) == Catch::Approx(static_cast<double>(simdHi)).margin(1e-3));

        double const width = static_cast<double>(simdHi) - static_cast<double>(simdLo);
        if (depth == 0) { width0 = width; }
        auto const& results = bench.results();
        double const scalarNs = results[results.size() - 2].average(nb::Result::Measure::elapsed) * 1e9;
        double const simdNs = results[results.size() - 1].average(nb::Result::Measure::elapsed) * 1e9;
        fmt::print("{:11s}  depth={:2d}  n_leaves={:4d}  bound=[{:12.6f},{:12.6f}]  width={:12.6f}  width_vs_depth0={:7.3f}%  scalar={:11.1f}ns  SIMD={:11.1f}ns  SIMD_speedup={:6.2f}x\n",
            ec.name, depth, 1 << depth, static_cast<double>(simdLo), static_cast<double>(simdHi), width,
            width0 > 0 ? 100.0 * width / width0 : 100.0,
            scalarNs, simdNs, scalarNs / simdNs);
    }
}
} // namespace

// Sweeps bisection depth across several expressions (pure arithmetic
// dependency-problem, trig+pow, division, exp/log -- covering every wide<T>
// body added this session) and compares the production wide<T>-batched SIMD
// path against the scalar (non-SIMD) reference above, at the SAME n_leaves
// -- an apples-to-apples SIMD-vs-scalar comparison, not "bisected vs
// unbisected" (depth=0's cost is a different question and not a fair
// speedup baseline: of course evaluating 1 box is cheaper than evaluating
// N>1 sub-boxes). Bounds are cross-checked identical between the scalar
// reference and the production SIMD path at every depth (both implement
// the same mathematical bisection and must agree).
//
// Type T is fixed at build time by USE_SINGLE_PRECISION (Operon::Scalar);
// run this binary once under each precision to compare float vs double.
TEST_CASE("Shape bisection depth sweep: SIMD vs scalar, multiple expressions", "[performance][shape-constraints][bisection]")
{
    fmt::print("T = {}\n", sizeof(Operon::Scalar) == sizeof(float) ? "float" : "double");
    nb::Bench bench;
    bench.title("shape-bisection-depth").batch(1);

    for (auto const& ec : kExprCases) { RunExprCase(bench, ec); }

    bench.render(nb::templates::csv(), std::cout);
}
