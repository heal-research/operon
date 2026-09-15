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

// Scalar reference: same balanced multi-axis grid schedule as production, but
// every leaf uses IntervalEvaluator<Operon::Scalar>. Production uses the
// wide<T> path for one axis and the scalar fallback for multi-axis grids; both
// enumerate exactly the same boxes.
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

    std::vector<Operon::Hash> axes;
    std::vector<Operon::Scalar> widths;
    for (auto const& n : tree.Nodes()) {
        if (!n.IsVariable() || std::ranges::find(axes, n.HashValue) != axes.end()) { continue; }
        auto const it = dom.find(n.HashValue);
        if (it == dom.end()) { continue; }
        auto const width = it->second.second - it->second.first;
        if (width <= Operon::Scalar{0}) { continue; }
        axes.push_back(n.HashValue);
        widths.push_back(width);
    }
    if (axes.empty()) { return directBound(); }

    std::vector<std::size_t> schedule;
    schedule.reserve(static_cast<std::size_t>(depth));
    for (int level = 0; level < depth; ++level) {
        auto selected = std::size_t{0};
        for (std::size_t axis = 1; axis < axes.size(); ++axis) {
            if (widths[axis] > widths[selected]) { selected = axis; }
        }
        schedule.push_back(selected);
        widths[selected] /= Operon::Scalar{2};
    }

    std::vector<int> splits(axes.size());
    for (auto axis : schedule) { ++splits[axis]; }
    int const nLeaves = 1 << depth;
    auto const coeff = tree.GetCoefficients();

    Operon::Scalar resLo = std::numeric_limits<Operon::Scalar>::infinity();
    Operon::Scalar resHi = -std::numeric_limits<Operon::Scalar>::infinity();
    auto leafDom = dom;
    for (int k = 0; k < nLeaves; ++k) {
        std::vector<int> cells(axes.size());
        std::vector<int> bits(axes.size());
        for (std::size_t bit = 0; bit < schedule.size(); ++bit) {
            auto const axis = schedule[bit];
            cells[axis] |= ((k >> bit) & 1) << bits[axis]++;
        }
        for (std::size_t axis = 0; axis < axes.size(); ++axis) {
            auto const [lo, hi] = dom.at(axes[axis]);
            auto const step = (hi - lo) / Operon::Scalar(std::size_t{1} << splits[axis]);
            leafDom[axes[axis]] = { lo + Operon::Scalar(cells[axis]) * step, lo + Operon::Scalar(cells[axis] + 1) * step };
        }
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
    { "cancellation", "X1 * X2 - X1 * X2", { {"X1", {Operon::Scalar{0}, Operon::Scalar{10}}}, {"X2", {Operon::Scalar{0}, Operon::Scalar{10}}} } },
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
        bench.run(fmt::format("{} production depth={:02d}", ec.name, depth), [&] {
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
        double const productionNs = results[results.size() - 1].average(nb::Result::Measure::elapsed) * 1e9;
        fmt::print("{:11s}  depth={:2d}  n_leaves={:4d}  bound=[{:12.6f},{:12.6f}]  width={:12.6f}  width_vs_depth0={:7.3f}%  scalar={:11.1f}ns  production={:11.1f}ns  production_speedup={:6.2f}x\n",
            ec.name, depth, 1 << depth, static_cast<double>(simdLo), static_cast<double>(simdHi), width,
            width0 > 0 ? 100.0 * width / width0 : 100.0,
            scalarNs, productionNs, scalarNs / productionNs);
    }
}
} // namespace

// Sweeps bisection depth across several operation mixes. The scalar reference
// and production implementation enumerate identical boxes and must agree on
// their enclosure at every depth. Production batches a one-axis grid through
// wide<T>; a multi-axis grid deliberately uses scalar leaves until its
// lane-varying endpoint pattern has an independent soundness proof.
//
// Type T is fixed at build time by USE_SINGLE_PRECISION (Operon::Scalar);
// run this binary once under each precision to compare float and double.
TEST_CASE("Shape bisection depth sweep: production vs scalar, multiple expressions", "[performance][shape-constraints][bisection]")
{
    fmt::print("T = {}\n", sizeof(Operon::Scalar) == sizeof(float) ? "float" : "double");
    nb::Bench bench;
    bench.title("shape-bisection-depth").batch(1);

    for (auto const& ec : kExprCases) { RunExprCase(bench, ec); }

    bench.render(nb::templates::csv(), std::cout);
}
