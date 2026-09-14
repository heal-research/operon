// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <fmt/format.h>
#include <cstdlib>
#include <iostream>
#include <string>

#include "../operon_test.hpp"

#include "operon/core/dataset.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/constraint.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/parser/infix.hpp"

namespace nb = ankerl::nanobench;



// Sweeps OPERON_SHAPE_INTERVAL_BISECTION_DEPTH's wall-clock cost (nanobench
// CSV, batch=1 -> elapsed = seconds per Measure() call) and reports bound
// tightness for each depth, to find where deeper bisection stops paying
// for itself. Tree: (X1-1)*(X1-1) over [0,10] -- a dependency-problem case
// with a known true range ([0, 81]) and a known naive interval-overestimate
// ([-9, 81], width 90), so tightening progress is directly measurable.
TEST_CASE("Shape bisection depth sweep: wall-clock vs bound tightness", "[performance][shape-constraints][bisection]")
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

    constexpr double trueWidth = 81.0;
    constexpr double naiveWidth = 90.0;

    nb::Bench bench;
    bench.title("shape-bisection-depth").batch(1);

    fmt::print("depth  n_leaves  bound                        width      excess_over_true  pct_of_naive\n");
    for (int depth = 0; depth <= 10; ++depth) {
        shapeEval.SetBoundOptions({.BisectionDepth = depth});

        auto const nLeaves = 1 << depth;
        bench.run(fmt::format("depth={:02d}", depth), [&] {
            auto r = shapeEval.Measure(tree);
            nb::doNotOptimizeAway(r.Measurements.size());
        });

        auto const r = shapeEval.Measure(tree);
        auto const [lo, hi] = *r.Measurements[0].Bound;
        double const width = static_cast<double>(hi) - static_cast<double>(lo);
        fmt::print("{:5d}  {:8d}  [{:10.6f}, {:10.6f}]  {:9.6f}  {:15.4f}%  {:11.4f}%\n",
            depth, nLeaves, static_cast<double>(lo), static_cast<double>(hi), width,
            100.0 * (width - trueWidth) / trueWidth, 100.0 * width / naiveWidth);
    }

    bench.render(nb::templates::csv(), std::cout);
}
