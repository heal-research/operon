// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <array>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <map>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "fixed_scaling_evaluator.hpp"
#include "operon/core/constraint.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/interval_jet_evaluator.hpp"
#include "operon/interpreter/interval_second_order_jet_evaluator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/linear_scaling.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"
#include "shape_constraints_config.hpp"

namespace Operon::Test {

namespace {

    // f(X1, X2) = X1 - X2 on [1,5]x[1,5], 20 rows. Known monotonicity:
    // non-decreasing in X1, non-increasing in X2.
    struct Fixture {
        static constexpr auto Nrow { 20 };
        static constexpr auto Ncol { 3 }; // X1, X2, y

        Operon::RandomGenerator rng { 0 };
        Eigen::Array<Operon::Scalar, -1, -1> data { Nrow, Ncol };
        Operon::Dataset ds;
        Operon::Tree tree;
        using DTable = DispatchTable<Operon::Scalar>;
        DTable dtable;
        Operon::Problem problem;
        Operon::Evaluator<DTable> nmse;

        Fixture()
            : ds([&]() -> Operon::Dataset {
                for (auto i = 0; i < Ncol - 1; ++i) {
                    auto col = data.col(i);
                    std::generate(
                        col.begin(), col.end(), [&]() -> float { return Operon::Random::Uniform(rng, 1.0F, 5.0F); });
                }
                data.col(Ncol - 1) = data.col(0) - data.col(1);
                return Operon::Dataset(gsl::not_null { data.data() }, Nrow, Ncol);
            }())
            , tree(InfixParser::ParseOrThrow("X1 - X2", ds))
            , problem(&ds)
            , nmse(&problem, &dtable, Operon::NMSE {})
        {
            problem.SetTrainingRange({ 0, Nrow });
            problem.SetTestRange({ 0, Nrow });
            problem.SetTarget("X3");
        }

        static auto MakeIndividual(Operon::Tree const& t) -> Operon::Individual
        {
            Operon::Individual ind;
            ind.Genotype = t;
            return ind;
        }
    };

    auto WriteShapeConfig(std::string const& name, std::string const& text) -> std::filesystem::path
    {
        auto path = std::filesystem::temp_directory_path() / ("operon_shape_constraints_" + name + ".json");
        std::ofstream out(path);
        out << text;
        return path;
    }

    // Standalone, reporting-path view of a ShapeCertifier: Measure/Feasible/RawViolation fit the tree's own canonical
    // transform (Problem::LinearScalingOmitsNonFinite). The bound-correctness tests below ask about a tree under its
    // own fit; the gate never does this -- it passes the transform its evaluator applied.
    class CanonicalShape {
    public:
        CanonicalShape(Operon::Problem const* problem, Operon::ScalarDispatch const* dtable,
            Operon::ShapeConstraintSet constraints, Operon::Scalar unknownViolation = Operon::Scalar { 1 },
            ShapeBoundOptions bound = {})
            : certifier_(problem, dtable, std::move(constraints), unknownViolation, bound)
        {
        }
        CanonicalShape(Operon::EvaluatorBase const* evaluator, Operon::ScalarDispatch const* dtable,
            Operon::ShapeConstraintSet constraints, Operon::Scalar unknownViolation = Operon::Scalar { 1 },
            ShapeBoundOptions bound = {})
            : CanonicalShape(evaluator->GetProblem(), dtable, std::move(constraints), unknownViolation, bound)
        {
        }

        [[nodiscard]] auto Measure(Operon::Tree const& tree) const -> Operon::ShapeConstraintMeasurementSummary
        {
            return certifier_.MeasureCanonical(tree);
        }
        [[nodiscard]] auto Feasible(Operon::Tree const& tree) const -> bool { return Measure(tree).Feasible; }
        [[nodiscard]] auto RawViolation(Operon::Tree const& tree) const -> Operon::Scalar
        {
            return Measure(tree).Violation;
        }

    private:
        Operon::ShapeCertifier certifier_;
    };

} // namespace

TEST_CASE("LoadShapeConstraints parses the field-based JSON schema", "[shape-constraints]")
{
    auto const path = WriteShapeConfig("valid", R"json({
        "domains": { "X1": [1, 5.0], "X2": [1.0, 5], "x2": [-2, 2] },
        "constraints": [
            { "op": "id", "bound": [-4, 4] },
            { "op": "id", "sign": 1 },
            { "op": "derivative", "variable": "X1", "order": 1, "sign": 1 },
            { "op": "derivative", "variable": "X2", "order": 2, "bound": [0, 0] },
            { "op": "derivative", "variable": "x2", "order": 2, "sign": -1 }
        ]
    })json");

    auto result = Operon::LoadShapeConstraints(path.string());
    REQUIRE(result); // outer Cli::Result: file I/O + JSON parse + schema checks
    REQUIRE(*result); // inner std::optional: a constraint set was produced
    auto const& loaded = **result;
    REQUIRE(loaded.Domains.size() == 3);
    CHECK(loaded.Domains.at("X1").first == Catch::Approx(1.0));
    CHECK(loaded.Domains.at("X2").second == Catch::Approx(5.0));
    CHECK(loaded.Domains.at("x2").first == Catch::Approx(-2.0));

    REQUIRE(loaded.Constraints.size() == 5);
    CHECK(loaded.Constraints[0].Op == ShapeConstraintOp::Identity);
    REQUIRE(loaded.Constraints[0].Bound);
    CHECK(loaded.Constraints[0].Bound->first == Catch::Approx(-4.0));
    CHECK(loaded.Constraints[1].Op == ShapeConstraintOp::Identity);
    REQUIRE(loaded.Constraints[1].Sign);
    CHECK(*loaded.Constraints[1].Sign == 1);
    CHECK(loaded.Constraints[2].Op == ShapeConstraintOp::FirstDerivative);
    CHECK(loaded.Constraints[2].Variable == "X1");
    CHECK(loaded.Constraints[3].Op == ShapeConstraintOp::SecondDerivative);
    CHECK(loaded.Constraints[3].Variable == "X2");
    REQUIRE(loaded.Constraints[3].Bound);
    CHECK(loaded.Constraints[4].Op == ShapeConstraintOp::SecondDerivative);
    CHECK(loaded.Constraints[4].Variable == "x2"); // unambiguous variable name ending in '2'
}

TEST_CASE("LoadShapeConstraints handles empty paths and JSON schema errors", "[shape-constraints]")
{
    // An empty path means the flag was not given: a successful Result
    // carrying no constraint set, not an error.
    auto empty = Operon::LoadShapeConstraints("");
    REQUIRE(empty);
    CHECK_FALSE(*empty);

    auto const missing = std::filesystem::temp_directory_path()
        / "operon_shape_constraints_missing_file_this_test_should_not_exist.json";
    std::filesystem::remove(missing);
    auto const missingResult = Operon::LoadShapeConstraints(missing.string());
    CHECK_FALSE(missingResult); // file I/O failures no longer throw
    CHECK(missingResult.error().Code == Operon::Cli::ErrorCode::Input);

    auto const malformed
        = Operon::LoadShapeConstraints(WriteShapeConfig("malformed", R"json({"domains":)json").string());
    CHECK_FALSE(malformed); // malformed JSON no longer throws
    CHECK(malformed.error().Code == Operon::Cli::ErrorCode::Configuration);

    auto rejectsConfig = [](std::string const& name, std::string const& json) {
        auto result = Operon::LoadShapeConstraints(WriteShapeConfig(name, json).string());
        CHECK_FALSE(result); // schema violations no longer throw
        CHECK(result.error().Code == Operon::Cli::ErrorCode::Configuration);
    };

    rejectsConfig("both_sign_bound", R"json({"constraints":[{"op":"id","sign":1,"bound":[0,1]}]})json");
    rejectsConfig("neither_sign_bound", R"json({"constraints":[{"op":"id"}]})json");
    rejectsConfig("non_integral_sign", R"json({"constraints":[{"op":"id","sign":1.5}]})json");
    rejectsConfig("out_of_range_sign", R"json({"constraints":[{"op":"id","sign":0}]})json");
    rejectsConfig("bad_order", R"json({"constraints":[{"op":"derivative","variable":"X1","order":3,"sign":1}]})json");
    rejectsConfig(
        "non_integral_order", R"json({"constraints":[{"op":"derivative","variable":"X1","order":1.5,"sign":1}]})json");
    rejectsConfig("missing_variable", R"json({"constraints":[{"op":"derivative","order":1,"sign":1}]})json");
    rejectsConfig("missing_order", R"json({"constraints":[{"op":"derivative","variable":"X1","sign":1}]})json");
    rejectsConfig("bad_domain", R"json({"domains":{"X1":[0,1,2]},"constraints":[{"op":"id","sign":1}]})json");
    rejectsConfig("domains_not_object", R"json({"domains":"not an object","constraints":[{"op":"id","sign":1}]})json");
    rejectsConfig("constraints_not_array", R"json({"constraints":{}})json");
    rejectsConfig("constraint_entry_not_object", R"json({"constraints":[7]})json");
    rejectsConfig("bound_not_array", R"json({"constraints":[{"op":"id","bound":7}]})json");
    rejectsConfig("non_string_op", R"json({"constraints":[{"op":7,"sign":1}]})json");
}

TEST_CASE("ShapeCertifier - value bound constraint", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    // f = X1 - X2 over [1,5]x[1,5] has range [-4, 4]; a [-4,4] bound holds, a [-1,1] bound doesn't.
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -4 }, Operon::Scalar { 4 } } });
    CanonicalShape wide(&fx.nmse, &fx.dtable, cs);
    CHECK(wide.Feasible(fx.tree));

    cs.Constraints[0].Bound = std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } };
    CanonicalShape narrow(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(narrow.Feasible(fx.tree));
}

TEST_CASE("ParseShapeBoundOptions parses mode and mode:N", "[shape-constraints]")
{
    struct Accepted {
        char const* Text;
        ShapeBoundMode Mode;
        int Depth;
    };
    for (auto const& [text, mode, depth] : {
             Accepted { "interval", ShapeBoundMode::Interval, 0 },
             Accepted { "combined", ShapeBoundMode::Combined, 0 },
             Accepted { "affine", ShapeBoundMode::Affine, 0 },
             Accepted { "interval:0", ShapeBoundMode::Interval, 0 },
             Accepted { "interval:3", ShapeBoundMode::Interval, 3 },
             Accepted { "interval:20", ShapeBoundMode::Interval, 20 },
             Accepted { "affine:2", ShapeBoundMode::Affine, 2 },
             Accepted { "combined:4", ShapeBoundMode::Combined, 4 },
         }) {
        INFO(text);
        auto const options = Operon::ParseShapeBoundOptions(text);
        CHECK(options.Mode == mode);
        CHECK(options.BisectionDepth == depth);
    }

    for (auto const* text :
        { "bisected", "bisected:2", "bisected:", "jet", "jet:3", "jets", "interval:21", "interval:-1", "interval:x",
            "interval:", "interval:3x", "interval:+3", "interval: 3", "interval:3:4", "foo:2", "not-a-mode", "Interval",
            "interval,affine", "combined,interval:5", "affine,bisected", ":3", "" }) {
        INFO(text);
        CHECK_THROWS_AS(Operon::ParseShapeBoundOptions(text), std::invalid_argument);
    }
    // The error lists the valid forms.
    CHECK_THROWS_WITH(Operon::ParseShapeBoundOptions("bisected:2"),
        Catch::Matchers::ContainsSubstring("combined, interval or affine") && Catch::Matchers::ContainsSubstring(":N"));
}

TEST_CASE("Bisection depth is validated at the certifier boundary", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1e6 }, Operon::Scalar { 1e6 } } });

    auto const make = [&](int depth) {
        return Operon::ShapeCertifier(&fx.problem, &fx.dtable, cs, Operon::Scalar { 1 },
            Operon::ShapeBoundOptions { .Mode = ShapeBoundMode::Interval, .BisectionDepth = depth });
    };
    CHECK_THROWS_AS(make(-1), std::invalid_argument);
    CHECK_NOTHROW(make(0));
    CHECK_NOTHROW(make(20));
    CHECK_THROWS_AS(make(21), std::invalid_argument);
}

TEST_CASE("ShapeCertifier - interval:N tightens a dependency-problem bound", "[shape-constraints]")
{
    // f(X1) = (X1 - 1) * (X1 - 1) over [0, 10]: true range [0, 81], naive
    // interval multiplication overestimates to [-9, 81] (dependency
    // problem). Bisection should narrow the overestimate.
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = data(static_cast<Eigen::Index>(i), 0);
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::ParseOrThrow("(X1 - 1) * (X1 - 1)", ds);
    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(false); // compare raw tree bounds directly, no fitted scale/offset
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 10 } });
    // Permissive bound: this test compares the reported raw bound widths,
    // not feasibility, so the constraint itself must never reject.
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1000 }, Operon::Scalar { 1000 } } });

    CanonicalShape plainEval(&nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    CanonicalShape shapeEval(
        &nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 4 });

    auto const plain = plainEval.Measure(tree);
    REQUIRE(plain.Measurements.size() == 1);
    REQUIRE(plain.Measurements[0].Bound.has_value());
    auto const [plo, phi] = *plain.Measurements[0].Bound;

    auto const subdivided = shapeEval.Measure(tree);
    REQUIRE(subdivided.Measurements.size() == 1);
    REQUIRE(subdivided.Measurements[0].Bound.has_value());
    auto const [blo, bhi] = *subdivided.Measurements[0].Bound;

    // Both must soundly contain the true analytical range.
    CHECK(plo <= Operon::Scalar { 0 });
    CHECK(phi >= Operon::Scalar { 81 });
    CHECK(blo <= Operon::Scalar { 0 });
    CHECK(bhi >= Operon::Scalar { 81 });

    // Bisection is a union of sound sub-box enclosures over the same
    // domain: it can only tighten or match the direct bound, never widen.
    CHECK(blo >= plo);
    CHECK(bhi <= phi);
    // And it must actually do something on this dependency-problem
    // example, not silently no-op.
    CHECK((blo > plo || bhi < phi));
}

TEST_CASE("ShapeCertifier - interval:N accepts a model naive interval wrongly rejects", "[shape-constraints]")
{
    // Same dependency-problem tree as the tightening test above: f(X1) =
    // (X1-1)*(X1-1) over [0,10], true range [0,81]. Naive interval
    // multiplication overestimates the lower bound to -9. A constraint
    // requiring the value stay >= -1 is therefore wrongly rejected under
    // Interval alone, even though the true range [0,81] satisfies it --
    // this is the mode's actual motivating use case (a feasibility
    // decision flipping), not just a narrower reported bound width.
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = data(static_cast<Eigen::Index>(i), 0);
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::ParseOrThrow("(X1 - 1) * (X1 - 1)", ds);
    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(false);
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 10 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1000 } } });

    CanonicalShape plainEval(&nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    CHECK_FALSE(plainEval.Feasible(tree));

    CanonicalShape shapeEval(
        &nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 3 });
    CHECK(shapeEval.Feasible(tree));
}

namespace {

    // Scalar (non-SIMD) leaf-by-leaf reference for the production wide<T>-batched BisectedIntervalBound: for a single
    // referenced axis, the same widest-axis-pick + uniform-split algorithm as the wide<T>-batched path, every leaf
    // evaluated through IntervalEvaluator<Operon::Scalar> in a plain loop -- an independent cross-check of the SIMD
    // path. For more than one referenced axis, production itself evaluates scalar-only (no wide<T> arithmetic has a
    // soundness proof yet for a multi-axis endpoint pattern), so this evaluates production's own balanced schedule
    // leaf-for-leaf; this is the CI-active twin of the soundness cross-check that until now lived only
    // in test/source/performance/shape_bisection.cpp (excluded from ctest via "~[performance]").
    auto ScalarSubdividedBound(
        Operon::Tree const& tree, Operon::IntervalEvaluator<Operon::Scalar>::DomainMap const& dom, int depth)
        -> std::pair<Operon::Scalar, Operon::Scalar>
    {
        using IE = Operon::IntervalEvaluator<Operon::Scalar>;
        auto const directBound = [&]() -> std::pair<Operon::Scalar, Operon::Scalar> {
            IE ie(&tree, dom);
            auto const iv = ie(tree.GetCoefficients());
            REQUIRE(iv);
            return { iv->inf(), iv->sup() };
        };
        if (depth <= 0) {
            return directBound();
        }

        // The production schedule itself (how often each variable's domain is halved): this reference checks the
        // leaf evaluation (packed lanes vs scalar), not the choice of schedule.
        auto const planned = Operon::detail::BisectionSplits(tree, dom, depth);
        if (planned.empty()) {
            return directBound();
        }
        std::vector<Operon::Hash> axes;
        std::vector<int> splits;
        std::vector<std::size_t> schedule;
        for (auto const& [axis, count] : planned) {
            schedule.insert(schedule.end(), static_cast<std::size_t>(count), axes.size());
            axes.push_back(axis);
            splits.push_back(count);
        }
        auto const nLeaves = std::size_t { 1 } << schedule.size();
        auto const coeff = tree.GetCoefficients();

        Operon::Scalar resLo = std::numeric_limits<Operon::Scalar>::infinity();
        Operon::Scalar resHi = -std::numeric_limits<Operon::Scalar>::infinity();
        for (std::size_t k = 0; k < nLeaves; ++k) {
            auto leafDom = dom;
            std::vector<int> cells(axes.size());
            std::vector<int> bits(axes.size());
            for (std::size_t bit = 0; bit < schedule.size(); ++bit) {
                auto const axis = schedule[bit];
                cells[axis] |= (static_cast<int>((k >> bit) & std::size_t { 1 }) << bits[axis]++);
            }
            for (std::size_t axis = 0; axis < axes.size(); ++axis) {
                auto const [lo, hi] = dom.at(axes[axis]);
                auto const step = (hi - lo) / Operon::Scalar(std::size_t { 1 } << splits[axis]);
                auto const cell = Operon::Scalar(cells[axis]);
                leafDom[axes[axis]] = { lo + cell * step, lo + (cell + Operon::Scalar { 1 }) * step };
            }
            IE ie(&tree, leafDom);
            auto const seg = ie(coeff);
            REQUIRE(seg);
            resLo = std::min(resLo, seg->inf());
            resHi = std::max(resHi, seg->sup());
        }
        return { resLo, resHi };
    }

    struct BisectionExprCase {
        std::string name;
        std::string expr;
        std::vector<std::pair<std::string, std::pair<Operon::Scalar, Operon::Scalar>>> domains;
    };

    // Diverse op mix (same set as the performance sweep): pure dependency-
    // problem arithmetic, trig+pow, division, exp/log -- covers every wide<T>
    // body the SIMD bisection dispatches through, not just one tree shape.
    auto const kBisectionExprCases = std::vector<BisectionExprCase> {
        { "dependency", "(X1 - 1) * (X1 - 1)", { { "X1", { Operon::Scalar { 0 }, Operon::Scalar { 10 } } } } },
        { "trig_pow", "sin(X1) + cos(X1) * X1 ^ 2 + X2",
            { { "X1", { Operon::Scalar { -5 }, Operon::Scalar { 5 } } },
                { "X2", { Operon::Scalar { -5 }, Operon::Scalar { 5 } } } } },
        { "division", "X1 / (X2 + 3)",
            { { "X1", { Operon::Scalar { -2 }, Operon::Scalar { 2 } } },
                { "X2", { Operon::Scalar { -2 }, Operon::Scalar { 2 } } } } },
        { "exp_log", "exp(X1) - log(X2 + 1)",
            { { "X1", { Operon::Scalar { -2 }, Operon::Scalar { 2 } } },
                { "X2", { Operon::Scalar { 0.5F }, Operon::Scalar { 5 } } } } },
    };

} // namespace

TEST_CASE("Wide interval evaluator preserves packed tree lanes", "[shape-constraints][bisection]")
{
    using S = Operon::Scalar;
    using W = eve::wide<S>;
    using WI = Operon::IntervalEvaluator<W>;
    using Pack = pappus::packed_subdomains<S, W>;

    Eigen::Array<S, -1, -1> data(2, 3);
    data << S { -5 }, S { -5 }, S { 0 }, S { 5 }, S { 5 }, S { 0 };
    Operon::Dataset ds(gsl::not_null { data.data() }, 2, 3);
    auto tree = Operon::InfixParser::ParseOrThrow("sin(X1) + cos(X1) * X1 ^ 2 + X2", ds);
    auto const x1 = ds.GetVariable("X1").value().Hash;
    auto const x2 = ds.GetVariable("X2").value().Hash;
    WI::DomainMap domains { { x1, { S { -5 }, S { 5 } } }, { x2, { S { -5 }, S { 5 } } } };
    pappus::box<S> domain { pappus::interval<S>(-5, 5), pappus::interval<S>(-5, 5) };
    std::array<std::size_t, 4> schedule { 0, 1, 0, 1 };
    pappus::subdivision_plan plan(std::move(domain), schedule);
    Pack pack(plan, 0);
    Operon::Vector<WI::LaneOverride> overrides {
        { x1, pack.lower_data(0), pack.upper_data(0) },
        { x2, pack.lower_data(1), pack.upper_data(1) },
    };
    WI evaluator(&tree, domains);
    auto const packed = evaluator(tree.GetCoefficients(), overrides);
    REQUIRE(packed);

    alignas(W) std::array<S, Pack::width> lower {};
    alignas(W) std::array<S, Pack::width> upper {};
    eve::store(packed->inf(), lower.data());
    eve::store(packed->sup(), upper.data());
    for (std::size_t lane = 0; lane < pack.valid_lanes(); ++lane) {
        auto const leaf = plan.leaf(lane);
        Operon::IntervalEvaluator<S>::DomainMap scalarDomains {
            { x1, { leaf[0].inf(), leaf[0].sup() } },
            { x2, { leaf[1].inf(), leaf[1].sup() } },
        };
        Operon::IntervalEvaluator<S> scalar(&tree, scalarDomains);
        auto const expected = scalar(tree.GetCoefficients());
        REQUIRE(expected);
        CHECK(lower[lane] == Catch::Approx(expected->inf()).margin(1e-4F));
        CHECK(upper[lane] == Catch::Approx(expected->sup()).margin(1e-4F));
    }
}

// The SIMD-vs-scalar soundness cross-check from the performance sweep,
// runnable in CI: the production wide<T>-batched interval:N identity bound must match
// the scalar leaf-by-leaf reference over the same expressions and depths.
TEST_CASE("interval:N identity bound agrees with a scalar leaf-by-leaf reference", "[shape-constraints][bisection]")
{
    for (auto const& ec : kBisectionExprCases) {
        auto const nvars = ec.domains.size();
        auto const nrow = std::size_t { 5 };
        auto const ncol = nvars + 1;
        Eigen::Array<Operon::Scalar, -1, -1> data(static_cast<Eigen::Index>(nrow), static_cast<Eigen::Index>(ncol));
        for (std::size_t i = 0; i < nrow; ++i) {
            for (std::size_t v = 0; v < nvars; ++v) {
                auto const [lo, hi] = ec.domains[v].second;
                data(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(v))
                    = lo + (hi - lo) * static_cast<Operon::Scalar>(i) / static_cast<Operon::Scalar>(nrow - 1);
            }
            data(static_cast<Eigen::Index>(i), static_cast<Eigen::Index>(ncol - 1)) = Operon::Scalar { 0 };
        }
        Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
        auto tree = Operon::InfixParser::ParseOrThrow(ec.expr, ds);

        Operon::Problem problem(&ds);
        problem.SetTrainingRange({ 0, nrow });
        problem.SetTestRange({ 0, nrow });
        problem.SetTarget("X" + std::to_string(ncol));
        problem.SetLinearScalingEnabled(false);
        Fixture::DTable dtable;
        Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

        Operon::ShapeConstraintSet cs;
        Operon::IntervalEvaluator<Operon::Scalar>::DomainMap dom;
        for (auto const& [name, bound] : ec.domains) {
            cs.Domains.insert_or_assign(name, bound);
            dom.emplace(ds.GetVariable(name).value().Hash, bound);
        }
        cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
            .Variable = "",
            .Sign = std::nullopt,
            .Bound = std::pair { Operon::Scalar { -1e6 }, Operon::Scalar { 1e6 } } });

        for (int depth = 0; depth <= 6; ++depth) {
            CanonicalShape shapeEval(&nmse, &dtable, cs, Operon::Scalar { 1 },
                { .Mode = ShapeBoundMode::Interval, .BisectionDepth = depth });
            auto const r = shapeEval.Measure(tree);
            REQUIRE(r.Measurements.size() == 1);
            REQUIRE(r.Measurements[0].Bound.has_value());
            auto const [simdLo, simdHi] = *r.Measurements[0].Bound;
            auto const [scalarLo, scalarHi] = ScalarSubdividedBound(tree, dom, depth);
            INFO(ec.name << " depth=" << depth << " simd=[" << simdLo << "," << simdHi << "] scalar=[" << scalarLo
                         << "," << scalarHi << "]");
            CHECK(static_cast<double>(simdLo) == Catch::Approx(static_cast<double>(scalarLo)).margin(1e-3));
            CHECK(static_cast<double>(simdHi) == Catch::Approx(static_cast<double>(scalarHi)).margin(1e-3));
        }
    }
}

// f(X1) = X1 over [-2^mant, nextafter(2^mant)]: at this magnitude the
// diameter hi-lo is not exactly representable and rounds DOWN, so the
// last leaf's sup, computed as fl(lo + 2^depth * fl(diam/2^depth)), lands
// strictly below the domain's own sup. A round-to-nearest partition then
// under-covers the box and the reported bound excludes the domain's upper
// endpoint -- unsound for e.g. a "f <= bound" constraint sat exactly at
// hi. The bisection must keep the last leaf's sup at or above the real
// sup. Precision-generic: 2^mant is exactly representable in either float
// or double Scalar builds while the +1-ULP step and the rounded diameter
// are not.
TEST_CASE("interval:N endpoints enclose the split domain", "[shape-constraints][bisection]")
{
    using S = Operon::Scalar;
    auto const mant = std::scalbn(S { 1 }, std::numeric_limits<S>::digits); // 2^24 (float) / 2^53 (double)
    auto const lo = -mant;
    auto const hi = std::nextafter(mant, std::numeric_limits<S>::infinity());

    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::ParseOrThrow("X1", ds);

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(false);
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { lo, hi });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { -std::numeric_limits<S>::max(), std::numeric_limits<S>::max() } });

    CanonicalShape shapeEval(
        &nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 3 });

    auto const r = shapeEval.Measure(tree);
    REQUIRE(r.Measurements.size() == 1);
    REQUIRE(r.Measurements[0].Bound.has_value());
    auto const [blo, bhi] = *r.Measurements[0].Bound;
    // The union of the leaves must cover the whole input box, in
    // particular its endpoints -- the identity tree makes the reported
    // bound exactly the covered interval.
    CHECK(static_cast<double>(blo) <= static_cast<double>(lo));
    CHECK(static_cast<double>(bhi) >= static_cast<double>(hi));
}

// f(X1) = sqrt(X1) + X1 over [-2, 6]: the scalar interval sqrt clamps a
// zero-straddling box to [0, sup] (direct bound [-2, ~8.45]), but the wide
// sqrt resolves a lane whose whole sub-box lies below zero to NaN bounds
// (pappus's documented lane policy). A NaN lane is `is_empty()`, and the
// lane-wise union used by the SIMD bisection drops it silently -- the
// subdivided bound then covered only [-1, ~8.45], excluding the negative
// part of the domain the scalar mode still accounts for. A batch with an
// empty or nonfinite lane must be rejected wholesale and fall back to the
// whole-box direct bound, matching plain Interval mode exactly.
TEST_CASE(
    "interval:N falls back to the whole-box bound when a sub-box is out of domain", "[shape-constraints][bisection]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::ParseOrThrow("sqrt(X1) + X1", ds);

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(false);
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { -2 }, Operon::Scalar { 6 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1000 }, Operon::Scalar { 1000 } } });

    CanonicalShape plainEval(&nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    CanonicalShape shapeEval(
        &nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 3 });

    auto const plain = plainEval.Measure(tree);
    REQUIRE(plain.Measurements.size() == 1);
    REQUIRE(plain.Measurements[0].Bound.has_value());
    auto const [plo, phi] = *plain.Measurements[0].Bound;

    auto const subdivided = shapeEval.Measure(tree);
    REQUIRE(subdivided.Measurements.size() == 1);
    REQUIRE(subdivided.Measurements[0].Bound.has_value());
    auto const [blo, bhi] = *subdivided.Measurements[0].Bound;

    INFO("plain=[" << plo << "," << phi << "] subdivided=[" << blo << "," << bhi << "]");
    CHECK(static_cast<double>(blo) == static_cast<double>(plo));
    CHECK(static_cast<double>(bhi) == static_cast<double>(phi));
}
// f(X1) = 1 / X1 over [-2, 6] has a nonfinite direct interval because the
// denominator spans zero. SIMD bisection produces nonfinite lanes for the
// slices touching zero; those lanes must force the same direct result rather
// than being silently omitted from the lane-wise hull.
TEST_CASE("interval:N falls back to the whole-box bound when a sub-box is unbounded", "[shape-constraints][bisection]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::ParseOrThrow("1 / X1", ds);

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(false);
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { -2 }, Operon::Scalar { 6 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1000 }, Operon::Scalar { 1000 } } });

    CanonicalShape plainEval(&nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    CanonicalShape shapeEval(
        &nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 3 });
    auto const plain = plainEval.Measure(tree);
    REQUIRE(plain.Measurements.size() == 1);
    CHECK_FALSE(plain.Measurements[0].Certified);
    CHECK_FALSE(plain.Feasible);
    CHECK(plain.Violation == Catch::Approx(1.0));

    auto const subdivided = shapeEval.Measure(tree);
    REQUIRE(subdivided.Measurements.size() == 1);
    CHECK_FALSE(subdivided.Measurements[0].Certified);
    CHECK_FALSE(subdivided.Feasible);
    CHECK(subdivided.Violation == Catch::Approx(plain.Violation));
}

// User-registered interval rules are scalar-only. The bisection preflight
// must reject this tree before it creates a wide local or enters the wide
// evaluator, then return the same whole-box direct bound as plain Interval
// mode.
TEST_CASE("interval:N takes the scalar path for a user function with no wide rule", "[shape-constraints][bisection]")
{
    auto const hash = Operon::Hasher {}("subdivided_user_recipx");
    RegisterUnaryInterval<Scalar>(hash, [](IntervalEvaluator<Scalar>::Interval const& v) {
        return IntervalEvaluator<Scalar>::Interval { Scalar { 1 } } / v; // recip(x) = 1/x
    });
    using WScalar = eve::wide<Scalar>;
    RegisterIntervalBuiltins<WScalar>();
    CHECK_FALSE(IntervalUnaryRules<WScalar>().Contains(hash));

    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i) + Operon::Scalar { 1 };
        data(static_cast<Eigen::Index>(i), 1) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto varHash = ds.GetVariable("X1").value().Hash;
    Node var(NodeType::Variable, varHash);
    var.Value = Operon::Scalar { 1 };
    auto tree = Tree({ var, Node::Function(hash, 1) }).UpdateNodes();

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(false);
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 4 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -10 }, Operon::Scalar { 10 } } });

    CanonicalShape plainEval(&nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    CanonicalShape shapeEval(
        &nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 3 });

    auto const plain = plainEval.Measure(tree);
    REQUIRE(plain.Measurements.size() == 1);
    REQUIRE(plain.Measurements[0].Bound.has_value());
    auto const [plo, phi] = *plain.Measurements[0].Bound;

    auto const subdivided = shapeEval.Measure(tree);
    REQUIRE(subdivided.Measurements.size() == 1);
    REQUIRE(subdivided.Measurements[0].Bound.has_value());
    auto const [blo, bhi] = *subdivided.Measurements[0].Bound;

    // 1/[1,4] = [0.25, 1]
    CHECK(static_cast<double>(plo) == Catch::Approx(0.25).margin(1e-6));
    CHECK(static_cast<double>(phi) == Catch::Approx(1.0).margin(1e-6));
    CHECK(static_cast<double>(blo) == static_cast<double>(plo));
    CHECK(static_cast<double>(bhi) == static_cast<double>(phi));
}

// The bisection depth is fixed at construction and takes effect in the bound.
TEST_CASE("Bisection depth is applied to the interval:N bound", "[shape-constraints]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::ParseOrThrow("X1 - X1", ds); // plain interval: [-8, 8]; depth d: [-h, h], h = 8/2^d

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(false);
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 8 } });
    // Satisfied by [-0.25, 0.25] (depth 5), violated by [-1, 1] (depth 3)
    // and by the naive [-8, 8].
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair {
            Operon::Scalar { -25 } / Operon::Scalar { 100 }, Operon::Scalar { 25 } / Operon::Scalar { 100 } } });

    auto const makeSubdivided
        = [&](int depth) { return ShapeBoundOptions { .Mode = ShapeBoundMode::Interval, .BisectionDepth = depth }; };

    CanonicalShape sce3(&nmse, &dtable, cs, Operon::Scalar { 1 }, makeSubdivided(3));
    CanonicalShape sve3(&problem, &dtable, cs, Operon::Scalar { 1 }, makeSubdivided(3));
    CHECK_FALSE(sce3.Feasible(tree)); // [-1, 1] vs [-0.25, 0.25]
    CHECK(static_cast<double>(sve3.RawViolation(tree)) == Catch::Approx(1.5).margin(1e-6)); // 0.75 + 0.75

    CanonicalShape sce5(&nmse, &dtable, cs, Operon::Scalar { 1 }, makeSubdivided(5));
    CanonicalShape sve5(&problem, &dtable, cs, Operon::Scalar { 1 }, makeSubdivided(5));
    CHECK(sce5.Feasible(tree)); // [-0.25, 0.25]
    CHECK(static_cast<double>(sve5.RawViolation(tree)) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("ShapeCertifier - balanced bisection resolves multi-axis dependency", "[shape-constraints]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 3 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 2) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::ParseOrThrow("X1 * X2 - X1 * X2", ds);
    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X3");
    problem.SetLinearScalingEnabled(false);
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 10 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 10 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -24 }, Operon::Scalar { 24 } } });

    CanonicalShape shapeEval(
        &nmse, &dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 6 });
    auto const summary = shapeEval.Measure(tree);
    REQUIRE(summary.Measurements.size() == 1);
    REQUIRE(summary.Measurements[0].Bound.has_value());
    auto const [lo, hi] = *summary.Measurements[0].Bound;
    CHECK(lo <= Operon::Scalar { 0 });
    CHECK(hi >= Operon::Scalar { 0 });
    CHECK(lo >= Operon::Scalar { -24 });
    CHECK(hi <= Operon::Scalar { 24 });
    CHECK(summary.Feasible);
}

TEST_CASE("ShapeCertifier - negative linear scale flips derivative constraints", "[shape-constraints]")
{
    Fixture fx;
    auto negated = InfixParser::ParseOrThrow("X2 - X1", fx.ds);

    auto scaling = Operon::FitLinearScaling(negated, fx.problem, fx.dtable, fx.problem.TrainingRange());
    REQUIRE(scaling);
    CHECK(scaling->Scale < 0.0);
    CHECK(scaling->Offset == Catch::Approx(0.0).margin(1e-5));

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = -1, .Bound = std::nullopt });

    CanonicalShape scaled(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(scaled.Feasible(negated));

    fx.problem.SetLinearScalingEnabled(false);
    CanonicalShape raw(&fx.nmse, &fx.dtable, cs);
    CHECK(raw.Feasible(negated));
}

TEST_CASE("ShapeCertifier - negative linear scale swaps derivative bound endpoints", "[shape-constraints]")
{
    Fixture fx;
    auto negated = InfixParser::ParseOrThrow("X2 - X1", fx.ds);

    auto scaling = Operon::FitLinearScaling(negated, fx.problem, fx.dtable, fx.problem.TrainingRange());
    REQUIRE(scaling);
    CHECK(scaling->Scale < 0.0);
    CHECK(scaling->Offset == Catch::Approx(0.0).margin(1e-5));

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    // Raw d(X2-X1)/dX1 is [-1,-1], satisfying this bound.  After the fitted
    // negative scale is applied to the delivered model, the derivative interval
    // becomes [1,1] via LinearScaling::ApplyToDerivativeInterval's endpoint swap,
    // and the bound-arithmetic violation path must reject it.
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
        .Variable = "X1",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1.5F }, Operon::Scalar { -0.5F } } });

    CanonicalShape scaled(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(scaled.Feasible(negated));

    auto const measurement = scaled.Measure(negated);
    REQUIRE(measurement.Measurements.size() == 1);
    CHECK(measurement.Measurements[0].Violation > 0.0F);

    fx.problem.SetLinearScalingEnabled(false);
    CanonicalShape raw(&fx.nmse, &fx.dtable, cs);
    CHECK(raw.Feasible(negated));
}

TEST_CASE("ShapeCertifier - offset shifts identity bound constraints", "[shape-constraints]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i) / static_cast<Operon::Scalar>(nrow - 1);
        data(static_cast<Eigen::Index>(i), 1) = data(static_cast<Eigen::Index>(i), 0) + Operon::Scalar { 10 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::ParseOrThrow("X1", ds);
    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, nrow });
    problem.SetTestRange({ 0, nrow });
    problem.SetTarget("X2");
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE {});

    auto scaling = Operon::FitLinearScaling(tree, problem, dtable, problem.TrainingRange());
    REQUIRE(scaling);
    CHECK(scaling->Scale == Catch::Approx(1.0));
    CHECK(scaling->Offset == Catch::Approx(10.0));

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } } });

    CanonicalShape scaled(&nmse, &dtable, cs);
    CHECK_FALSE(scaled.Feasible(tree));

    problem.SetLinearScalingEnabled(false);
    CanonicalShape raw(&nmse, &dtable, cs);
    CHECK(raw.Feasible(tree));
}

TEST_CASE("ShapeCertifier reports directly fitted scaled bounds under the canonical transform", "[shape-constraints]")
{
    Fixture fx;
    auto tree = InfixParser::ParseOrThrow("2 * (X1 - X2) + 3", fx.ds);
    auto scaling = Operon::FitLinearScaling(tree, fx.problem, fx.dtable, fx.problem.TrainingRange());
    REQUIRE(scaling);

    // The public API does not expose BoundFor directly. For this affine tree/domain
    // we can reconstruct the raw identity enclosure exactly by hand: [2*(-4)+3, 2*4+3].
    auto const [expectedLo, expectedHi] = scaling->ApplyToValueInterval(Operon::Scalar { -5 }, Operon::Scalar { 11 });

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -100 }, Operon::Scalar { 100 } } });

    CanonicalShape sce(&fx.nmse, &fx.dtable, cs);
    auto const constrained = sce.Measure(tree);
    REQUIRE(constrained.Measurements.size() == 1);
    REQUIRE(constrained.Measurements[0].Bound);
    CHECK(constrained.Measurements[0].Bound->first == Catch::Approx(expectedLo));
    CHECK(constrained.Measurements[0].Bound->second == Catch::Approx(expectedHi));

    CanonicalShape sve(&fx.problem, &fx.dtable, cs);
    auto const violation = sve.Measure(tree);
    REQUIRE(violation.Measurements.size() == 1);
    REQUIRE(violation.Measurements[0].Bound);
    CHECK(violation.Measurements[0].Bound->first == Catch::Approx(expectedLo));
    CHECK(violation.Measurements[0].Bound->second == Catch::Approx(expectedHi));
}

TEST_CASE("ShapeCertifier - derivative through an unsupported op is not falsely feasible", "[shape-constraints]")
{
    // f(x) = abs(X1): Abs has no registered symbolic derivative rule
    // (tree_diff.cpp intentionally leaves it unregistered, non-smooth).
    // A sign constraint on d/dX1 must NOT be certified feasible just
    // because Deriv() falls back to "zero" for the unsupported op --
    // that fallback is sound for LM coefficient fitting but not for a
    // feasibility certificate (a real, previously-latent soundness gap).
    Fixture fx;
    auto tree = InfixParser::ParseOrThrow("abs(X1)", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    CanonicalShape sce(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(sce.Feasible(tree));
}

TEST_CASE("ShapeCertifier - identically-zero derivative satisfies either sign", "[shape-constraints]")
{
    // f(X2) = X2 does not reference X1 at all -> d/dX1 is identically
    // zero (BuildVariableGradientDag's NoGrad root path, no
    // AffineEvaluator call needed for it), which must satisfy a
    // non-decreasing (0 >= 0) or non-increasing (0 <= 0) constraint.
    Fixture fx;
    auto tree = InfixParser::ParseOrThrow("X2", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    CanonicalShape nonDecreasing(&fx.nmse, &fx.dtable, cs);
    CHECK(nonDecreasing.Feasible(tree));

    cs.Constraints[0].Sign = -1;
    CanonicalShape nonIncreasing(&fx.nmse, &fx.dtable, cs);
    CHECK(nonIncreasing.Feasible(tree));
}

TEST_CASE("IntervalEvaluator returns multiple VariableGradientDag roots from one traversal", "[shape-constraints]")
{
    Fixture fx;
    auto tree = InfixParser::ParseOrThrow("X1 * X1 * X2", fx.ds);
    auto const x1 = fx.ds.GetVariable("X1").value().Hash;
    auto const x2 = fx.ds.GetVariable("X2").value().Hash;
    auto const coeff = tree.GetCoefficients();
    auto const dag = BuildVariableGradientDag(tree, coeff);
    auto const x1Index = std::ranges::find(dag.Variables, x1);
    auto const x2Index = std::ranges::find(dag.Variables, x2);
    REQUIRE(x1Index != dag.Variables.end());
    REQUIRE(x2Index != dag.Variables.end());

    Operon::Vector<std::size_t> roots { dag.OriginalSize - 1,
        dag.Roots[static_cast<std::size_t>(std::distance(dag.Variables.begin(), x1Index))],
        dag.Roots[static_cast<std::size_t>(std::distance(dag.Variables.begin(), x2Index))] };
    REQUIRE(
        std::ranges::none_of(roots, [](std::size_t root) { return root == std::numeric_limits<std::size_t>::max(); }));

    IntervalEvaluator<Scalar>::DomainMap domains;
    domains.insert_or_assign(x1, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    domains.insert_or_assign(x2, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    IntervalEvaluator<Scalar> fused(&tree, domains);
    fused.SetNodes(dag.Nodes);
    Operon::Vector<IntervalEvaluator<Scalar>::Interval> fusedBounds(roots.size());
    auto const evaluation
        = fused.EvaluateAll(coeff, [&](Operon::Span<IntervalEvaluator<Scalar>::Interval const> values) {
              for (std::size_t i = 0; i < roots.size(); ++i) {
                  fusedBounds[i] = values[roots[i]];
              }
          });
    REQUIRE(evaluation);

    for (std::size_t i = 0; i < roots.size(); ++i) {
        Tree sliced(
            Operon::Vector<Node>(dag.Nodes.begin(), dag.Nodes.begin() + static_cast<std::ptrdiff_t>(roots[i]) + 1));
        sliced.UpdateNodes();
        IntervalEvaluator<Scalar> individual(&sliced, domains);
        auto const individualBound = individual(sliced.GetCoefficients());
        REQUIRE(individualBound);
        CHECK(fusedBounds[i].inf() == Catch::Approx(individualBound->inf()));
        CHECK(fusedBounds[i].sup() == Catch::Approx(individualBound->sup()));
    }
}

TEST_CASE("BuildVariableGradientDag filters unreferenced variables", "[shape-constraints]")
{
    Fixture fx;
    auto tree = InfixParser::ParseOrThrow("X1 * X1 * X2", fx.ds);
    auto const x1 = fx.ds.GetVariable("X1").value().Hash;
    auto const x2 = fx.ds.GetVariable("X2").value().Hash;
    auto const coeff = tree.GetCoefficients();
    auto const full = BuildVariableGradientDag(tree, coeff);
    Operon::Vector<Operon::Hash> filter { x1 };
    auto const filtered = BuildVariableGradientDag(tree, coeff, filter);

    auto const fullX1 = std::ranges::find(full.Variables, x1);
    auto const fullX2 = std::ranges::find(full.Variables, x2);
    REQUIRE(fullX1 != full.Variables.end());
    REQUIRE(fullX2 != full.Variables.end());
    REQUIRE(filtered.Variables.size() == 1);
    CHECK(filtered.Variables[0] == x1);
    auto const fullIndex = static_cast<std::size_t>(std::distance(full.Variables.begin(), fullX1));
    CHECK(filtered.Certain[0] == full.Certain[fullIndex]);
    Tree fullDerivative(Operon::Vector<Node>(
        full.Nodes.begin(), full.Nodes.begin() + static_cast<std::ptrdiff_t>(full.Roots[fullIndex]) + 1));
    Tree filteredDerivative(Operon::Vector<Node>(
        filtered.Nodes.begin(), filtered.Nodes.begin() + static_cast<std::ptrdiff_t>(filtered.Roots[0]) + 1));
    fullDerivative.UpdateNodes();
    filteredDerivative.UpdateNodes();

    IntervalEvaluator<Scalar>::DomainMap domains;
    domains.insert_or_assign(x1, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    domains.insert_or_assign(x2, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    IntervalEvaluator<Scalar> fullEvaluator(&fullDerivative, domains);
    IntervalEvaluator<Scalar> filteredEvaluator(&filteredDerivative, domains);
    auto const fullBound = fullEvaluator(fullDerivative.GetCoefficients());
    auto const filteredBound = filteredEvaluator(filteredDerivative.GetCoefficients());

    REQUIRE(fullBound);
    REQUIRE(filteredBound);
    CHECK(filteredBound->inf() == Catch::Approx(fullBound->inf()));
    CHECK(filteredBound->sup() == Catch::Approx(fullBound->sup()));
}

TEST_CASE("ShapeCertifier certifies a single first-derivative constraint", "[shape-constraints]")
{
    Fixture fx;
    auto tree = InfixParser::ParseOrThrow("X1 * X1 * X2", fx.ds);
    fx.problem.SetLinearScalingEnabled(false);

    Operon::ShapeConstraintSet constraints;
    constraints.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    constraints.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    constraints.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
        .Variable = "X1",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { 2 }, Operon::Scalar { 50 } } });

    CanonicalShape penalty(&fx.problem, &fx.dtable, constraints);
    auto const measurement = penalty.Measure(tree);

    REQUIRE(measurement.Measurements.size() == 1);
    REQUIRE(measurement.Measurements[0].Certified);
    REQUIRE(measurement.Measurements[0].Bound);
    CHECK(measurement.Measurements[0].Bound->first == Catch::Approx(2));
    CHECK(measurement.Measurements[0].Bound->second == Catch::Approx(50));
    CHECK(measurement.Violation == Catch::Approx(0));
}

TEST_CASE("ShapeCertifier handles a fused single-derivative failure as unknown", "[shape-constraints]")
{
    Fixture fx;
    auto tree = InfixParser::ParseOrThrow("1 / X1", fx.ds);
    fx.problem.SetLinearScalingEnabled(false);

    Operon::ShapeConstraintSet constraints;
    constraints.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } });
    constraints.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    constraints.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
        .Variable = "X1",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -100 }, Operon::Scalar { 100 } } });

    CanonicalShape penalty(
        &fx.problem, &fx.dtable, constraints, Operon::Scalar { 2.5 }, { .Mode = ShapeBoundMode::Interval });
    auto const measurement = penalty.Measure(tree);

    REQUIRE(measurement.Measurements.size() == 1);
    CHECK_FALSE(measurement.Measurements[0].Certified);
    CHECK(measurement.Measurements[0].Violation == Catch::Approx(2.5));
    CHECK_FALSE(measurement.Feasible);
}

TEST_CASE("ShapeCertifier - unknown variable in domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("NotAColumn", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } });
    CHECK_THROWS_AS(CanonicalShape(&fx.nmse, &fx.dtable, cs), std::invalid_argument);
}

TEST_CASE("ShapeCertifier - constraint variable missing from domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    // No domains entry for X1 at all.
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });
    CHECK_THROWS_AS(CanonicalShape(&fx.nmse, &fx.dtable, cs), std::invalid_argument);
}

TEST_CASE("ShapeCertifier - problem input variable missing from domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -4 }, Operon::Scalar { 4 } } });

    CHECK_THROWS_WITH(CanonicalShape(&fx.nmse, &fx.dtable, cs), Catch::Matchers::ContainsSubstring("X2"));
    CHECK_THROWS_WITH(CanonicalShape(&fx.problem, &fx.dtable, cs), Catch::Matchers::ContainsSubstring("X2"));
}

TEST_CASE("ShapeCertifier - domain error (e.g. division by zero-containing interval) is treated as "
          "infeasible, not a crash",
    "[shape-constraints]")
{
    // f(x) = 1 / X1, with X1's domain spanning zero -- AffineEvaluator
    // throws std::invalid_argument for this (affine_form::inv is
    // undefined over an interval containing zero); GP generates trees
    // like this constantly, so Feasible() must swallow it as "can't be
    // certified feasible" rather than letting a run crash.
    Fixture fx;
    auto tree = InfixParser::ParseOrThrow("1 / X1", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -100 }, Operon::Scalar { 100 } } });

    CanonicalShape sce(&fx.nmse, &fx.dtable, cs);
    auto const measurement = sce.Measure(tree);
    REQUIRE(measurement.Measurements.size() == 1);
    CHECK_FALSE(measurement.Measurements[0].Certified);
    CHECK(measurement.Measurements[0].Violation == Catch::Approx(1.0));
    CHECK_FALSE(sce.Feasible(tree));
}

TEST_CASE(
    "ShapeCertifier - a throwing user-registered rule is treated as infeasible, not a crash", "[shape-constraints]")
{
    // A throwing user-registered rule must degrade to an uncertified
    // bound (see the file-top comment in shape_constrained_evaluator.cpp),
    // never escape Measure()/Feasible() into a GP worker thread.
    //
    // Each section registers under its own hash: the registries are
    // process-wide and write-once, and other tests in this binary
    // register user rules of their own.
    Fixture fx;
    fx.problem.SetLinearScalingEnabled(false); // the custom op has no numeric dispatch entry
    auto const x1 = fx.ds.GetVariable("X1").value().Hash;

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -100 }, Operon::Scalar { 100 } } });

    auto const makeTree = [&](Operon::Hash hash) {
        Node var(NodeType::Variable, x1);
        var.Value = Operon::Scalar { 1 };
        return Tree({ var, Node::Function(hash, 1) }).UpdateNodes();
    };

    auto const assertDegradesToUncertified = [&](CanonicalShape const& sce, Operon::Tree const& tree) {
        Operon::ShapeConstraintMeasurementSummary measurement;
        REQUIRE_NOTHROW(measurement = sce.Measure(tree));
        REQUIRE(measurement.Measurements.size() == 1);
        CHECK_FALSE(measurement.Measurements[0].Certified);
        CHECK(measurement.Measurements[0].Violation == Catch::Approx(1.0)); // unknownViolation default
        CHECK_FALSE(measurement.Feasible);
        bool feasible = true;
        REQUIRE_NOTHROW(feasible = sce.Feasible(tree));
        CHECK_FALSE(feasible);
    };

    SECTION("affine rule returning tl::unexpected, combined mode: degrades to uncertified")
    {
        auto const hash = Operon::Hasher {}("shape_throw_affine_rule");
        bool invoked = false;
        RegisterUnaryAffine<Scalar>(hash,
            [&invoked](AffineEvaluator<Scalar>::Context const&,
                AffineEvaluator<Scalar>::Affine const&) -> tl::expected<AffineEvaluator<Scalar>::Affine, std::string> {
                invoked = true;
                return tl::unexpected("user affine rule failed");
            });
        CanonicalShape sce(&fx.nmse, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Combined });
        assertDegradesToUncertified(sce, makeTree(hash));
        CHECK(
            invoked); // proves the callback's tl::unexpected genuinely propagated, not a missing-registry short-circuit
    }

    SECTION("interval rule returning tl::unexpected, combined mode: degrades to uncertified")
    {
        auto const hash = Operon::Hasher {}("shape_throw_interval_rule_combined");
        bool invoked = false;
        RegisterUnaryInterval<Scalar>(hash,
            [&invoked](IntervalEvaluator<Scalar>::Interval const&)
                -> tl::expected<IntervalEvaluator<Scalar>::Interval, std::string> {
                invoked = true;
                return tl::unexpected("user interval rule failed");
            });
        CanonicalShape sce(&fx.nmse, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Combined });
        assertDegradesToUncertified(sce, makeTree(hash));
        CHECK(invoked);
    }

    SECTION("interval rule returning tl::unexpected, interval mode: degrades to uncertified")
    {
        auto const hash = Operon::Hasher {}("shape_throw_interval_rule_interval_mode");
        bool invoked = false;
        RegisterUnaryInterval<Scalar>(hash,
            [&invoked](IntervalEvaluator<Scalar>::Interval const&)
                -> tl::expected<IntervalEvaluator<Scalar>::Interval, std::string> {
                invoked = true;
                return tl::unexpected("user interval rule failed");
            });
        CanonicalShape sce(&fx.nmse, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
        assertDegradesToUncertified(sce, makeTree(hash));
        CHECK(invoked);
    }
}

TEST_CASE("ShapeCertifier - certifies constant integer powers of a negative base", "[shape-constraints]")
{
    // A fully constant subtree (degenerate base AND exponent) dispatches
    // AffineEvaluator's Pow rule to pow(ctx, base, Scalar exponent), whose
    // terms_.empty() branch allows a negative base for an integer exponent
    // via plain std::pow -- unlike the general affine-affine overload, which
    // unconditionally rejects any negative base. IntervalEvaluator handles
    // the same case directly and would also certify this if affine somehow
    // regressed back to rejecting it.
    Fixture fx;
    // This test's whole point is the raw (-0.91)^2 bound computation, not
    // FitLinearScaling's fit against Fixture's data -- and that data comes
    // from std::uniform_real_distribution, whose output sequence for a
    // given seed the C++ standard does not guarantee portable across
    // library implementations (confirmed via a diagnostic-instrumented CI
    // run, 2026-08-21: identical std::pow(-0.91, 2) on every platform, but
    // a materially different fitted Scale/Offset on macOS vs Linux, flipping
    // Feasible). Skip the fit so this checks the one thing it's meant to.
    fx.problem.SetLinearScalingEnabled(false);
    auto tree = InfixParser::ParseOrThrow("(-0.91) ^ 2", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } } });

    CanonicalShape sce(&fx.nmse, &fx.dtable, cs);
    auto const measurement = sce.Measure(tree);
    REQUIRE(measurement.Measurements.size() == 1);
    CHECK(measurement.Measurements[0].Certified);
    REQUIRE(measurement.Measurements[0].Bound);
    CHECK(measurement.Measurements[0].Bound->first == Catch::Approx(0.8281).margin(1e-3));
    CHECK(measurement.Measurements[0].Bound->second == Catch::Approx(0.8281).margin(1e-3));
    CHECK(measurement.Feasible);
    CHECK(sce.Feasible(tree));
}

TEST_CASE("ShapeCertifier - a NaN bound endpoint (Scale==0 times an unbounded raw interval) is not falsely "
          "feasible",
    "[shape-constraints]")
{
    // Regression test: TransformBound multiplies the raw tree's affine
    // interval by the fitted Scale. If the target column is constant, OLS
    // gives Scale == 0 exactly (covariance with a constant is 0); if the
    // raw tree's own affine interval also overflows to +-inf somewhere in
    // the box (exp() over a wide-enough domain, no exception -- unlike the
    // zero-containing-interval division case above, which throws), the
    // product is 0 * inf == NaN. std::max(0, NaN) in ConstraintViolation
    // returns 0 (comparisons against NaN are always false), which used to
    // certify this as a zero-violation, feasible tree instead of flagging
    // it as uncertified.
    constexpr auto Nrow = 20;
    Eigen::Array<Operon::Scalar, -1, -1> data(Nrow, 2); // X1, y
    for (auto i = 0; i < Nrow; ++i) {
        data(i, 0) = static_cast<Operon::Scalar>(i);
    } // 0..19 -- training points stay finite under exp(); the domain box below is what overflows
    data.col(1).setConstant(Operon::Scalar { 5 }); // constant target -> covariance(*, y) == 0
    Operon::Dataset ds(gsl::not_null { data.data() }, Nrow, 2);

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, Nrow });
    problem.SetTestRange({ 0, Nrow });
    problem.SetTarget("X2");
    problem.SetLinearScalingEnabled(true);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    Operon::Evaluator<DTable> nmse(&problem, &dtable, Operon::NMSE {});

    auto tree = InfixParser::ParseOrThrow("exp(X1)", ds);
    auto const scaling = Operon::FitLinearScaling(tree, problem, dtable, problem.TrainingRange());
    REQUIRE(scaling.has_value());
    CHECK(scaling->Scale == Operon::Scalar { 0 }); // exact zero, not a fallback

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign(
        "X1", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1000 } }); // exp(1000) overflows double to +inf
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } } });

    CanonicalShape sce(&nmse, &dtable, cs);
    CHECK_FALSE(sce.Feasible(tree));
}

TEST_CASE("ShapeConstraintPolicy validation covers GP and NSGA2 mode rules", "[shape-constraints]")
{
    using E = Operon::ShapeConstraintEnforcement;

    auto valid = [](E enforcement, bool isNsga2) {
        CHECK_FALSE(Operon::ValidatePolicy({ .Enforcement = enforcement,
                                               .UnknownViolation = Operon::Scalar { 1 },
                                               .PenaltyWeight = Operon::Scalar { 1 } },
            isNsga2));
    };
    auto invalid = [](E enforcement, bool isNsga2) {
        CHECK(Operon::ValidatePolicy({ .Enforcement = enforcement,
                                         .UnknownViolation = Operon::Scalar { 1 },
                                         .PenaltyWeight = Operon::Scalar { 1 } },
            isNsga2));
    };

    SECTION("GP accepts only hard-reject and penalty")
    {
        valid(E::HardReject, false);
        valid(E::Penalty, false);
        invalid(E::ExtraObjective, false);
        invalid(E::PenaltyAndExtraObjective, false);
    }

    SECTION("NSGA2 accepts hard-reject, penalty, extra-objective, and penalty plus extra-objective")
    {
        valid(E::HardReject, true);
        valid(E::Penalty, true);
        valid(E::ExtraObjective, true);
        valid(E::PenaltyAndExtraObjective, true);
    }

    CHECK(Operon::ValidatePolicy({ .Enforcement = E::HardReject,
                                     .UnknownViolation = Operon::Scalar { -1 },
                                     .PenaltyWeight = Operon::Scalar { 1 } },
        false));
    CHECK(Operon::ValidatePolicy({ .Enforcement = E::HardReject,
                                     .UnknownViolation = Operon::Scalar { 1 },
                                     .PenaltyWeight = Operon::Scalar { -1 } },
        false));
    CHECK(Operon::ValidatePolicy({ .Enforcement = E::HardReject,
                                     .UnknownViolation = std::numeric_limits<Operon::Scalar>::quiet_NaN(),
                                     .PenaltyWeight = Operon::Scalar { 1 } },
        false));
    CHECK(Operon::ValidatePolicy({ .Enforcement = E::HardReject,
                                     .UnknownViolation = Operon::Scalar { 1 },
                                     .PenaltyWeight = std::numeric_limits<Operon::Scalar>::quiet_NaN() },
        false));
}

TEST_CASE("ParseShapeEnforcement parses CLI enforcement tokens", "[shape-constraints]")
{
    using E = Operon::ShapeConstraintEnforcement;

    CHECK(Operon::ParseShapeEnforcement("hard-reject") == E::HardReject);
    CHECK(Operon::ParseShapeEnforcement("penalty") == E::Penalty);
    CHECK(Operon::ParseShapeEnforcement("extra-objective") == E::ExtraObjective);
    CHECK(Operon::ParseShapeEnforcement("penalty,extra-objective") == E::PenaltyAndExtraObjective);
    CHECK(Operon::ParseShapeEnforcement("extra-objective,penalty") == E::PenaltyAndExtraObjective);

    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("hard-reject,penalty"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("hard-reject,hard-reject"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("penalty,penalty"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("feasibility-first"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("penalty,feasibility-first"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement(""), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("penalty,"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("unknown"), std::invalid_argument);
}

TEST_CASE("ShapeCertifier - sign constraint violation magnitudes", "[shape-constraints]")
{
    Fixture fx;
    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    auto bad = InfixParser::ParseOrThrow("X2 - X1", fx.ds);
    CanonicalShape sve(&fx.problem, &fx.dtable, cs);
    auto m = sve.Measure(bad);
    REQUIRE(m.Measurements.size() == 1);
    CHECK_FALSE(m.Feasible);
    CHECK(m.Measurements[0].Certified);
    CHECK(m.Measurements[0].Violation == Catch::Approx(1.0));
    CHECK(sve.RawViolation(bad) == Catch::Approx(1.0));

    cs.Constraints[0].Sign = -1;
    CanonicalShape mirror(&fx.problem, &fx.dtable, cs);
    CHECK(mirror.RawViolation(fx.tree) == Catch::Approx(1.0));
}

TEST_CASE("ShapeCertifier - bound constraint violation magnitude", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } } });

    CanonicalShape sve(&fx.problem, &fx.dtable, cs);
    auto m = sve.Measure(fx.tree);
    REQUIRE(m.Measurements.size() == 1);
    REQUIRE(m.Measurements[0].Bound);
    CHECK(m.Measurements[0].Bound->first == Catch::Approx(-4.0));
    CHECK(m.Measurements[0].Bound->second == Catch::Approx(4.0));
    CHECK(m.Violation == Catch::Approx(6.0));
    CHECK_FALSE(m.Feasible);
}

TEST_CASE("ShapeCertifier - identity, first-derivative, and second-derivative measurements", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -4 }, Operon::Scalar { 4 } } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
        .Variable = "X1",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { 1 }, Operon::Scalar { 1 } } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::SecondDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    CanonicalShape sve(&fx.problem, &fx.dtable, cs);
    auto m = sve.Measure(fx.tree);
    REQUIRE(m.Measurements.size() == 3);
    CHECK(m.Feasible);
    CHECK(m.Violation == Catch::Approx(0.0));
    CHECK(m.Measurements[0].Certified);
    CHECK(m.Measurements[1].Certified);
    CHECK(m.Measurements[2].Certified);

    auto square = InfixParser::ParseOrThrow("X1 * X1", fx.ds);
    Operon::ShapeConstraintSet secondDerivativeOnly;
    secondDerivativeOnly.Domains = cs.Domains;
    secondDerivativeOnly.Constraints.push_back(
        { .Op = ShapeConstraintOp::SecondDerivative, .Variable = "X1", .Sign = -1, .Bound = std::nullopt });
    fx.problem.SetLinearScalingEnabled(false);
    CanonicalShape violated(&fx.problem, &fx.dtable, secondDerivativeOnly);
    auto v = violated.Measure(square);
    CHECK_FALSE(v.Feasible);
    CHECK(v.Violation == Catch::Approx(2.0));
}

TEST_CASE("ShapeCertifier - unknown violation and empty constraint set", "[shape-constraints]")
{
    Fixture fx;
    auto unknownTree = InfixParser::ParseOrThrow("abs(X1)", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    CanonicalShape sve(&fx.problem, &fx.dtable, cs, Operon::Scalar { 2.5 });
    auto m = sve.Measure(unknownTree);
    REQUIRE(m.Measurements.size() == 1);
    CHECK_FALSE(m.Measurements[0].Certified);
    CHECK_FALSE(m.Feasible);
    CHECK(m.Violation == Catch::Approx(2.5));

    cs.Constraints.clear();
    CanonicalShape empty(&fx.problem, &fx.dtable, cs);
    auto e = empty.Measure(fx.tree);
    CHECK(e.Feasible);
    CHECK(e.Measurements.empty());
    CHECK(e.Violation == Catch::Approx(0.0));
    CHECK(empty.RawViolation(fx.tree) == Catch::Approx(0.0));
}

TEST_CASE("ShapeCertifier - constructor rejects malformed constraints", "[shape-constraints]")
{
    Fixture fx;
    auto withOneConstraint = [](Operon::ShapeConstraint c) {
        Operon::ShapeConstraintSet cs;
        cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
        cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
        cs.Constraints.push_back(std::move(c));
        return cs;
    };

    // neither Sign nor Bound set
    CHECK_THROWS_AS(
        CanonicalShape(&fx.nmse, &fx.dtable,
            withOneConstraint(
                { .Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::nullopt })),
        std::invalid_argument);
    // both set
    CHECK_THROWS_AS(CanonicalShape(&fx.nmse, &fx.dtable,
                        withOneConstraint({ .Op = ShapeConstraintOp::Identity,
                            .Variable = "",
                            .Sign = 1,
                            .Bound = std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } } })),
        std::invalid_argument);
    // invalid sign
    CHECK_THROWS_AS(
        CanonicalShape(&fx.nmse, &fx.dtable,
            withOneConstraint({ .Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = 2, .Bound = std::nullopt })),
        std::invalid_argument);
    // lo > hi
    CHECK_THROWS_AS(CanonicalShape(&fx.nmse, &fx.dtable,
                        withOneConstraint({ .Op = ShapeConstraintOp::Identity,
                            .Variable = "",
                            .Sign = std::nullopt,
                            .Bound = std::pair { Operon::Scalar { 5 }, Operon::Scalar { 1 } } })),
        std::invalid_argument);
}

TEST_CASE("SCRATCH pappus-fix false-feasibility repro", "[.][shape-constraints-scratch]")
{
    auto runCase = [](std::string const& label, std::string const& csvPath, std::string const& target,
                       std::string const& constraintsJson, std::string const& model) {
        Operon::Dataset ds(csvPath, /*hasHeader=*/true);
        Operon::Problem problem(&ds);
        problem.SetTrainingRange({ 0, static_cast<std::size_t>(ds.Rows()) });
        problem.SetTestRange({ 0, static_cast<std::size_t>(ds.Rows()) });
        problem.SetTarget(target);

        auto path = WriteShapeConfig(label, constraintsJson);
        auto result = Operon::LoadShapeConstraints(path.string());
        REQUIRE(result); // outer Cli::Result
        REQUIRE(*result); // inner std::optional
        auto const& loaded = **result;

        using DTable = DispatchTable<Operon::Scalar>;
        DTable dtable;
        Operon::Evaluator<DTable> nmse(&problem, &dtable, Operon::NMSE {});

        auto tree = InfixParser::ParseOrThrow(model, ds);
        CanonicalShape sce(&nmse, &dtable, loaded);
        auto feasible = sce.Feasible(tree);
        WARN(label << " Feasible()=" << feasible << " (expected false -- independent check found a real violation)");
        CHECK_FALSE(feasible);
    };

    auto const base = std::string { "/home/bogdb/src/operon-workspace/operon-publications/experiments/"
                                    "shape-constraints-reproduction/results/full_run/" };
    auto const flowStressConstraints
        = R"json({"domains": {"T": [350, 510], "phi": [0, 0.7], "phi_dot": [0.001, 10]}, "constraints": [{"op": "id", "bound": [0, 200]}, {"op": "derivative", "variable": "T", "order": 1, "sign": -1}, {"op": "derivative", "variable": "phi_dot", "order": 1, "sign": 1}, {"op": "derivative", "variable": "phi", "order": 2, "sign": -1}]})json";
    auto const carsConstraints
        = R"json({"domains": {"cylinders": [3, 8], "displacement": [68, 455], "horsepower": [46, 230], "weight": [1613, 5140], "acceleration": [8, 23.5]}, "constraints": [{"op": "id", "bound": [9.0, 46.6]}, {"op": "derivative", "variable": "displacement", "order": 1, "sign": -1}, {"op": "derivative", "variable": "horsepower", "order": 1, "sign": -1}, {"op": "derivative", "variable": "weight", "order": 1, "sign": -1}]})json";

    SECTION("Case A")
    {
        runCase("caseA", base + "Flow_stress_data.csv", "kf", flowStressConstraints,
            "((-1656471.375000) + (3.006961 * (((2.070481 * phi) + ((((((-0.107675) * T) + (2.079886 + (1.591433 * "
            "phi_dot))) + (((2.379283 * phi_dot) + (sin(((1.591433 * phi_dot) / 1.596721)) + (exp(exp(2.646259)) + "
            "(sin((1.591433 * phi_dot)) + (log((1.155026 + (1.591433 * phi_dot))) + (((-0.133379) * T) + log((1.155026 "
            "+ (1.591433 * phi_dot))))))))) + ((sin((1.766540 * phi_dot)) + exp(2.313587)) / (exp(1.995921) ^ 2)))) * "
            "0.065602) / 1.268149)) * exp(2.079886))))");
    }
    SECTION("Case B")
    {
        runCase("caseB", base + "Flow_stress_data.csv", "kf", flowStressConstraints,
            "(175.725388 + (0.000001 * (((((1.661356 * phi_dot) ^ 2) ^ 2) + ((((((cos(((3.070189 * phi_dot) / "
            "(-3.244338))) / 0.146962) ^ 2) ^ 2) / (-2.819170)) + (((((((2.715578 * phi_dot) ^ 2) + (((3.070189 * "
            "phi_dot) / (-3.244338)) / 0.035275)) + ((-0.302006) * T)) + (((1.920551 * phi_dot) ^ 2) + ((((((3.367852 "
            "* phi_dot) / (-3.460277)) / (-0.037834)) + ((-1.131171) * T)) / (-0.037834)) + ((1.920551 * phi_dot) ^ "
            "2)))) + ((-1.244043) * T)) / (-0.037834))) / (-0.037834))) / (-0.037834))))");
    }
    SECTION("Case C")
    {
        runCase("caseC", base + "Flow_stress_data.csv", "kf", flowStressConstraints,
            "((-0.541280) + (0.997766 * (((((((((((((((((-0.000020) * phi) + ((-0.000002) * phi_dot)) + (-1.965610)) + "
            "1.881476) * 2041.673462) + ((-1.431116) * (-119.989891))) + (0.000224 * T)) / 168.553055) * "
            "((-145.032639) + ((1.275833 * phi_dot) + (0.093371 * T)))) + (-1.971097)) + 1.879035) * 2041.673462) + "
            "((-1.337756) * (-112.162704))) + ((-0.049931) * T)) + 159.525269) + (-6.337012))))");
    }
    SECTION("Case D")
    {
        runCase("caseD", base + "Cars_data.csv", "mpg", carsConstraints,
            "(13634192.000000 + ((-0.001787) * ((((2.305640 * displacement) + ((1.891310 * weight) + "
            "((exp(exp(1.582666)) * ((1.250606 * cylinders) + exp(exp(2.884111)))) + (0.360743 * displacement)))) + "
            "((0.781982 * cylinders) + (exp(0.058231) + ((((-0.539599) * cylinders) + ((0.506669 * displacement) + "
            "((0.506669 * displacement) + 0.303519))) + (exp(exp(1.582666)) + (exp(0.303519) + (exp(1.582666) * "
            "0.058231))))))) + (((1.525301 * weight) * exp((0.058231 + exp(0.303519)))) * 0.058231))))");
    }
}

// Reproduces the 2026-08-08/09 Operon-vs-HL bound cross-comparison finding
// (operon-publications shape-constraints-reproduction: 1506/2796 jointly-
// certified derivative bounds disagreed in SIGN between Operon's affine
// bound and HL's plain interval bound, on the same tree/variable/domain
// box). This individual (from a real constraint-dynamics run, II_11_27,
// seed 500001, final generation) is the smallest confirmed case: Operon
// certified d/dn in [0, 96.14] (non-negative -> feasible), but 200k random
// finite-difference samples over the domain box show the true derivative
// is always negative (empirical range roughly [-268, -0.001]) -- Operon's
// bound is unsound here, not just conservative. Step through
// TryAffineBoundDirect/TryAffineBound in a debugger with this test's tag
// alone (`-t "[shape-constraints-scratch]"`) to see which path (affine-
// direct / ill-conditioned-fallback / exception-fallback) produced the
// wrong bound.
TEST_CASE("SCRATCH ind333 sign-wrong derivative bound repro", "[.][shape-constraints-scratch]")
{
    // Must be the real training data (not a degenerate stand-in): the
    // linear-scaling fit (Scale/Offset) that TransformBound applies to the
    // raw derivative bound depends on it, and a degenerate/constant target
    // fits Scale=0, which zeroes out any bound (including a correct one)
    // and produces a misleadingly "feasible" result unrelated to the real bug.
    Operon::Dataset ds("/home/bogdb/src/operon-workspace/operon-publications/experiments/"
                       "shape-constraints-reproduction/results/full_sweep/II_11_27_without_noise_rep00_data.csv",
        /*hasHeader=*/true);
    Operon::Problem problem(&ds);
    problem.SetTrainingRange({ 0, 100 });
    problem.SetTestRange({ 100, 200 });
    problem.SetTarget("y");

    auto const constraintsJson
        = R"json({"domains": {"n": [0, 1], "alpha": [0, 1], "epsilon": [1, 2], "Ef": [1, 2]}, "constraints": [{"op": "derivative", "variable": "n", "order": 1, "sign": 1}, {"op": "derivative", "variable": "alpha", "order": 1, "sign": 1}, {"op": "derivative", "variable": "epsilon", "order": 1, "sign": 1}, {"op": "derivative", "variable": "Ef", "order": 1, "sign": 1}]})json";
    auto const path = WriteShapeConfig("ind333", constraintsJson);
    auto result = Operon::LoadShapeConstraints(path.string());
    REQUIRE(result); // outer Cli::Result
    REQUIRE(*result); // inner std::optional
    auto const& loaded = **result;

    auto const model
        = "(exp((((0.903161883 * n) * (4.700151443 * alpha)) * ((((-4.018970013) * Ef) * (4.741394520 * epsilon)) + "
          "(((2.177207947 * n) * (2.639619350 * alpha)) * (((-1.325337768) * Ef) * (1.827161431 * epsilon)))))) + "
          "(((-4.018970013) * Ef) * ((0.903161883 * n) * (4.700151443 * alpha))))";
    auto tree = InfixParser::ParseOrThrow(model, ds);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    Operon::Evaluator<DTable> nmse(&problem, &dtable, Operon::NMSE {});
    CanonicalShape sce(&nmse, &dtable, loaded);
    auto const summary = sce.Measure(tree);
    REQUIRE(summary.Measurements.size() == 4);
    auto const& dn = summary.Measurements[0]; // d/dn, sign >= 0 required
    INFO("d/dn certified=" << dn.Certified << " bound=[" << (dn.Bound ? dn.Bound->first : 0) << ", "
                           << (dn.Bound ? dn.Bound->second : 0) << "] violation=" << dn.Violation);
    WARN("Operon certifies feasible=" << summary.Feasible
                                      << " for a tree whose true d/dn is always negative over the domain box (verified "
                                         "by finite-difference sampling) -- expected infeasible.");
    CHECK(dn.Certified);
}

// Direct affine-vs-plain-interval comparison on two of the 107 "Operon
// infeasible, HL feasible" disagreements from the 2026-08-09 cross-engine
// bound comparison. Empirical (500k-sample finite-difference) ground truth
// for both: d/dn of the SCALED model is always positive, HL's plain
// interval bound [0, ~7.85] is tight and correct, Operon's affine bound
// [-5.15, 7.85] is a sound but needlessly loose superset (overshoots
// negative). This reproduces that gap directly against Operon's own plain
// IntervalEvaluator (the same one TryAffineBound falls back to) to see
// whether IT also stays tight here -- i.e. whether the looseness is
// specific to AffineEvaluator's linearization of repeated correlated
// multiplication, not interval arithmetic in general.
TEST_CASE("SCRATCH ind431/ind450 affine-vs-interval bound comparison", "[.][shape-constraints-scratch]")
{
    Operon::Dataset ds("/home/bogdb/src/operon-workspace/operon-publications/experiments/"
                       "shape-constraints-reproduction/results/full_sweep/II_11_27_without_noise_rep00_data.csv",
        /*hasHeader=*/true);

    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domains;
    domains.insert_or_assign(ds.GetVariable("n")->Hash, std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } });
    domains.insert_or_assign(ds.GetVariable("alpha")->Hash, std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } });
    domains.insert_or_assign(ds.GetVariable("epsilon")->Hash, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 2 } });
    domains.insert_or_assign(ds.GetVariable("Ef")->Hash, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 2 } });

    struct Case {
        std::string label;
        std::string model;
        Operon::Scalar scale;
    };
    std::vector<Case> const cases {
        { "ind431",
            "(exp(exp((((-1.325337768) * Ef) * (1.827161431 * epsilon)))) + (((0.903161883 * n) * (4.700151443 * "
            "alpha)) * ((((-4.018970013) * Ef) * (4.741394520 * epsilon)) + (((0.903161883 * n) * (4.700151443 * "
            "alpha)) * (((-1.325337768) * Ef) * (1.827161431 * epsilon))))))",
            Operon::Scalar { -0.0116670932 } },
        { "ind450",
            "(((4.700151443 * alpha) * (((-1.325337768) * Ef) * (1.827161431 * epsilon))) + (((0.903161883 * n) * "
            "(4.700151443 * alpha)) * ((((-4.018970013) * Ef) * (4.741394520 * epsilon)) + (((0.903161883 * n) * "
            "(4.700151443 * alpha)) * (((-1.325337768) * Ef) * (1.827161431 * epsilon))))))",
            Operon::Scalar { -0.0104688368 } },
    };

    for (auto const& c : cases) {
        auto tree = InfixParser::ParseOrThrow(c.model, ds);
        auto const nHash = ds.GetVariable("n")->Hash;
        auto const dag = Operon::BuildVariableGradientDag(tree, tree.GetCoefficients());
        auto it = std::ranges::find(dag.Variables, nHash);
        REQUIRE(it != dag.Variables.end());
        auto const k = static_cast<std::size_t>(std::distance(dag.Variables.begin(), it));
        REQUIRE(dag.Certain[k]);
        auto const root = dag.Roots[k];
        REQUIRE(root != std::numeric_limits<std::size_t>::max());

        Operon::Vector<Operon::Node> sliced(
            dag.Nodes.begin(), dag.Nodes.begin() + static_cast<std::ptrdiff_t>(root) + 1);
        Operon::Tree dtree(std::move(sliced));
        dtree.UpdateNodes();

        Operon::AffineEvaluator<Operon::Scalar> ae(&dtree, domains);
        auto const affineResult = ae(dtree.GetCoefficients());
        REQUIRE(affineResult);
        auto const affineRaw = affineResult->to_interval();

        Operon::IntervalEvaluator<Operon::Scalar> ie(
            &dtree, Operon::IntervalEvaluator<Operon::Scalar>::DomainMap { domains });
        auto const intervalResult = ie(dtree.GetCoefficients());
        REQUIRE(intervalResult);
        auto const intervalRaw = *intervalResult;

        // Apply the scale factor manually (matches TransformBound's
        // ApplyToDerivativeInterval for a pure scale multiply -- offset drops
        // out of a derivative).
        auto const scaleInterval
            = [&](Operon::Scalar lo, Operon::Scalar hi) -> std::pair<Operon::Scalar, Operon::Scalar> {
            auto a = c.scale * lo;
            auto b = c.scale * hi;
            return a <= b ? std::pair { a, b } : std::pair { b, a };
        };
        auto const [affLo, affHi] = scaleInterval(affineRaw.inf(), affineRaw.sup());
        auto const [intLo, intHi] = scaleInterval(intervalRaw.inf(), intervalRaw.sup());

        WARN(c.label << " d/dn (scaled): affine=[" << affLo << ", " << affHi << "] "
                     << "operon-interval=[" << intLo << ", " << intHi << "]");
    }
}

// Pins the certifier's bound plumbing against an independent oracle: for a mixed Identity / d/dX1 / d2/dX1^2 constraint
// set, every reported bound must equal what the corresponding evaluator gives for the same tree or hand-sliced
// derivative tree, in each backend mode. A regression in the shared derivative slicing or in a mode's policy shows up
// here instead of in a downstream search run.
TEST_CASE("ShapeCertifier - bounds match the evaluators in every bound mode", "[shape-constraints]")
{
    using IE = IntervalEvaluator<Scalar>;
    using AE = AffineEvaluator<Scalar>;

    Fixture fx;
    auto const tree = InfixParser::ParseOrThrow("(X1 - 2) * (X1 - 2) * X2", fx.ds);
    auto const x1 = fx.ds.GetVariable("X1").value().Hash;
    auto const x2 = fx.ds.GetVariable("X2").value().Hash;

    auto const derivativeSlice = [](Tree const& t, Operon::Hash variable) {
        auto const dag = BuildVariableGradientDag(t, t.GetCoefficients());
        auto const it = std::ranges::find(dag.Variables, variable);
        REQUIRE(it != dag.Variables.end());
        auto const root = dag.Roots[static_cast<std::size_t>(std::distance(dag.Variables.begin(), it))];
        REQUIRE(root != std::numeric_limits<std::size_t>::max());
        Tree sliced(Operon::Vector<Node>(dag.Nodes.begin(), dag.Nodes.begin() + static_cast<std::ptrdiff_t>(root) + 1));
        sliced.UpdateNodes();
        return sliced;
    };
    auto const d1 = derivativeSlice(tree, x1);
    auto const d2 = derivativeSlice(d1, x1);
    std::array<Tree const*, 3> const trees { &tree, &d1, &d2 };

    IE::DomainMap domains;
    domains.insert_or_assign(x1, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    domains.insert_or_assign(x2, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });

    auto const intervalBound = [&](Tree const& t) {
        auto const result = IE(&t, domains)(t.GetCoefficients());
        REQUIRE(result);
        return *result;
    };
    auto const affineBound = [&](Tree const& t) {
        auto const result = AE(&t, domains)(t.GetCoefficients());
        REQUIRE(result);
        return result->to_interval();
    };

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1e6 }, Operon::Scalar { 1e6 } } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::SecondDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    auto const measure = [&](ShapeBoundMode mode, int depth = 0) {
        Operon::ShapeCertifier certifier(
            &fx.problem, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = mode, .BisectionDepth = depth });
        auto summary = certifier.Measure(tree, std::nullopt);
        REQUIRE(summary.Measurements.size() == trees.size());
        return summary;
    };
    constexpr Operon::Scalar tol { 1e-4F };

    auto const violationOf = [&](std::size_t i, IE::Interval const& b) -> Operon::Scalar {
        auto const& c = cs.Constraints[i];
        if (c.Sign) {
            return *c.Sign > 0 ? std::max(Operon::Scalar { 0 }, -b.inf()) : std::max(Operon::Scalar { 0 }, b.sup());
        }
        return std::max(Operon::Scalar { 0 }, c.Bound->first - b.inf())
            + std::max(Operon::Scalar { 0 }, b.sup() - c.Bound->second);
    };

    // d/dX1 = 2 (X1 - 2) X2 reaches -10 at (1, 5), so the non-decreasing constraint is genuinely violated in every
    // mode: a sound enclosure cannot report less than 10, and the set is infeasible.
    constexpr Operon::Scalar minFirstDerivativeViolation { 10 };

    // The summary of each exact-oracle mode: bounds, per-constraint violation, total violation and feasibility.
    auto const checkExact = [&](ShapeBoundMode mode, auto const& expectedBound) {
        auto const summary = measure(mode);
        Operon::Scalar total { 0 };
        bool feasible = true;
        for (std::size_t i = 0; i < trees.size(); ++i) {
            auto const& m = summary.Measurements[i];
            REQUIRE(m.Certified);
            REQUIRE(m.Bound);
            auto const expected = expectedBound(*trees[i]);
            CHECK(m.Bound->first == Catch::Approx(expected.inf()).margin(tol));
            CHECK(m.Bound->second == Catch::Approx(expected.sup()).margin(tol));
            auto const violation = violationOf(i, expected);
            CHECK(m.Violation == Catch::Approx(violation).margin(tol));
            total += violation;
            feasible = feasible && violation == Operon::Scalar { 0 };
        }
        CHECK(summary.Violation == Catch::Approx(total).margin(tol));
        CHECK(summary.Feasible == feasible);
        CHECK_FALSE(summary.Feasible);
        CHECK(summary.Measurements[1].Violation >= minFirstDerivativeViolation - 1e-3F);
        CHECK(summary.Measurements[0].Violation == Operon::Scalar { 0 });
    };

    // Interval mode bounds the second derivative with the second-order interval jet over the original tree (no
    // materialized derivative trees); its enclosure of d2/dX1^2 = 2 X2 must also contain the true range [2, 10].
    SECTION("Interval equals the interval evaluator, and the second derivative the second-order interval jet")
    {
        auto const jet
            = IntervalSecondOrderJetEvaluator<Scalar>(&tree, domains, std::vector { x1 })(tree.GetCoefficients());
        REQUIRE(jet);
        REQUIRE(jet->Status[0] == IntervalJetStatus::Certain);
        CHECK(jet->Second[0].inf() <= 2.0F);
        CHECK(jet->Second[0].sup() >= 10.0F);
        checkExact(ShapeBoundMode::Interval,
            [&](Tree const& t) -> IE::Interval { return &t == &d2 ? jet->Second[0] : intervalBound(t); });
    }

    SECTION("Affine equals the affine evaluator") { checkExact(ShapeBoundMode::Affine, affineBound); }

    SECTION("Combined is the intersection of both evaluators")
    {
        checkExact(ShapeBoundMode::Combined, [&](Tree const& t) {
            auto const a = affineBound(t);
            auto const b = intervalBound(t);
            auto const lo = std::max(a.inf(), b.inf());
            auto const hi = std::min(a.sup(), b.sup());
            REQUIRE(lo <= hi);
            return IE::Interval(lo, hi);
        });
    }

    // mode:3 stays inside the same mode's whole-box bound (depth 0), reports the violation of the bound it returns, and
    // still finds the constraint set infeasible.
    auto const checkSubdivided = [&](ShapeBoundMode mode) {
        auto const whole = measure(mode);
        auto const summary = measure(mode, 3);
        Operon::Scalar total { 0 };
        for (std::size_t i = 0; i < trees.size(); ++i) {
            auto const& m = summary.Measurements[i];
            auto const& w = whole.Measurements[i];
            REQUIRE(m.Certified);
            REQUIRE(m.Bound);
            REQUIRE(w.Bound);
            CHECK(m.Bound->first >= w.Bound->first - tol);
            CHECK(m.Bound->second <= w.Bound->second + tol);
            auto const reported = violationOf(i, IE::Interval(m.Bound->first, m.Bound->second));
            CHECK(m.Violation == Catch::Approx(reported).margin(tol));
            CHECK(m.Violation <= w.Violation + tol);
            total += reported;
        }
        CHECK(summary.Violation == Catch::Approx(total).margin(tol));
        CHECK_FALSE(summary.Feasible);
        CHECK(summary.Measurements[1].Violation >= minFirstDerivativeViolation - 1e-3F);
        CHECK(summary.Measurements[0].Violation == Operon::Scalar { 0 });
    };

    SECTION("Interval:3 stays inside the whole-box interval bound") { checkSubdivided(ShapeBoundMode::Interval); }

    SECTION("Affine:3 stays inside the whole-box affine bound") { checkSubdivided(ShapeBoundMode::Affine); }

    SECTION("Combined:3 stays inside the whole-box combined bound") { checkSubdivided(ShapeBoundMode::Combined); }
}

namespace {
    // A Problem over the input columns X1..Xn plus a trailing target column, for trees with many variables.
    struct WideFixture {
        static constexpr auto Nrow { 8 };

        Eigen::Array<Operon::Scalar, -1, -1> data;
        Operon::Dataset ds;
        DispatchTable<Operon::Scalar> dtable;
        Operon::Problem problem;

        explicit WideFixture(int variables)
            : data(Eigen::Array<Operon::Scalar, -1, -1>::Constant(Nrow, variables + 1, Operon::Scalar { 1.5 }))
            , ds(gsl::not_null { data.data() }, Nrow, variables + 1)
            , problem(&ds)
        {
            problem.SetTrainingRange({ 0, Nrow });
            problem.SetTestRange({ 0, Nrow });
            problem.SetTarget("X" + std::to_string(variables + 1));
        }
    };

    auto BoundConstraint(ShapeConstraintOp op, std::string variable) -> Operon::ShapeConstraint
    {
        return { .Op = op,
            .Variable = std::move(variable),
            .Sign = std::nullopt,
            .Bound = std::pair { Operon::Scalar { -1e6 }, Operon::Scalar { 1e6 } } };
    }

    using SecondOrderEvaluator = IntervalSecondOrderJetEvaluator<Operon::Scalar>;

    // The jet enclosure of d2 tree / d variable^2 over `domains`, one direction.
    auto SingleDirectionSecondDerivative(Tree const& tree, SecondOrderEvaluator::DomainMap const& domains,
        Operon::Hash variable) -> SecondOrderEvaluator::Interval
    {
        auto const result = SecondOrderEvaluator(&tree, domains, std::vector { variable })(tree.GetCoefficients());
        REQUIRE(result);
        REQUIRE(result->Status[0] == IntervalJetStatus::Certain);
        return result->Second[0];
    }
} // namespace
namespace {
    using FirstOrderEvaluator = IntervalJetEvaluator<Operon::Scalar>;

    auto SingleDirectionFirstDerivative(Tree const& tree, FirstOrderEvaluator::DomainMap const& domains,
        Operon::Hash variable) -> FirstOrderEvaluator::Interval
    {
        auto const result = FirstOrderEvaluator(&tree, domains, std::vector { variable })(tree.GetCoefficients());
        REQUIRE(result);
        REQUIRE(result->Status[0] == IntervalJetStatus::Certain);
        return result->Derivative[0];
    }
} // namespace

TEST_CASE("ShapeCertifier - Interval-mode first derivatives batch duplicate and chunked jet directions",
    "[shape-constraints][interval-jet]")
{
    constexpr int Variables = 6;
    WideFixture fx(Variables);
    auto const tree = InfixParser::ParseOrThrow(
        "X1 * X1 + X2 * X2 * X2 + exp(X3) + X4 * X4 * X4 * X4 + sin(X5) + X6 * X6 * X6", fx.ds);

    Operon::ShapeConstraintSet cs;
    FirstOrderEvaluator::DomainMap domains;
    std::array<Operon::Hash, Variables> hashes {};
    for (int i = 0; i < Variables; ++i) {
        auto const name = "X" + std::to_string(i + 1);
        cs.Domains.insert_or_assign(name, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 2 } });
        hashes[static_cast<std::size_t>(i)] = fx.ds.GetVariable(name).value().Hash;
        domains.insert_or_assign(
            hashes[static_cast<std::size_t>(i)], std::pair { Operon::Scalar { 1 }, Operon::Scalar { 2 } });
    }

    // Six distinct directions take two four-wide passes. X2 is deliberately duplicated and Identity is interleaved.
    constexpr std::array<int, 8> order { 2, 0, 6, 1, 5, 3, 4, 2 }; // 0 = Identity, otherwise variable index
    for (auto const v : order) {
        cs.Constraints.push_back(v == 0 ? BoundConstraint(ShapeConstraintOp::Identity, "")
                                        : BoundConstraint(ShapeConstraintOp::FirstDerivative, "X" + std::to_string(v)));
    }

    Operon::ShapeCertifier certifier(
        &fx.problem, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    auto const summary = certifier.Measure(tree, std::nullopt);
    REQUIRE(summary.Measurements.size() == order.size());
    for (std::size_t i = 0; i < order.size(); ++i) {
        auto const& measurement = summary.Measurements[i];
        REQUIRE(measurement.Certified);
        REQUIRE(measurement.Bound);
        if (order[i] == 0) {
            auto const value = IntervalEvaluator<Scalar>(&tree, domains)(tree.GetCoefficients());
            REQUIRE(value);
            CHECK(measurement.Bound->first == value->inf());
            CHECK(measurement.Bound->second == value->sup());
            continue;
        }
        auto const expected
            = SingleDirectionFirstDerivative(tree, domains, hashes[static_cast<std::size_t>(order[i] - 1)]);
        CHECK(measurement.Bound->first == expected.inf());
        CHECK(measurement.Bound->second == expected.sup());
    }
    CHECK(summary.Measurements[0].Bound == summary.Measurements[7].Bound);
}

TEST_CASE("ShapeCertifier - first-order jet uncertainty is isolated by direction", "[shape-constraints][interval-jet]")
{
    Fixture fx;
    auto const tree = InfixParser::ParseOrThrow("abs(X2) + X1 * X1", fx.ds);
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::FirstDerivative, "X1"));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::FirstDerivative, "X2"));

    Operon::ShapeCertifier certifier(
        &fx.problem, &fx.dtable, cs, Operon::Scalar { 2.5 }, { .Mode = ShapeBoundMode::Interval });
    auto const summary = certifier.Measure(tree, std::nullopt);
    REQUIRE(summary.Measurements.size() == 2);
    REQUIRE(summary.Measurements[0].Certified);
    REQUIRE(summary.Measurements[0].Bound);
    CHECK(summary.Measurements[0].Bound->first == Catch::Approx(2.0).margin(1e-4));
    CHECK(summary.Measurements[0].Bound->second == Catch::Approx(10.0).margin(1e-4));
    CHECK_FALSE(summary.Measurements[1].Certified);
    CHECK(summary.Measurements[1].Violation == Catch::Approx(2.5));
}
TEST_CASE("ShapeCertifier - first-order jet failure falls back without contaminating another direction",
    "[shape-constraints][interval-jet]")
{
    Fixture fx;
    auto const tree = InfixParser::ParseOrThrow("sqrt(X1) + X2", fx.ds);
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 0 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::FirstDerivative, "X1"));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::FirstDerivative, "X2"));

    Operon::ShapeCertifier certifier(
        &fx.problem, &fx.dtable, cs, Operon::Scalar { 2.5 }, { .Mode = ShapeBoundMode::Interval });
    auto const summary = certifier.Measure(tree, std::nullopt);
    REQUIRE(summary.Measurements.size() == 2);
    CHECK_FALSE(summary.Measurements[0].Certified);
    CHECK(summary.Measurements[0].Violation == Catch::Approx(2.5));
    REQUIRE(summary.Measurements[1].Certified);
    REQUIRE(summary.Measurements[1].Bound);
    CHECK(summary.Measurements[1].Bound->first == Catch::Approx(1.0));
    CHECK(summary.Measurements[1].Bound->second == Catch::Approx(1.0));
}

TEST_CASE("ShapeCertifier - Interval-mode second derivatives are enclosed by the second-order jet",
    "[shape-constraints][second-order-jet]")
{
    Fixture fx;
    auto const tree = InfixParser::ParseOrThrow("(X1 - 2) * (X1 - 2) * X2", fx.ds);
    auto const x1 = fx.ds.GetVariable("X1").value().Hash;
    auto const x2 = fx.ds.GetVariable("X2").value().Hash;

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X1"));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X2"));

    SecondOrderEvaluator::DomainMap domains;
    domains.insert_or_assign(x1, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    domains.insert_or_assign(x2, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });

    Operon::ShapeCertifier certifier(
        &fx.problem, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    auto const summary = certifier.Measure(tree, std::nullopt);
    REQUIRE(summary.Measurements.size() == 2);

    // d2/dX1^2 = 2 X2 in [2, 10]; the tree is linear in X2, so d2/dX2^2 is exactly 0
    auto const& first = summary.Measurements[0];
    REQUIRE(first.Certified);
    REQUIRE(first.Bound);
    CHECK(first.Bound->first <= 2.0F);
    CHECK(first.Bound->second >= 10.0F);
    auto const expectedFirst = SingleDirectionSecondDerivative(tree, domains, x1);
    CHECK(first.Bound->first == expectedFirst.inf());
    CHECK(first.Bound->second == expectedFirst.sup());

    auto const& second = summary.Measurements[1];
    REQUIRE(second.Certified);
    REQUIRE(second.Bound);
    CHECK(second.Bound->first == 0.0F);
    CHECK(second.Bound->second == 0.0F);
    CHECK(summary.Feasible);
}

TEST_CASE("ShapeCertifier - repeated variables and Identity constraints map to the right jet directions",
    "[shape-constraints][second-order-jet]")
{
    Fixture fx;
    auto const tree = InfixParser::ParseOrThrow("X1 * X1 * X2 + sin(X2)", fx.ds);
    auto const x1 = fx.ds.GetVariable("X1").value().Hash;
    auto const x2 = fx.ds.GetVariable("X2").value().Hash;

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    // 0 d2/dX2^2, 1 value, 2 d2/dX1^2, 3 d2/dX2^2 again, 4 d/dX1 (symbolic dag), 5 d2/dX1^2 again
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X2"));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::Identity, ""));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X1"));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X2"));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::FirstDerivative, "X1"));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X1"));

    SecondOrderEvaluator::DomainMap domains;
    domains.insert_or_assign(x1, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    domains.insert_or_assign(x2, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    auto const expectedX1 = SingleDirectionSecondDerivative(tree, domains, x1);
    auto const expectedX2 = SingleDirectionSecondDerivative(tree, domains, x2);
    // d2/dX1^2 = 2 X2 in [2, 10] and d2/dX2^2 = -sin(X2) in [-1, 1] over [1, 5]: distinct, so a swapped direction
    // mapping cannot pass unnoticed
    REQUIRE(expectedX1.inf() <= 2.0F);
    REQUIRE(expectedX1.sup() >= 10.0F);
    REQUIRE(expectedX2.inf() <= -1.0F);
    REQUIRE(expectedX2.sup() >= 1.0F);
    REQUIRE(expectedX1.sup() > expectedX2.sup());

    Operon::ShapeCertifier certifier(
        &fx.problem, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    auto const summary = certifier.Measure(tree, std::nullopt);
    REQUIRE(summary.Measurements.size() == 6);
    for (auto const& m : summary.Measurements) {
        REQUIRE(m.Certified);
        REQUIRE(m.Bound);
    }
    CHECK(summary.Feasible);

    for (auto const i : { std::size_t { 0 }, std::size_t { 3 } }) {
        CHECK(summary.Measurements[i].Bound->first == expectedX2.inf());
        CHECK(summary.Measurements[i].Bound->second == expectedX2.sup());
    }
    for (auto const i : { std::size_t { 2 }, std::size_t { 5 } }) {
        CHECK(summary.Measurements[i].Bound->first == expectedX1.inf());
        CHECK(summary.Measurements[i].Bound->second == expectedX1.sup());
    }

    // the Identity constraint is still the plain interval bound of the tree (it never takes a jet direction)
    IntervalEvaluator<Scalar>::DomainMap intervalDomains;
    intervalDomains.insert_or_assign(x1, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    intervalDomains.insert_or_assign(x2, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    auto const value = IntervalEvaluator<Scalar>(&tree, intervalDomains)(tree.GetCoefficients());
    REQUIRE(value);
    CHECK(summary.Measurements[1].Bound->first == value->inf());
    CHECK(summary.Measurements[1].Bound->second == value->sup());

    // d/dX1 = 2 X1 X2 in [2, 50] (symbolic derivative dag)
    CHECK(summary.Measurements[4].Bound->first <= 2.0F);
    CHECK(summary.Measurements[4].Bound->second >= 50.0F);

    // the same set with the constraints reversed maps each bound to the same variable
    auto reversed = cs;
    std::ranges::reverse(reversed.Constraints);
    Operon::ShapeCertifier reversedCertifier(
        &fx.problem, &fx.dtable, reversed, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    auto const reversedSummary = reversedCertifier.Measure(tree, std::nullopt);
    REQUIRE(reversedSummary.Measurements.size() == 6);
    for (std::size_t i = 0; i < 6; ++i) {
        auto const& a = summary.Measurements[5 - i];
        auto const& b = reversedSummary.Measurements[i];
        REQUIRE(b.Certified);
        CHECK(a.Bound->first == b.Bound->first);
        CHECK(a.Bound->second == b.Bound->second);
    }
}

TEST_CASE("ShapeCertifier - more distinct second-derivative variables than jet directions",
    "[shape-constraints][second-order-jet]")
{
    constexpr int Variables = 6;
    WideFixture fx(Variables);
    auto const tree = InfixParser::ParseOrThrow(
        "X1 * X1 + 2 * X2 * X2 * X2 + exp(X3) + X4 * X4 * X4 * X4 + sin(X5) + X6 * X6 * X6", fx.ds);

    Operon::ShapeConstraintSet cs;
    SecondOrderEvaluator::DomainMap domains;
    std::array<Operon::Hash, Variables> hashes {};
    for (int i = 0; i < Variables; ++i) {
        auto const name = "X" + std::to_string(i + 1);
        cs.Domains.insert_or_assign(name, std::pair { Operon::Scalar { 1 }, Operon::Scalar { 2 } });
        hashes[static_cast<std::size_t>(i)] = fx.ds.GetVariable(name).value().Hash;
        domains.insert_or_assign(
            hashes[static_cast<std::size_t>(i)], std::pair { Operon::Scalar { 1 }, Operon::Scalar { 2 } });
    }

    // six distinct variables (two jet passes), one repeated, the Identity constraint between them
    constexpr std::array<int, 8> order { 6, 1, 5, 0, 2, 4, 3, 6 }; // 0 = Identity, else the variable's index
    for (auto const v : order) {
        cs.Constraints.push_back(v == 0
                ? BoundConstraint(ShapeConstraintOp::Identity, "")
                : BoundConstraint(ShapeConstraintOp::SecondDerivative, "X" + std::to_string(v)));
    }

    Operon::ShapeCertifier certifier(
        &fx.problem, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = ShapeBoundMode::Interval });
    auto const summary = certifier.Measure(tree, std::nullopt);
    REQUIRE(summary.Measurements.size() == order.size());

    // true ranges of d2/dXv^2 over [1, 2]: 2, 12 X2 in [12, 24], exp(X3) in [e, e^2], 12 X4^2 in [12, 48],
    // -sin(X5) in [-1, -sin(1)], 6 X6 in [6, 12]
    std::array<std::pair<double, double>, Variables + 1> const truth { { { 0.0, 0.0 }, { 2.0, 2.0 }, { 12.0, 24.0 },
        { std::exp(1.0), std::exp(2.0) }, { 12.0, 48.0 }, { -1.0, -std::sin(1.0) }, { 6.0, 12.0 } } };
    for (std::size_t i = 0; i < order.size(); ++i) {
        auto const& m = summary.Measurements[i];
        REQUIRE(m.Certified);
        REQUIRE(m.Bound);
        if (order[i] == 0) {
            continue;
        }
        INFO("constraint " << i << ": d2/dX" << order[i] << "^2");
        auto const variable = hashes[static_cast<std::size_t>(order[i] - 1)];
        auto const expected = SingleDirectionSecondDerivative(tree, domains, variable);
        CHECK(m.Bound->first == expected.inf());
        CHECK(m.Bound->second == expected.sup());
        auto const [lo, hi] = truth[static_cast<std::size_t>(order[i])];
        CHECK(static_cast<double>(m.Bound->first) <= lo + 1e-4);
        CHECK(static_cast<double>(m.Bound->second) >= hi - 1e-4);
    }
}

TEST_CASE("ShapeCertifier - an unsupported operation only makes the directions it depends on uncertain",
    "[shape-constraints][second-order-jet]")
{
    Fixture fx;
    auto const tree = InfixParser::ParseOrThrow("abs(X2) + X1 * X1", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X1"));
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X2"));

    for (auto const mode : { ShapeBoundMode::Interval, ShapeBoundMode::Combined }) {
        DYNAMIC_SECTION("mode " << static_cast<int>(mode))
        {
            Operon::ShapeCertifier certifier(&fx.problem, &fx.dtable, cs, Operon::Scalar { 2.5 }, { .Mode = mode });
            auto const summary = certifier.Measure(tree, std::nullopt);
            REQUIRE(summary.Measurements.size() == 2);
            REQUIRE(summary.Measurements[0].Certified);
            CHECK(summary.Measurements[0].Bound->first == Catch::Approx(2.0).margin(1e-4));
            CHECK(summary.Measurements[0].Bound->second == Catch::Approx(2.0).margin(1e-4));
            CHECK_FALSE(summary.Measurements[1].Certified);
            CHECK(summary.Measurements[1].Violation == Catch::Approx(2.5));
            CHECK_FALSE(summary.Feasible);
        }
    }
}

TEST_CASE("ShapeCertifier - second-derivative bounds enclose the true range in every bound mode",
    "[shape-constraints][second-order-jet]")
{
    Fixture fx;
    auto const tree = InfixParser::ParseOrThrow("sin(X1) * X2 + exp(X1 / 5) / (X2 + 1)", fx.ds);

    // d2/dX1^2 = -sin(X1) X2 + exp(X1 / 5) / (25 (X2 + 1)) over [1, 5]^2, sampled on a grid
    double lo = std::numeric_limits<double>::infinity();
    double hi = -lo;
    constexpr int Grid = 64;
    for (int i = 0; i <= Grid; ++i) {
        for (int j = 0; j <= Grid; ++j) {
            auto const x = 1.0 + 4.0 * i / Grid;
            auto const y = 1.0 + 4.0 * j / Grid;
            auto const d2 = -std::sin(x) * y + std::exp(x / 5.0) / (25.0 * (y + 1.0));
            lo = std::min(lo, d2);
            hi = std::max(hi, d2);
        }
    }

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(BoundConstraint(ShapeConstraintOp::SecondDerivative, "X1"));

    for (auto const [mode, depth] :
        { std::pair { ShapeBoundMode::Interval, 0 }, std::pair { ShapeBoundMode::Interval, 3 },
            std::pair { ShapeBoundMode::Combined, 0 }, std::pair { ShapeBoundMode::Combined, 3 },
            std::pair { ShapeBoundMode::Affine, 0 }, std::pair { ShapeBoundMode::Affine, 3 } }) {
        DYNAMIC_SECTION("mode " << static_cast<int>(mode) << " depth " << depth)
        {
            Operon::ShapeCertifier certifier(
                &fx.problem, &fx.dtable, cs, Operon::Scalar { 1 }, { .Mode = mode, .BisectionDepth = depth });
            auto const summary = certifier.Measure(tree, std::nullopt);
            REQUIRE(summary.Measurements.size() == 1);
            REQUIRE(summary.Measurements[0].Certified);
            REQUIRE(summary.Measurements[0].Bound);
            CHECK(static_cast<double>(summary.Measurements[0].Bound->first) <= lo + 1e-4);
            CHECK(static_cast<double>(summary.Measurements[0].Bound->second) >= hi - 1e-4);
        }
    }
}

namespace {
    // The schedule of `expr` over [0.75, 1.25] x [0.75, 1.25] (or the given widths), as halvings per variable name.
    auto PlannedSplits(std::string const& expr, int depth, Operon::Scalar width1 = Operon::Scalar { 0.5 },
        Operon::Scalar width2 = Operon::Scalar { 0.5 }) -> std::map<std::string, int>
    {
        Eigen::Array<Operon::Scalar, -1, -1> data(2, 3);
        data << Operon::Scalar { 0.75 }, Operon::Scalar { 0.75 }, Operon::Scalar { 0 }, Operon::Scalar { 1.25 },
            Operon::Scalar { 1.25 }, Operon::Scalar { 0 };
        Operon::Dataset ds(gsl::not_null { data.data() }, 2, 3);
        auto const tree = Operon::InfixParser::ParseOrThrow(expr, ds);
        auto const x1 = ds.GetVariable("X1").value().Hash;
        auto const x2 = ds.GetVariable("X2").value().Hash;
        Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domains {
            { x1, { Operon::Scalar { 0.75 }, Operon::Scalar { 0.75 } + width1 } },
            { x2, { Operon::Scalar { 0.75 }, Operon::Scalar { 0.75 } + width2 } },
        };
        std::map<std::string, int> result;
        for (auto const& [axis, count] : detail::BisectionSplits(tree, domains, depth)) {
            result[axis == x1 ? "X1" : "X2"] = count;
        }
        return result;
    }
} // namespace

TEST_CASE("Bisection schedule spends the splits on variables that occur repeatedly", "[shape-constraints][bisection]")
{
    // X1 occurs twice and X2 once, so X2 is enclosed exactly and every split goes to X1, whichever variable is
    // wider and wherever it appears in the tree.
    for (auto const* expr : { "X1 + X2 * (X1 - 0.9) ^ 4", "X2 * (X1 - 0.9) ^ 4 + X1", "X1 * X1 + X2" }) {
        for (int const depth : { 1, 3, 6 }) {
            INFO(expr << " depth " << depth);
            auto const equalWidths = PlannedSplits(expr, depth);
            CHECK(equalWidths == std::map<std::string, int> { { "X1", depth } });
            auto const wideSingle = PlannedSplits(expr, depth, Operon::Scalar { 0.5 }, Operon::Scalar { 8 });
            CHECK(wideSingle == std::map<std::string, int> { { "X1", depth } });
        }
    }
}

TEST_CASE("Bisection schedule is widest-first when no variable repeats", "[shape-constraints][bisection]")
{
    // X1 over width 4 and X2 over width 1: three halvings of X1 equalise the widths, so depth 3 is all X1 and
    // depth 5 is X1 x4 / X2 x1 (4 -> 0.25 vs 1 -> 0.5 after four, then the fifth goes to X2).
    CHECK(PlannedSplits("X1 + X2", 3, Operon::Scalar { 4 }, Operon::Scalar { 1 })
        == std::map<std::string, int> { { "X1", 3 } });
    CHECK(PlannedSplits("X1 * X2", 5, Operon::Scalar { 4 }, Operon::Scalar { 1 })
        == std::map<std::string, int> { { "X1", 4 }, { "X2", 1 } });
}

TEST_CASE("Bisection schedule does not depend on how the tree is laid out", "[shape-constraints][bisection]")
{
    // Equal widths and equal occurrence counts: the tie goes to the variable with the smaller hash, so swapping the
    // operands (which swaps first appearance) cannot change which variable gets the extra split.
    for (int const depth : { 1, 3, 5 }) {
        INFO("depth " << depth);
        CHECK(PlannedSplits("X1 * X2", depth) == PlannedSplits("X2 * X1", depth));
        CHECK(PlannedSplits("sin(X1) + cos(X2)", depth) == PlannedSplits("cos(X2) + sin(X1)", depth));
    }
}

TEST_CASE("Bisection schedule counts variables in the expanded expression", "[shape-constraints][bisection]")
{
    Eigen::Array<Operon::Scalar, -1, -1> data(2, 3);
    data << Operon::Scalar { 0.75 }, Operon::Scalar { 0.75 }, Operon::Scalar { 0 }, Operon::Scalar { 1.25 },
        Operon::Scalar { 1.25 }, Operon::Scalar { 0 };
    Operon::Dataset ds(gsl::not_null { data.data() }, 2, 3);
    auto const x1 = ds.GetVariable("X1").value().Hash;
    auto const x2 = ds.GetVariable("X2").value().Hash;
    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domains {
        { x1, { Operon::Scalar { 0.75 }, Operon::Scalar { 1.25 } } },
        { x2, { Operon::Scalar { 0.75 }, Operon::Scalar { 1.25 } } },
    };
    auto const variable = [](Operon::Hash hash) { return Node { NodeType::Variable, hash }; };
    auto const function
        = [](BuiltinOp op, std::size_t arity) { return Node::Function(static_cast<Operon::Hash>(op), arity); };
    auto const splits = [&](Operon::Vector<Node> nodes, int depth) {
        Tree tree { std::move(nodes) };
        tree.UpdateNodes();
        return detail::BisectionSplits(tree, domains, depth);
    };

    SECTION("a Ref slot counts as the subtree it points at")
    {
        // X2 + X1 * X1, the second X1 written as a Ref slot: X1 occurs twice in the expanded expression, X2 once.
        auto const result = splits(
            { variable(x2), variable(x1), Node::Ref(1), function(BuiltinOp::Mul, 2), function(BuiltinOp::Add, 2) }, 4);
        REQUIRE(result.size() == 1);
        CHECK(result[0] == std::pair { x1, 4 });
    }

    SECTION("a variable the root does not reach takes no axis")
    {
        // [X2, X1]: the root is X1 alone; X2 is a dead node.
        auto const result = splits({ variable(x2), variable(x1) }, 3);
        REQUIRE(result.size() == 1);
        CHECK(result[0] == std::pair { x1, 3 });
    }
}

TEST_CASE("Bisection schedule stays balanced on deeply shared expressions", "[shape-constraints][bisection]")
{
    Eigen::Array<Operon::Scalar, -1, -1> data(2, 3);
    data << Operon::Scalar { -1 }, Operon::Scalar { -1 }, Operon::Scalar { 0 }, Operon::Scalar { 1 },
        Operon::Scalar { 1 }, Operon::Scalar { 0 };
    Operon::Dataset ds(gsl::not_null { data.data() }, 2, 3);
    auto const x1 = ds.GetVariable("X1").value().Hash;
    auto const x2 = ds.GetVariable("X2").value().Hash;
    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domains {
        { x1, { Operon::Scalar { -1 }, Operon::Scalar { 1 } } },
        { x2, { Operon::Scalar { -1 }, Operon::Scalar { 1 } } },
    };

    // g0 = X1 + X2, g(n+1) = 0.5 * (g(n) + Ref(g(n))): every level doubles how often each variable is evaluated, so
    // the expanded occurrence counts reach 2^levels (the expression is still X1 + X2). Equal counts and widths must
    // keep splitting the two variables alternately however deep the sharing goes.
    for (int const levels : { 4, 100, 511, 512, 1024 }) {
        INFO("levels " << levels);
        Operon::Vector<Node> nodes { Node { NodeType::Variable, x1 }, Node { NodeType::Variable, x2 },
            Node::Function(static_cast<Operon::Hash>(BuiltinOp::Add), 2) };
        for (int level = 0; level < levels; ++level) {
            nodes.push_back(Node::Ref(static_cast<std::uint16_t>(nodes.size() - 1)));
            nodes.push_back(Node::Function(static_cast<Operon::Hash>(BuiltinOp::Add), 2));
            nodes.push_back(Node::Constant(0.5F));
            nodes.push_back(Node::Function(static_cast<Operon::Hash>(BuiltinOp::Mul), 2));
        }
        Tree tree { nodes };
        tree.UpdateNodes();
        REQUIRE(tree.Validate());
        auto const result = detail::BisectionSplits(tree, domains, 6);
        REQUIRE(result.size() == 2);
        CHECK(result[0].second == 3);
        CHECK(result[1].second == 3);
    }
}

namespace {

    // One expression with its domain box and, per constraint of `SubdivisionFixture::Constraints()`, an analytic
    // reference (nullptr when none is given): Identity, d/dX1, d2/dX1^2, d/dX2, d2/dX2^2.
    struct SubdivisionCase {
        using F = double (*)(double, double);
        char const* Expression;
        std::pair<Operon::Scalar, Operon::Scalar> X1;
        std::pair<Operon::Scalar, Operon::Scalar> X2;
        std::array<F, 5> Truth;
        bool StrictlyTighterFirstX1; // interval:N (N > 0) is strictly tighter on d/dX1 (dependency problem)
    };

    auto SubdivisionCases() -> std::array<SubdivisionCase, 3>
    {
        return { {
            // X1 repeated three times; d/dX1 = 3 X1^2 on [-1, 1]: whole-box jet gives [-3, 3], bisection [0, 3].
            { "X1 * X1 * X1", { -1, 1 }, { 1, 2 },
                { [](double x, double) { return x * x * x; }, [](double x, double) { return 3 * x * x; },
                    [](double x, double) { return 6 * x; }, [](double, double) { return 0.0; },
                    [](double, double) { return 0.0; } },
                true },
            // X1 repeated inside a product, X2 once.
            { "sin(X1) * X1 + X2", { -1, 2 }, { 1, 2 },
                { [](double x, double y) { return std::sin(x) * x + y; },
                    [](double x, double) { return std::cos(x) * x + std::sin(x); },
                    [](double x, double) { return -std::sin(x) * x + 2 * std::cos(x); },
                    [](double, double) { return 1.0; }, [](double, double) { return 0.0; } },
                false },
            // two variables, X2 repeated through the quotient.
            { "exp(X1 * X2) / (1 + X2 * X2)", { 0.5F, 2 }, { 0.5F, 2 },
                { [](double x, double y) { return std::exp(x * y) / (1 + y * y); },
                    [](double x, double y) { return y * std::exp(x * y) / (1 + y * y); },
                    [](double x, double y) { return y * y * std::exp(x * y) / (1 + y * y); },
                    [](double x, double y) {
                        return std::exp(x * y) * (x * (1 + y * y) - 2 * y) / ((1 + y * y) * (1 + y * y));
                    },
                    nullptr },
                false },
        } };
    }

    // The bound modes that accept a subdivision depth, with their CLI names.
    constexpr std::array<std::pair<ShapeBoundMode, char const*>, 3> SubdividedModes { {
        { ShapeBoundMode::Interval, "interval" },
        { ShapeBoundMode::Affine, "affine" },
        { ShapeBoundMode::Combined, "combined" },
    } };

    // Dataset X1, X2, X3 (target); raw (unscaled) bounds.
    struct SubdivisionFixture {
        Eigen::Array<Operon::Scalar, -1, -1> data { 4, 3 };
        Operon::Dataset ds;
        Operon::Problem problem;
        DispatchTable<Operon::Scalar> dtable;

        SubdivisionFixture()
            : ds([&]() -> Operon::Dataset {
                data << 1, 1, 1, 2, 1, 2, 1, 2, 3, 2, 2, 4;
                return Operon::Dataset(gsl::not_null { data.data() }, 4, 3);
            }())
            , problem(&ds)
        {
            problem.SetTrainingRange({ 0, 4 });
            problem.SetTestRange({ 0, 4 });
            problem.SetTarget("X3");
            problem.SetLinearScalingEnabled(false);
        }

        [[nodiscard]] static auto Constraints(SubdivisionCase const& jc) -> Operon::ShapeConstraintSet
        {
            Operon::ShapeConstraintSet cs;
            cs.Domains.insert_or_assign("X1", jc.X1);
            cs.Domains.insert_or_assign("X2", jc.X2);
            auto const permissive = std::pair { Operon::Scalar { -1e9 }, Operon::Scalar { 1e9 } };
            cs.Constraints.push_back(
                { .Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = permissive });
            for (auto const* variable : { "X1", "X2" }) {
                cs.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
                    .Variable = variable,
                    .Sign = std::nullopt,
                    .Bound = permissive });
                cs.Constraints.push_back({ .Op = ShapeConstraintOp::SecondDerivative,
                    .Variable = variable,
                    .Sign = std::nullopt,
                    .Bound = permissive });
            }
            // order: Identity, d/dX1, d2/dX1^2, d/dX2, d2/dX2^2
            return cs;
        }

        [[nodiscard]] auto Measure(SubdivisionCase const& jc, Operon::ShapeBoundOptions const& opts) const
            -> Operon::ShapeConstraintMeasurementSummary
        {
            auto const tree = InfixParser::ParseOrThrow(jc.Expression, ds);
            Operon::ShapeCertifier certifier(&problem, &dtable, Constraints(jc), Operon::Scalar { 1 }, opts);
            return certifier.MeasureCanonical(tree);
        }
    };

    // Both summaries certify the same constraints with bitwise-equal bounds.
    void CheckSameMeasurements(Operon::ShapeConstraintMeasurementSummary const& actual,
        Operon::ShapeConstraintMeasurementSummary const& expected)
    {
        REQUIRE(actual.Measurements.size() == expected.Measurements.size());
        for (std::size_t k = 0; k < actual.Measurements.size(); ++k) {
            INFO("constraint " << k);
            CHECK(actual.Measurements[k].Certified == expected.Measurements[k].Certified);
            REQUIRE(actual.Measurements[k].Bound.has_value() == expected.Measurements[k].Bound.has_value());
            if (actual.Measurements[k].Bound) {
                CHECK(actual.Measurements[k].Bound->first == expected.Measurements[k].Bound->first);
                CHECK(actual.Measurements[k].Bound->second == expected.Measurements[k].Bound->second);
            }
        }
    }

} // namespace

TEST_CASE("mode:0 equals the plain mode", "[shape-constraints][subdivision]")
{
    SubdivisionFixture fx;
    for (auto const& [mode, name] : SubdividedModes) {
        for (auto const& jc : SubdivisionCases()) {
            INFO(name << " on " << jc.Expression);
            auto const plain = fx.Measure(jc, { .Mode = mode });
            for (auto const* text : { "", ":0" }) {
                INFO("token " << name << text);
                CheckSameMeasurements(
                    fx.Measure(jc, Operon::ParseShapeBoundOptions(std::string { name } + text)), plain);
            }
        }
    }
}

TEST_CASE("mode:N bounds are sound, never wider than the plain mode, and tighter under the dependency problem",
    "[shape-constraints][subdivision]")
{
    SubdivisionFixture fx;
    constexpr auto Samples = 241;
    for (auto const& [mode, name] : SubdividedModes) {
        for (auto const& jc : SubdivisionCases()) {
            INFO(name << " on " << jc.Expression);
            auto const plain = fx.Measure(jc, { .Mode = mode });
            REQUIRE(plain.Measurements.size() == 5);

            for (int depth = 1; depth <= 4; ++depth) {
                INFO("depth " << depth);
                auto const subdivided = fx.Measure(jc, { .Mode = mode, .BisectionDepth = depth });
                REQUIRE(subdivided.Measurements.size() == 5);
                for (std::size_t k = 0; k < 5; ++k) {
                    INFO("constraint " << k);
                    auto const& m = subdivided.Measurements[k];
                    auto const& base = plain.Measurements[k];
                    REQUIRE(m.Certified);
                    REQUIRE(base.Certified);
                    auto const [lo, hi] = *m.Bound;
                    auto const [baseLo, baseHi] = *base.Bound;
                    // contained in the whole-box (depth 0) bound, up to float rounding
                    auto const slack
                        = 1e-5 * (1.0 + std::max(std::abs(double { baseLo }), std::abs(double { baseHi })));
                    CHECK(double { lo } >= double { baseLo } - slack);
                    CHECK(double { hi } <= double { baseHi } + slack);

                    // contains a dense sample of the true values
                    if (jc.Truth[k] == nullptr) {
                        continue;
                    }
                    auto lowest = std::numeric_limits<double>::infinity();
                    auto highest = -std::numeric_limits<double>::infinity();
                    for (auto i = 0; i < Samples; ++i) {
                        for (auto j = 0; j < Samples; ++j) {
                            auto const x = double { jc.X1.first }
                                + (double { jc.X1.second } - double { jc.X1.first }) * i / (Samples - 1);
                            auto const y = double { jc.X2.first }
                                + (double { jc.X2.second } - double { jc.X2.first }) * j / (Samples - 1);
                            auto const v = jc.Truth[k](x, y);
                            lowest = std::min(lowest, v);
                            highest = std::max(highest, v);
                        }
                    }
                    auto const tol = 1e-5 * (1.0 + std::max(std::abs(lowest), std::abs(highest)));
                    CHECK(double { lo } <= lowest + tol);
                    CHECK(double { hi } >= highest - tol);

                    if (mode == ShapeBoundMode::Interval && jc.StrictlyTighterFirstX1 && k == 1) {
                        CHECK(lo > baseLo);
                        CHECK(hi <= baseHi);
                    }
                }
            }
        }
    }
}

TEST_CASE(
    "interval:N bound shrinks monotonically with depth on a single-axis expression", "[shape-constraints][subdivision]")
{
    SubdivisionFixture fx;
    auto const jc = SubdivisionCases()[0]; // X1 only: the leaf boxes of depth d + 1 refine those of depth d
    auto previous = fx.Measure(jc, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 0 });
    for (int depth = 1; depth <= 6; ++depth) {
        auto const current = fx.Measure(jc, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = depth });
        for (std::size_t k = 0; k < 5; ++k) {
            INFO("depth " << depth << " constraint " << k);
            REQUIRE(current.Measurements[k].Bound);
            REQUIRE(previous.Measurements[k].Bound);
            auto const [lo, hi] = *current.Measurements[k].Bound;
            auto const [prevLo, prevHi] = *previous.Measurements[k].Bound;
            auto const slack = 1e-5F * (1.0F + std::max(std::abs(prevLo), std::abs(prevHi)));
            CHECK(lo >= prevLo - slack);
            CHECK(hi <= prevHi + slack);
        }
        previous = current;
    }
}

TEST_CASE("mode:N falls back to the whole-box result when a sub-box is undecided", "[shape-constraints][subdivision]")
{
    // 1 / X1 over a box containing 0 has no finite enclosure on any sub-box touching 0 either.
    SubdivisionFixture fx;
    SubdivisionCase const jc { "1 / X1 + X1", { -1, 1 }, { 1, 2 }, {}, false };
    for (auto const& [mode, name] : SubdividedModes) {
        INFO(name);
        auto const plain = fx.Measure(jc, { .Mode = mode });
        auto const subdivided = fx.Measure(jc, { .Mode = mode, .BisectionDepth = 3 });
        CheckSameMeasurements(subdivided, plain);
        CHECK_FALSE(subdivided.Measurements[1].Certified);
    }
}

// The step of a leaf box is (hi - lo) / 2^splits rounded to nearest, so the top of the last cell lo + 2^splits * step
// can fall below hi (for example lo = -2.018f, hi = 1.853f at one halving). Unless the outer endpoints are pinned to
// the domain the point hi is in no leaf box and the bound misses f(hi).
TEST_CASE(
    "interval:N leaf boxes cover the top of the domain when the step rounds down", "[shape-constraints][subdivision]")
{
    using S = Operon::Scalar;
    auto const coversTop = [](S lo, S hi, int splits) {
        auto const n = static_cast<S>(1 << splits);
        auto const step = (hi - lo) / n;
        return pappus::fp::ropu<pappus::fp::op_add>(lo, n * step) >= hi;
    };
    // Domains found by search for which the unpinned top falls short: single precision, then double precision.
    constexpr std::array<std::pair<double, double>, 4> candidates { { { -2.018, 1.853 }, { -3.625, 0.211 },
        { -0.951, 0.108 }, { -2.218, 2.914 } } };
    SubdivisionFixture fx;
    auto exercised = 0;
    for (auto const& [candLo, candHi] : candidates) {
        auto const lo = static_cast<S>(candLo);
        auto const hi = static_cast<S>(candHi);
        // Only a domain whose unpinned top falls short at some number of halvings exercises the pinning.
        if (std::ranges::all_of(std::array { 0, 1, 2, 3 }, [&](int splits) { return coversTop(lo, hi, splits); })) {
            continue;
        }
        ++exercised;
        for (int depth = 1; depth <= 3; ++depth) {
            // X1 * X2 over [lo, hi] x [1, 2]: the upper bound is 2 * hi, attained at the top of X1.
            SubdivisionCase const jc { "X1 * X2", { lo, hi }, { 1, 2 }, {}, false };
            auto const measured = fx.Measure(jc, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = depth });
            INFO("lo " << lo << " hi " << hi << " depth " << depth);
            REQUIRE(measured.Measurements[0].Bound);
            CHECK(measured.Measurements[0].Bound->second >= S { 2 } * hi);
            CHECK(measured.Measurements[0].Bound->first <= S { 2 } * lo);
        }
    }
    REQUIRE(exercised > 0);
}

// For a subnormal-width axis the step (hi - lo) / 2^splits underflows to zero, so every leaf box of the axis is the
// point lo unless the top of the last cell is pinned to hi. The packed multi-axis identity route must pin it too.
TEST_CASE("interval:N packed leaf boxes cover a subnormal-width domain", "[shape-constraints][subdivision]")
{
    using S = Operon::Scalar;
    constexpr auto tiny = std::numeric_limits<S>::denorm_min();
    SubdivisionFixture fx;
    // two split axes: the packed SIMD route. Up to 2^12 leaves, so at least one full pack at any vector width.
    SubdivisionCase const jc { "X1 + X2", { 0, tiny }, { 0, tiny }, {}, false };
    for (int depth : { 6, 12 }) {
        INFO("depth " << depth);
        auto const measured = fx.Measure(jc, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = depth });
        REQUIRE(measured.Measurements[0].Bound);
        CHECK(measured.Measurements[0].Bound->first <= S { 0 });
        CHECK(measured.Measurements[0].Bound->second >= S { 2 } * tiny);
    }
}

// The routes that run one evaluator per leaf stop at 12 halvings, so that the largest accepted depth is not a
// 2^20-evaluator-run bound: a single-variable model gives the same jet and affine bounds at 20 as at 12.
TEST_CASE("per-leaf routes perform at most 12 halvings", "[shape-constraints][subdivision]")
{
    SubdivisionFixture fx;
    auto const jc = SubdivisionCases()[0]; // X1 * X1 * X1: a single split variable
    SECTION("interval jet derivatives")
    {
        auto const capped = fx.Measure(jc, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 12 });
        auto const deep = fx.Measure(jc, { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 20 });
        // Identity (constraint 0) is the SIMD route and takes the full depth; the derivatives (1, 2) are jets.
        for (std::size_t k : { 1UL, 2UL }) {
            INFO("constraint " << k);
            REQUIRE(capped.Measurements[k].Bound);
            REQUIRE(deep.Measurements[k].Bound);
            CHECK(*deep.Measurements[k].Bound == *capped.Measurements[k].Bound);
        }
    }
    SECTION("affine")
    {
        CheckSameMeasurements(fx.Measure(jc, { .Mode = ShapeBoundMode::Affine, .BisectionDepth = 20 }),
            fx.Measure(jc, { .Mode = ShapeBoundMode::Affine, .BisectionDepth = 12 }));
    }
}

} // namespace Operon::Test
