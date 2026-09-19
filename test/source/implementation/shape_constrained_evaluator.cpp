// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include "operon/core/constraint.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/core/tree_hash.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
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
            , tree(InfixParser::Parse("X1 - X2", ds))
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

TEST_CASE("ShapeConstrainedEvaluator - correctly-signed constraints are feasible", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
        .Variable = "X1",
        .Sign = 1,
        .Bound = std::nullopt }); // non-decreasing: true
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
        .Variable = "X2",
        .Sign = -1,
        .Bound = std::nullopt }); // non-increasing: true

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    CHECK(sce.Feasible(fx.tree));

    auto ind = Fixture::MakeIndividual(fx.tree);
    std::vector<Operon::Scalar> buf(fx.problem.TrainingRange().Size());
    auto fit = sce(fx.rng, ind, buf);
    auto expected = fx.nmse(fx.rng, ind, buf);
    REQUIRE(fit.size() == expected.size());
    CHECK(fit[0] == Catch::Approx(expected[0])); // passes through to the wrapped NMSE evaluator
    CHECK(sce.Violations() == 0);
}

TEST_CASE("ShapeConstrainedEvaluator preserves a wrapped derived evaluator's objective", "[shape-constraints]")
{
    // Regression: the gate must always delegate phase 2 to the concrete
    // wrapped evaluator. A derived class such as MinimumDescriptionLengthEvaluator
    // shares Evaluator<DTable>'s phase 1 but has a distinct scoring objective.
    Fixture fx;
    Operon::MinimumDescriptionLengthEvaluator<Fixture::DTable, Operon::GaussianLikelihood<Operon::Scalar>> mdl {
        &fx.problem, &fx.dtable
    };

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    Operon::ShapeConstrainedEvaluator sce(&mdl, &fx.dtable, cs);

    auto ind = Fixture::MakeIndividual(fx.tree);
    std::vector<Operon::Scalar> buf(fx.problem.TrainingRange().Size());

    // First call on this tree starts with a feasibleCache_ miss. The gate's phase 1
    // must certify from the wrapped evaluator's values, then phase 2 must preserve
    // the wrapped evaluator's MDL result.
    auto const wrapped = sce(fx.rng, ind, buf);
    auto const direct = mdl(fx.rng, ind, buf);
    REQUIRE(wrapped.size() == direct.size());
    CHECK(wrapped[0] == Catch::Approx(direct[0]));
}

TEST_CASE("ShapeConstrainedEvaluator cache miss and hit preserve the wrapped score", "[shape-constraints]")
{
    // Both calls must reach the wrapped evaluator's phase 2. The first also
    // populates feasibility data from its phase-1 values; the second reads it.
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);

    auto ind = Fixture::MakeIndividual(fx.tree);
    std::vector<Operon::Scalar> buf(fx.problem.TrainingRange().Size());

    auto const miss = sce(fx.rng, ind, buf);
    auto const hit = sce(fx.rng, ind, buf);
    REQUIRE(miss.size() == hit.size());
    CHECK(miss[0] == Catch::Approx(hit[0]));
}

TEST_CASE("ShapeConstrainedEvaluator - wrongly-signed constraint is rejected with WorstValue", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    // f is actually non-decreasing in X1; asserting the opposite must be rejected.
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = -1, .Bound = std::nullopt });

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(sce.Feasible(fx.tree));

    auto ind = Fixture::MakeIndividual(fx.tree);
    std::vector<Operon::Scalar> buf(fx.problem.TrainingRange().Size());
    auto fit = sce(fx.rng, ind, buf);
    REQUIRE(fit.size() == 1);
    CHECK(fit[0] == Catch::Approx(1.0)); // default WorstValue
    CHECK(sce.Violations() == 1);
}

TEST_CASE("ShapeConstrainedEvaluator - value bound constraint", "[shape-constraints]")
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
    Operon::ShapeConstrainedEvaluator wide(&fx.nmse, &fx.dtable, cs);
    CHECK(wide.Feasible(fx.tree));

    cs.Constraints[0].Bound = std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } };
    Operon::ShapeConstrainedEvaluator narrow(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(narrow.Feasible(fx.tree));
}

TEST_CASE("ParseShapeBoundMode parses flags and rejects invalid combinations", "[shape-constraints]")
{
    CHECK(Operon::ParseShapeBoundMode("combined") == ShapeBoundMode::Combined);
    CHECK(Operon::ParseShapeBoundMode("interval") == ShapeBoundMode::Interval);
    CHECK(Operon::ParseShapeBoundMode("affine") == ShapeBoundMode::Affine);
    CHECK(Operon::ParseShapeBoundMode("interval,bisected") == (ShapeBoundMode::Interval | ShapeBoundMode::Bisected));
    CHECK_THROWS_AS(Operon::ParseShapeBoundMode("not-a-mode"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeBoundMode("interval,affine"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeBoundMode("bisected"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeBoundMode("affine,bisected"), std::invalid_argument);
}

TEST_CASE("SetBoundMode rejects invalid combinations the same way ParseShapeBoundMode does", "[shape-constraints]")
{
    // A programmatically-constructed ShapeBoundMode bypasses the string
    // parser entirely -- SetBoundMode must enforce the same invariants
    // (via ValidateShapeBoundMode) rather than silently accepting a mode
    // whose Bisected flag then gets ignored downstream.
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -100 }, Operon::Scalar { 100 } } });

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    CHECK_THROWS_AS(sce.SetBoundMode(ShapeBoundMode::Bisected), std::invalid_argument);
    CHECK_THROWS_AS(sce.SetBoundMode(ShapeBoundMode::Affine | ShapeBoundMode::Bisected), std::invalid_argument);
    CHECK_THROWS_AS(sce.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Affine), std::invalid_argument);
    CHECK_THROWS_AS(sce.SetBoundMode(static_cast<ShapeBoundMode>(1U << 3U)), std::invalid_argument);
    CHECK_NOTHROW(sce.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected));
    CHECK(sce.BoundMode() == (ShapeBoundMode::Interval | ShapeBoundMode::Bisected));

    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs);
    CHECK_THROWS_AS(sve.SetBoundMode(ShapeBoundMode::Bisected), std::invalid_argument);
    CHECK_NOTHROW(sve.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected));
    CHECK(sve.BoundMode() == (ShapeBoundMode::Interval | ShapeBoundMode::Bisected));
}

TEST_CASE("ShapeConstrainedEvaluator - bisected interval tightens a dependency-problem bound", "[shape-constraints]")
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
    auto tree = InfixParser::Parse("(X1 - 1) * (X1 - 1)", ds);
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

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);
    shapeEval.SetBoundOptions(
        { .BisectionDepth = 4 }); // Exceeds one AVX2 float batch; every batch needs fresh lane bounds.

    shapeEval.SetBoundMode(ShapeBoundMode::Interval);
    auto const plain = shapeEval.Measure(tree);
    REQUIRE(plain.Measurements.size() == 1);
    REQUIRE(plain.Measurements[0].Bound.has_value());
    auto const [plo, phi] = *plain.Measurements[0].Bound;

    shapeEval.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
    auto const bisected = shapeEval.Measure(tree);
    REQUIRE(bisected.Measurements.size() == 1);
    REQUIRE(bisected.Measurements[0].Bound.has_value());
    auto const [blo, bhi] = *bisected.Measurements[0].Bound;

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

TEST_CASE("ShapeConstrainedEvaluator - bisected interval accepts a model naive interval wrongly rejects",
    "[shape-constraints]")
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
    auto tree = InfixParser::Parse("(X1 - 1) * (X1 - 1)", ds);
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

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);

    shapeEval.SetBoundMode(ShapeBoundMode::Interval);
    CHECK_FALSE(shapeEval.Feasible(tree));

    shapeEval.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
    CHECK(shapeEval.Feasible(tree));
}

namespace {

    // Scalar (non-SIMD) leaf-by-leaf reference for the production wide<T>-batched BisectedIntervalBound: for a single
    // referenced axis, the same widest-axis-pick + uniform-split algorithm as the wide<T>-batched path, every leaf
    // evaluated through IntervalEvaluator<Operon::Scalar> in a plain loop -- an independent cross-check of the SIMD
    // path. For more than one referenced axis, production itself evaluates scalar-only (no wide<T> arithmetic has a
    // soundness proof yet for a multi-axis endpoint pattern), so this mirrors production's own balanced greedy-widest-
    // axis schedule leaf-for-leaf; this is the CI-active twin of the soundness cross-check that until now lived only
    // in test/source/performance/shape_bisection.cpp (excluded from ctest via "~[performance]").
    auto ScalarBisectedBound(Operon::Tree const& tree, Operon::IntervalEvaluator<Operon::Scalar>::DomainMap const& dom,
        int depth) -> std::pair<Operon::Scalar, Operon::Scalar>
    {
        using IE = Operon::IntervalEvaluator<Operon::Scalar>;
        auto const directBound = [&]() -> std::pair<Operon::Scalar, Operon::Scalar> {
            IE ie(&tree, dom);
            auto const iv = ie.Evaluate(tree.GetCoefficients());
            return { iv.inf(), iv.sup() };
        };
        if (depth <= 0) {
            return directBound();
        }

        std::vector<Operon::Hash> axes;
        std::vector<Operon::Scalar> widths;
        for (auto const& n : tree.Nodes()) {
            if (!n.IsVariable() || std::ranges::find(axes, n.HashValue) != axes.end()) {
                continue;
            }
            auto const it = dom.find(n.HashValue);
            if (it == dom.end()) {
                continue;
            }
            auto const width = it->second.second - it->second.first;
            if (width <= Operon::Scalar { 0 }) {
                continue;
            }
            axes.push_back(n.HashValue);
            widths.push_back(width);
        }
        if (axes.empty()) {
            return directBound();
        }

        // Mirrors BisectedIntervalBound's own materially lower cap for the
        // unbatched multi-axis sweep.
        constexpr int MaxMultiAxisBisectionDepth = 12;
        auto const effectiveDepth = axes.size() > 1 ? std::min(depth, MaxMultiAxisBisectionDepth) : depth;

        std::vector<std::size_t> schedule;
        schedule.reserve(static_cast<std::size_t>(effectiveDepth));
        for (int level = 0; level < effectiveDepth; ++level) {
            auto selected = std::size_t { 0 };
            for (std::size_t axis = 1; axis < axes.size(); ++axis) {
                if (widths[axis] > widths[selected]) {
                    selected = axis;
                }
            }
            schedule.push_back(selected);
            widths[selected] /= Operon::Scalar { 2 };
        }

        auto const nLeaves = std::size_t { 1 } << effectiveDepth;
        std::vector<int> splits(axes.size());
        for (auto axis : schedule) {
            ++splits[axis];
        }
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
            auto const seg = ie.Evaluate(coeff);
            resLo = std::min(resLo, seg.inf());
            resHi = std::max(resHi, seg.sup());
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
    auto tree = Operon::InfixParser::Parse("sin(X1) + cos(X1) * X1 ^ 2 + X2", ds);
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
    auto const packed = evaluator.TryEvaluate(tree.GetCoefficients(), overrides);
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
        auto const expected = scalar.TryEvaluate(tree.GetCoefficients());
        REQUIRE(expected);
        CHECK(lower[lane] == Catch::Approx(expected->inf()).margin(1e-4F));
        CHECK(upper[lane] == Catch::Approx(expected->sup()).margin(1e-4F));
    }
}

// The SIMD-vs-scalar soundness cross-check from the performance sweep,
// runnable in CI: the production wide<T>-batched bisected bound must match
// the scalar leaf-by-leaf reference over the same expressions and depths.
TEST_CASE("Bisected interval bound agrees with a scalar leaf-by-leaf reference", "[shape-constraints][bisection]")
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
        auto tree = Operon::InfixParser::Parse(ec.expr, ds);

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

        Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);
        shapeEval.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);

        for (int depth = 0; depth <= 6; ++depth) {
            shapeEval.SetBoundOptions({ .BisectionDepth = depth });
            auto const r = shapeEval.Measure(tree);
            REQUIRE(r.Measurements.size() == 1);
            REQUIRE(r.Measurements[0].Bound.has_value());
            auto const [simdLo, simdHi] = *r.Measurements[0].Bound;
            auto const [scalarLo, scalarHi] = ScalarBisectedBound(tree, dom, depth);
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
TEST_CASE("Bisected interval endpoints enclose the split domain", "[shape-constraints][bisection]")
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
    auto tree = InfixParser::Parse("X1", ds);

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

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);
    shapeEval.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
    shapeEval.SetBoundOptions({ .BisectionDepth = 3 });

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
// bisected bound then covered only [-1, ~8.45], excluding the negative
// part of the domain the scalar mode still accounts for. A batch with an
// empty or nonfinite lane must be rejected wholesale and fall back to the
// whole-box direct bound, matching plain Interval mode exactly.
TEST_CASE("Bisected interval falls back to the direct bound when a sub-box is out of domain",
    "[shape-constraints][bisection]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::Parse("sqrt(X1) + X1", ds);

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

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);

    shapeEval.SetBoundMode(ShapeBoundMode::Interval);
    auto const plain = shapeEval.Measure(tree);
    REQUIRE(plain.Measurements.size() == 1);
    REQUIRE(plain.Measurements[0].Bound.has_value());
    auto const [plo, phi] = *plain.Measurements[0].Bound;

    shapeEval.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
    shapeEval.SetBoundOptions({ .BisectionDepth = 3 });
    auto const bisected = shapeEval.Measure(tree);
    REQUIRE(bisected.Measurements.size() == 1);
    REQUIRE(bisected.Measurements[0].Bound.has_value());
    auto const [blo, bhi] = *bisected.Measurements[0].Bound;

    INFO("plain=[" << plo << "," << phi << "] bisected=[" << blo << "," << bhi << "]");
    CHECK(static_cast<double>(blo) == static_cast<double>(plo));
    CHECK(static_cast<double>(bhi) == static_cast<double>(phi));
}
// f(X1) = 1 / X1 over [-2, 6] has a nonfinite direct interval because the
// denominator spans zero. SIMD bisection produces nonfinite lanes for the
// slices touching zero; those lanes must force the same direct result rather
// than being silently omitted from the lane-wise hull.
TEST_CASE(
    "Bisected interval falls back to the direct bound when a sub-box is unbounded", "[shape-constraints][bisection]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::Parse("1 / X1", ds);

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

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);
    shapeEval.SetBoundMode(ShapeBoundMode::Interval);
    auto const plain = shapeEval.Measure(tree);
    REQUIRE(plain.Measurements.size() == 1);
    CHECK_FALSE(plain.Measurements[0].Certified);
    CHECK_FALSE(plain.Feasible);
    CHECK(plain.Violation == Catch::Approx(1.0));

    shapeEval.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
    shapeEval.SetBoundOptions({ .BisectionDepth = 3 });
    auto const bisected = shapeEval.Measure(tree);
    REQUIRE(bisected.Measurements.size() == 1);
    CHECK_FALSE(bisected.Measurements[0].Certified);
    CHECK_FALSE(bisected.Feasible);
    CHECK(bisected.Violation == Catch::Approx(plain.Violation));
}

// User-registered interval rules are scalar-only. The bisection preflight
// must reject this tree before it creates a wide local or enters the wide
// evaluator, then return the same whole-box direct bound as plain Interval
// mode.
TEST_CASE(
    "Bisected interval takes the scalar path for a user function with no wide rule", "[shape-constraints][bisection]")
{
    auto const hash = Operon::Hasher {}("bisected_user_recipx");
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

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);

    shapeEval.SetBoundMode(ShapeBoundMode::Interval);
    auto const plain = shapeEval.Measure(tree);
    REQUIRE(plain.Measurements.size() == 1);
    REQUIRE(plain.Measurements[0].Bound.has_value());
    auto const [plo, phi] = *plain.Measurements[0].Bound;

    shapeEval.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
    shapeEval.SetBoundOptions({ .BisectionDepth = 3 });
    auto const bisected = shapeEval.Measure(tree);
    REQUIRE(bisected.Measurements.size() == 1);
    REQUIRE(bisected.Measurements[0].Bound.has_value());
    auto const [blo, bhi] = *bisected.Measurements[0].Bound;

    // 1/[1,4] = [0.25, 1]
    CHECK(static_cast<double>(plo) == Catch::Approx(0.25).margin(1e-6));
    CHECK(static_cast<double>(phi) == Catch::Approx(1.0).margin(1e-6));
    CHECK(static_cast<double>(blo) == static_cast<double>(plo));
    CHECK(static_cast<double>(bhi) == static_cast<double>(phi));
}

// Both bisection-depth knobs are validated by the SetBoundOptions setters
// of both evaluator classes (mirroring SetBoundMode's contract), and a
// options change drops the cached feasibility/measurement entries that
// were computed under the previous depths -- the memo key covers the bound
// mode but not the options, so a stale cache would keep answering with the
// old bisection depth's bound.
TEST_CASE("SetBoundOptions validates bisection depths and invalidates cached measurements", "[shape-constraints]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i);
        data(static_cast<Eigen::Index>(i), 1) = Operon::Scalar { 0 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::Parse("X1 - X1", ds); // plain interval: [-8, 8]; bisected depth d: [-h, h], h = 8/2^d

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

    Operon::ShapeConstrainedEvaluator sce(&nmse, &dtable, cs);
    Operon::ShapeViolationEvaluator sve(&problem, &dtable, cs);

    // (a) validation, both classes, both knobs.
    CHECK_THROWS_AS(sce.SetBoundOptions({ .BisectionDepth = -1 }), std::invalid_argument);
    CHECK_THROWS_AS(sce.SetBoundOptions({ .BisectionDepth = 25 }), std::invalid_argument);
    CHECK_THROWS_AS(sce.SetBoundOptions({ .AffineBisectionMaxDepth = -1 }), std::invalid_argument);
    CHECK_THROWS_AS(sce.SetBoundOptions({ .AffineBisectionMaxDepth = 25 }), std::invalid_argument);
    CHECK_THROWS_AS(sve.SetBoundOptions({ .BisectionDepth = -1 }), std::invalid_argument);
    CHECK_THROWS_AS(sve.SetBoundOptions({ .AffineBisectionMaxDepth = 25 }), std::invalid_argument);
    CHECK_NOTHROW(sce.SetBoundOptions({ .BisectionDepth = 0 }));
    CHECK_NOTHROW(sce.SetBoundOptions({ .BisectionDepth = 20 }));
    CHECK_NOTHROW(sve.SetBoundOptions({ .AffineBisectionMaxDepth = 20 }));
    // An options value that failed validation must not have been applied.
    CHECK(sce.BoundOptions().BisectionDepth == 20);

    // (b) cache invalidation: Feasible()/RawViolation() answers computed
    // under depth 3 must be recomputed after the depth changes to 5.
    sce.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
    sve.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);

    sce.SetBoundOptions({ .BisectionDepth = 3 });
    CHECK_FALSE(sce.Feasible(tree)); // [-1, 1] vs [-0.25, 0.25]
    sve.SetBoundOptions({ .BisectionDepth = 3 });
    CHECK(static_cast<double>(sve.RawViolation(tree)) == Catch::Approx(1.5).margin(1e-6)); // 0.75 + 0.75

    sce.SetBoundOptions({ .BisectionDepth = 5 });
    CHECK(sce.Feasible(tree)); // [-0.25, 0.25] -- recomputed, not the stale depth-3 entry
    sve.SetBoundOptions({ .BisectionDepth = 5 });
    CHECK(static_cast<double>(sve.RawViolation(tree)) == Catch::Approx(0.0).margin(1e-12));
}

TEST_CASE("ShapeConstrainedEvaluator - balanced bisection resolves multi-axis dependency", "[shape-constraints]")
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
    auto tree = InfixParser::Parse("X1 * X2 - X1 * X2", ds);
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

    Operon::ShapeConstrainedEvaluator shapeEval(&nmse, &dtable, cs);
    shapeEval.SetBoundMode(ShapeBoundMode::Interval | ShapeBoundMode::Bisected);
    shapeEval.SetBoundOptions({ .BisectionDepth = 6 });
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

TEST_CASE("Shape cache memo key includes a reference target", "[shape-constraints]")
{
    Fixture fx;
    auto const x1 = fx.ds.GetVariable("X1").value().Hash;
    auto const x2 = fx.ds.GetVariable("X2").value().Hash;
    auto x = Node(NodeType::Variable, x1);
    auto y = Node(NodeType::Variable, x2);
    auto add = Node::Function(static_cast<Hash>(BuiltinOp::Add), 2);
    auto const refX = Tree({ x, y, Node::Ref(0), add }).UpdateNodes();
    auto const refY = Tree({ x, y, Node::Ref(1), add }).UpdateNodes();

    CHECK(detail::HashTreeForMemo(refX) != detail::HashTreeForMemo(refY));
    CHECK(detail::HashTreeForMemo(refX, static_cast<Hash>(ShapeBoundMode::Combined))
        != detail::HashTreeForMemo(refX, static_cast<Hash>(ShapeBoundMode::Interval)));
    CHECK(detail::HashTreeForMemo(refX, static_cast<Hash>(ShapeBoundMode::Interval))
        != detail::HashTreeForMemo(refX, static_cast<Hash>(ShapeBoundMode::Interval | ShapeBoundMode::Bisected)));
}

TEST_CASE("ShapeConstrainedEvaluator - negative linear scale flips derivative constraints", "[shape-constraints]")
{
    Fixture fx;
    auto negated = InfixParser::Parse("X2 - X1", fx.ds);

    auto scaling = Operon::FitLinearScaling(negated, fx.problem, fx.dtable, fx.problem.TrainingRange());
    REQUIRE(scaling);
    CHECK(scaling->Scale < 0.0);
    CHECK(scaling->Offset == Catch::Approx(0.0).margin(1e-5));

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = -1, .Bound = std::nullopt });

    Operon::ShapeConstrainedEvaluator scaled(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(scaled.Feasible(negated));

    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstrainedEvaluator raw(&fx.nmse, &fx.dtable, cs);
    CHECK(raw.Feasible(negated));
}

TEST_CASE("ShapeConstrainedEvaluator - negative linear scale swaps derivative bound endpoints", "[shape-constraints]")
{
    Fixture fx;
    auto negated = InfixParser::Parse("X2 - X1", fx.ds);

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

    Operon::ShapeConstrainedEvaluator scaled(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(scaled.Feasible(negated));

    auto const measurement = scaled.Measure(negated);
    REQUIRE(measurement.Measurements.size() == 1);
    CHECK(measurement.Measurements[0].Violation > 0.0F);

    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstrainedEvaluator raw(&fx.nmse, &fx.dtable, cs);
    CHECK(raw.Feasible(negated));
}

TEST_CASE("ShapeConstrainedEvaluator - offset shifts identity bound constraints", "[shape-constraints]")
{
    constexpr auto nrow = std::size_t { 5 };
    constexpr auto ncol = std::size_t { 2 };
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i) / static_cast<Operon::Scalar>(nrow - 1);
        data(static_cast<Eigen::Index>(i), 1) = data(static_cast<Eigen::Index>(i), 0) + Operon::Scalar { 10 };
    }
    Operon::Dataset ds(gsl::not_null { data.data() }, nrow, ncol);
    auto tree = InfixParser::Parse("X1", ds);
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

    Operon::ShapeConstrainedEvaluator scaled(&nmse, &dtable, cs);
    CHECK_FALSE(scaled.Feasible(tree));

    problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstrainedEvaluator raw(&nmse, &dtable, cs);
    CHECK(raw.Feasible(tree));
}

TEST_CASE(
    "ShapeConstrainedEvaluator and ShapeViolationEvaluator report directly fitted scaled bounds", "[shape-constraints]")
{
    Fixture fx;
    auto tree = InfixParser::Parse("2 * (X1 - X2) + 3", fx.ds);
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

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    auto const constrained = sce.Measure(tree);
    REQUIRE(constrained.Measurements.size() == 1);
    REQUIRE(constrained.Measurements[0].Bound);
    CHECK(constrained.Measurements[0].Bound->first == Catch::Approx(expectedLo));
    CHECK(constrained.Measurements[0].Bound->second == Catch::Approx(expectedHi));

    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs);
    auto const violation = sve.Measure(tree);
    REQUIRE(violation.Measurements.size() == 1);
    REQUIRE(violation.Measurements[0].Bound);
    CHECK(violation.Measurements[0].Bound->first == Catch::Approx(expectedLo));
    CHECK(violation.Measurements[0].Bound->second == Catch::Approx(expectedHi));
}

TEST_CASE(
    "ShapeConstrainedEvaluator - derivative through an unsupported op is not falsely feasible", "[shape-constraints]")
{
    // f(x) = abs(X1): Abs has no registered symbolic derivative rule
    // (tree_diff.cpp intentionally leaves it unregistered, non-smooth).
    // A sign constraint on d/dX1 must NOT be certified feasible just
    // because Deriv() falls back to "zero" for the unsupported op --
    // that fallback is sound for LM coefficient fitting but not for a
    // feasibility certificate (a real, previously-latent soundness gap).
    Fixture fx;
    auto tree = InfixParser::Parse("abs(X1)", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(sce.Feasible(tree));
}

TEST_CASE("ShapeConstrainedEvaluator - identically-zero derivative satisfies either sign", "[shape-constraints]")
{
    // f(X2) = X2 does not reference X1 at all -> d/dX1 is identically
    // zero (BuildVariableGradientDag's NoGrad root path, no
    // AffineEvaluator call needed for it), which must satisfy a
    // non-decreasing (0 >= 0) or non-increasing (0 <= 0) constraint.
    Fixture fx;
    auto tree = InfixParser::Parse("X2", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    Operon::ShapeConstrainedEvaluator nonDecreasing(&fx.nmse, &fx.dtable, cs);
    CHECK(nonDecreasing.Feasible(tree));

    cs.Constraints[0].Sign = -1;
    Operon::ShapeConstrainedEvaluator nonIncreasing(&fx.nmse, &fx.dtable, cs);
    CHECK(nonIncreasing.Feasible(tree));
}

TEST_CASE("ShapeConstrainedEvaluator - unknown variable in domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("NotAColumn", std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } });
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable, cs), std::invalid_argument);
}

TEST_CASE("ShapeConstrainedEvaluator - constraint variable missing from domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    // No domains entry for X1 at all.
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable, cs), std::invalid_argument);
}

TEST_CASE("ShapeConstrainedEvaluator - problem input variable missing from domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -4 }, Operon::Scalar { 4 } } });

    CHECK_THROWS_WITH(
        Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable, cs), Catch::Matchers::ContainsSubstring("X2"));
    CHECK_THROWS_WITH(
        Operon::ShapeViolationEvaluator(&fx.problem, &fx.dtable, cs), Catch::Matchers::ContainsSubstring("X2"));
}

TEST_CASE("ShapeConstrainedEvaluator - domain error (e.g. division by zero-containing interval) is treated as "
          "infeasible, not a crash",
    "[shape-constraints]")
{
    // f(x) = 1 / X1, with X1's domain spanning zero -- AffineEvaluator
    // throws std::invalid_argument for this (affine_form::inv is
    // undefined over an interval containing zero); GP generates trees
    // like this constantly, so Feasible() must swallow it as "can't be
    // certified feasible" rather than letting a run crash.
    Fixture fx;
    auto tree = InfixParser::Parse("1 / X1", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -100 }, Operon::Scalar { 100 } } });

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    auto const measurement = sce.Measure(tree);
    REQUIRE(measurement.Measurements.size() == 1);
    CHECK_FALSE(measurement.Measurements[0].Certified);
    CHECK(measurement.Measurements[0].Violation == Catch::Approx(1.0));
    CHECK_FALSE(sce.Feasible(tree));
}

TEST_CASE("ShapeConstrainedEvaluator - a throwing user-registered rule is treated as infeasible, not a crash",
    "[shape-constraints]")
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

    auto const assertDegradesToUncertified
        = [&](Operon::ShapeConstrainedEvaluator const& sce, Operon::Tree const& tree) {
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

    SECTION("throwing affine rule, combined mode: the catch around ae.TryEvaluate degrades it")
    {
        auto const hash = Operon::Hasher {}("shape_throw_affine_rule");
        RegisterUnaryAffine<Scalar>(hash,
            [](AffineEvaluator<Scalar>::Context const&, AffineEvaluator<Scalar>::Affine const&)
                -> AffineEvaluator<Scalar>::Affine { throw std::runtime_error("user affine rule failed"); });
        Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
        sce.SetBoundMode(ShapeBoundMode::Combined);
        assertDegradesToUncertified(sce, makeTree(hash));
    }

    SECTION("throwing interval rule, combined mode: TryEvaluate's own internal catch degrades it")
    {
        auto const hash = Operon::Hasher {}("shape_throw_interval_rule_combined");
        RegisterUnaryInterval<Scalar>(
            hash, [](IntervalEvaluator<Scalar>::Interval const&) -> IntervalEvaluator<Scalar>::Interval {
                throw std::runtime_error("user interval rule failed");
            });
        Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
        sce.SetBoundMode(ShapeBoundMode::Combined);
        assertDegradesToUncertified(sce, makeTree(hash));
    }

    SECTION("throwing interval rule, interval mode: TryEvaluate's own internal catch degrades it")
    {
        auto const hash = Operon::Hasher {}("shape_throw_interval_rule_interval_mode");
        RegisterUnaryInterval<Scalar>(
            hash, [](IntervalEvaluator<Scalar>::Interval const&) -> IntervalEvaluator<Scalar>::Interval {
                throw std::runtime_error("user interval rule failed");
            });
        Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
        sce.SetBoundMode(ShapeBoundMode::Interval);
        assertDegradesToUncertified(sce, makeTree(hash));
    }
}

TEST_CASE("ShapeConstrainedEvaluator - certifies constant integer powers of a negative base", "[shape-constraints]")
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
    auto tree = InfixParser::Parse("(-0.91) ^ 2", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } } });

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    auto const measurement = sce.Measure(tree);
    REQUIRE(measurement.Measurements.size() == 1);
    CHECK(measurement.Measurements[0].Certified);
    REQUIRE(measurement.Measurements[0].Bound);
    CHECK(measurement.Measurements[0].Bound->first == Catch::Approx(0.8281).margin(1e-3));
    CHECK(measurement.Measurements[0].Bound->second == Catch::Approx(0.8281).margin(1e-3));
    CHECK(measurement.Feasible);
    CHECK(sce.Feasible(tree));
}

TEST_CASE("ShapeConstrainedEvaluator - a NaN bound endpoint (Scale==0 times an unbounded raw interval) is not falsely "
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

    auto tree = InfixParser::Parse("exp(X1)", ds);
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

    Operon::ShapeConstrainedEvaluator sce(&nmse, &dtable, cs);
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

    SECTION("GP accepts only hard-reject, penalty, feasibility-first, and allowed GP combinations")
    {
        valid(E::HardReject, false);
        valid(E::Penalty, false);
        valid(E::FeasibilityFirst, false);
        valid(E::HardReject | E::FeasibilityFirst, false);
        valid(E::Penalty | E::FeasibilityFirst, false);

        invalid(E::HardReject | E::Penalty, false);
        invalid(E::ExtraObjective, false);
        invalid(E::Penalty | E::ExtraObjective, false);
        invalid(E::HardReject | E::ExtraObjective, false);
        invalid(E::FeasibilityFirst | E::ExtraObjective, false);
        invalid(E::HardReject | E::Penalty | E::ExtraObjective, false);
    }

    SECTION("NSGA2 accepts hard-reject, penalty, extra-objective, and penalty plus extra-objective")
    {
        valid(E::HardReject, true);
        valid(E::Penalty, true);
        valid(E::ExtraObjective, true);
        valid(E::Penalty | E::ExtraObjective, true);

        invalid(E::HardReject | E::Penalty, true);
        invalid(E::HardReject | E::ExtraObjective, true);
        invalid(E::FeasibilityFirst, true);
        invalid(E::HardReject | E::FeasibilityFirst, true);
        invalid(E::Penalty | E::FeasibilityFirst, true);
        invalid(E::ExtraObjective | E::FeasibilityFirst, true);
        invalid(E::HardReject | E::Penalty | E::ExtraObjective, true);
    }

    CHECK(Operon::ValidatePolicy(
        { .Enforcement = E::None, .UnknownViolation = Operon::Scalar { 1 }, .PenaltyWeight = Operon::Scalar { 1 } },
        false));
    CHECK(Operon::ValidatePolicy({ .Enforcement = static_cast<E>(1U << 9U),
                                     .UnknownViolation = Operon::Scalar { 1 },
                                     .PenaltyWeight = Operon::Scalar { 1 } },
        false));
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
    CHECK(Operon::ParseShapeEnforcement("feasibility-first") == E::FeasibilityFirst);
    CHECK(Operon::ParseShapeEnforcement("penalty,feasibility-first") == (E::Penalty | E::FeasibilityFirst));
    CHECK(Operon::ParseShapeEnforcement("penalty,extra-objective") == (E::Penalty | E::ExtraObjective));
    CHECK(Operon::ParseShapeEnforcement("hard-reject,hard-reject") == E::HardReject);

    CHECK_THROWS_AS(Operon::ParseShapeEnforcement(""), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("penalty,"), std::invalid_argument);
    CHECK_THROWS_AS(Operon::ParseShapeEnforcement("unknown"), std::invalid_argument);
}

TEST_CASE("Shape constraint CLI-adjacent composition works for representative enforcement modes", "[shape-constraints]")
{
    Fixture fx;
    auto const path = WriteShapeConfig("composition", R"json({
        "domains": { "X1": [1, 5], "X2": [1, 5] },
        "constraints": [ { "op": "derivative", "variable": "X1", "order": 1, "sign": 1 } ]
    })json");
    auto result = Operon::LoadShapeConstraints(path.string());
    REQUIRE(result); // outer Cli::Result
    REQUIRE(*result); // inner std::optional
    auto const& loaded = **result;

    auto requireValid = [](Operon::ShapeConstraintPolicy const& policy, bool isNsga2) {
        if (auto error = Operon::ValidatePolicy(policy, isNsga2)) {
            throw std::invalid_argument(*error);
        }
    };

    SECTION("GP hard-reject constructs the rejecting evaluator")
    {
        Operon::ShapeConstraintPolicy policy { .Enforcement = Operon::ParseShapeEnforcement("hard-reject"),
            .UnknownViolation = Operon::Scalar { 1 },
            .PenaltyWeight = Operon::Scalar { 1 } };
        REQUIRE_NOTHROW(requireValid(policy, false));
        Operon::ShapeConstrainedEvaluator gated(&fx.nmse, &fx.dtable, loaded);
        CHECK(gated.Feasible(fx.tree));
    }

    SECTION("GP penalty constructs the violation evaluator and summed aggregate")
    {
        Operon::ShapeConstraintPolicy policy {
            .Enforcement = Operon::ParseShapeEnforcement("penalty"),
            .UnknownViolation = Operon::Scalar { 2 },
            .PenaltyWeight = Operon::Scalar { 3 },
        };
        REQUIRE_NOTHROW(requireValid(policy, false));
        Operon::ShapeViolationEvaluator violation(
            &fx.problem, &fx.dtable, loaded, policy.PenaltyWeight, policy.UnknownViolation);
        Operon::MultiEvaluator aggregate(&fx.problem);
        aggregate.Add(&fx.nmse);
        aggregate.Add(&violation);
        aggregate.SetAggregateType(Operon::MultiEvaluator::AggregateType::Sum);
        CHECK(violation.RawViolation(fx.tree) == Catch::Approx(0.0));
    }

    SECTION("GP feasibility-first constructs the comparator-side violation evaluator")
    {
        Operon::ShapeConstraintPolicy policy { .Enforcement = Operon::ParseShapeEnforcement("feasibility-first"),
            .UnknownViolation = Operon::Scalar { 1 },
            .PenaltyWeight = Operon::Scalar { 1 } };
        REQUIRE_NOTHROW(requireValid(policy, false));
        fx.problem.SetLinearScalingEnabled(false);
        Operon::ShapeViolationEvaluator violation(
            &fx.problem, &fx.dtable, loaded, Operon::Scalar { 1 }, policy.UnknownViolation);
        Operon::FeasibilityFirstComparison comp(
            [&violation](Operon::Tree const& t) { return violation.Measure(t).Feasible; });
        auto feasible = Fixture::MakeIndividual(fx.tree);
        feasible.Fitness = { 10.0F };
        auto infeasible = Fixture::MakeIndividual(InfixParser::Parse("X2 - X1", fx.ds));
        infeasible.Fitness = { 0.1F };
        CHECK(comp(feasible, infeasible));
    }

    SECTION("NSGA2 extra-objective constructs the added shape objective")
    {
        Operon::ShapeConstraintPolicy policy { .Enforcement = Operon::ParseShapeEnforcement("extra-objective"),
            .UnknownViolation = Operon::Scalar { 1 },
            .PenaltyWeight = Operon::Scalar { 1 } };
        REQUIRE_NOTHROW(requireValid(policy, true));
        Operon::ShapeViolationEvaluator extra(
            &fx.problem, &fx.dtable, loaded, Operon::Scalar { 1 }, policy.UnknownViolation);
        Operon::MultiEvaluator objectives(&fx.problem);
        objectives.Add(&fx.nmse);
        objectives.Add(&extra);
        CHECK(extra.RawViolation(fx.tree) == Catch::Approx(0.0));
    }

    SECTION("invalid CLI mode combination is rejected by the same validation step")
    {
        Operon::ShapeConstraintPolicy policy { .Enforcement = Operon::ParseShapeEnforcement("penalty,extra-objective"),
            .UnknownViolation = Operon::Scalar { 1 },
            .PenaltyWeight = Operon::Scalar { 1 } };
        CHECK_THROWS_AS(requireValid(policy, false), std::invalid_argument);
    }
}

TEST_CASE("ShapeViolationEvaluator - sign constraint violation magnitudes", "[shape-constraints]")
{
    Fixture fx;
    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    auto bad = InfixParser::Parse("X2 - X1", fx.ds);
    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs, Operon::Scalar { 3 });
    auto m = sve.Measure(bad);
    REQUIRE(m.Measurements.size() == 1);
    CHECK_FALSE(m.Feasible);
    CHECK(m.Measurements[0].Certified);
    CHECK(m.Measurements[0].Violation == Catch::Approx(1.0));
    CHECK(sve.RawViolation(bad) == Catch::Approx(1.0));

    auto ind = Fixture::MakeIndividual(bad);
    std::vector<Operon::Scalar> buf(fx.problem.TrainingRange().Size());
    auto fit = sve(fx.rng, ind, buf);
    REQUIRE(fit.size() == 1);
    CHECK(fit[0] == Catch::Approx(3.0));

    cs.Constraints[0].Sign = -1;
    Operon::ShapeViolationEvaluator mirror(&fx.problem, &fx.dtable, cs);
    CHECK(mirror.RawViolation(fx.tree) == Catch::Approx(1.0));
}

TEST_CASE("ShapeViolationEvaluator - bound constraint violation magnitude", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } } });

    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs);
    auto m = sve.Measure(fx.tree);
    REQUIRE(m.Measurements.size() == 1);
    REQUIRE(m.Measurements[0].Bound);
    CHECK(m.Measurements[0].Bound->first == Catch::Approx(-4.0));
    CHECK(m.Measurements[0].Bound->second == Catch::Approx(4.0));
    CHECK(m.Violation == Catch::Approx(6.0));
    CHECK_FALSE(m.Feasible);
}

TEST_CASE(
    "ShapeViolationEvaluator - identity, first-derivative, and second-derivative measurements", "[shape-constraints]")
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

    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs);
    auto m = sve.Measure(fx.tree);
    REQUIRE(m.Measurements.size() == 3);
    CHECK(m.Feasible);
    CHECK(m.Violation == Catch::Approx(0.0));
    CHECK(m.Measurements[0].Certified);
    CHECK(m.Measurements[1].Certified);
    CHECK(m.Measurements[2].Certified);

    auto square = InfixParser::Parse("X1 * X1", fx.ds);
    Operon::ShapeConstraintSet secondDerivativeOnly;
    secondDerivativeOnly.Domains = cs.Domains;
    secondDerivativeOnly.Constraints.push_back(
        { .Op = ShapeConstraintOp::SecondDerivative, .Variable = "X1", .Sign = -1, .Bound = std::nullopt });
    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeViolationEvaluator violated(&fx.problem, &fx.dtable, secondDerivativeOnly);
    auto v = violated.Measure(square);
    CHECK_FALSE(v.Feasible);
    CHECK(v.Violation == Catch::Approx(2.0));
}

TEST_CASE("ShapeViolationEvaluator - Measure() uses stable cached results across Prepare()", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { Operon::Scalar { -1 }, Operon::Scalar { 1 } } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs);
    auto const first = sve.Measure(fx.tree);
    auto const second = sve.Measure(fx.tree);

    auto checkSame = [](auto const& lhs, auto const& rhs) {
        REQUIRE(lhs.Measurements.size() == rhs.Measurements.size());
        CHECK(lhs.Feasible == rhs.Feasible);
        CHECK(lhs.Violation == rhs.Violation);
        for (std::size_t i = 0; i < lhs.Measurements.size(); ++i) {
            CHECK(lhs.Measurements[i].Certified == rhs.Measurements[i].Certified);
            CHECK(lhs.Measurements[i].Bound == rhs.Measurements[i].Bound);
            CHECK(lhs.Measurements[i].Violation == rhs.Measurements[i].Violation);
        }
    };
    checkSame(first, second);

    std::vector<Operon::Individual> pop { Fixture::MakeIndividual(fx.tree) };
    sve.Prepare(pop);
    checkSame(first, sve.Measure(fx.tree));
}

TEST_CASE("ShapeViolationEvaluator - unknown violation and empty constraint set", "[shape-constraints]")
{
    Fixture fx;
    auto unknownTree = InfixParser::Parse("abs(X1)", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back(
        { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });

    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs, Operon::Scalar { 1 }, Operon::Scalar { 2.5 });
    auto m = sve.Measure(unknownTree);
    REQUIRE(m.Measurements.size() == 1);
    CHECK_FALSE(m.Measurements[0].Certified);
    CHECK_FALSE(m.Feasible);
    CHECK(m.Violation == Catch::Approx(2.5));

    cs.Constraints.clear();
    Operon::ShapeViolationEvaluator empty(&fx.problem, &fx.dtable, cs);
    auto e = empty.Measure(fx.tree);
    CHECK(e.Feasible);
    CHECK(e.Measurements.empty());
    CHECK(e.Violation == Catch::Approx(0.0));
    CHECK(empty.RawViolation(fx.tree) == Catch::Approx(0.0));
}

TEST_CASE("ShapeConstrainedEvaluator - constructor rejects malformed constraints", "[shape-constraints]")
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
        Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable,
            withOneConstraint(
                { .Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::nullopt })),
        std::invalid_argument);
    // both set
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable,
                        withOneConstraint({ .Op = ShapeConstraintOp::Identity,
                            .Variable = "",
                            .Sign = 1,
                            .Bound = std::pair { Operon::Scalar { 0 }, Operon::Scalar { 1 } } })),
        std::invalid_argument);
    // invalid sign
    CHECK_THROWS_AS(
        Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable,
            withOneConstraint({ .Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = 2, .Bound = std::nullopt })),
        std::invalid_argument);
    // lo > hi
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable,
                        withOneConstraint({ .Op = ShapeConstraintOp::Identity,
                            .Variable = "",
                            .Sign = std::nullopt,
                            .Bound = std::pair { Operon::Scalar { 5 }, Operon::Scalar { 1 } } })),
        std::invalid_argument);
}

TEST_CASE("FeasibilityFirstComparison - feasible precedes infeasible regardless of fitness", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
        .Variable = "X1",
        .Sign = 1,
        .Bound = std::nullopt }); // true: f is non-decreasing in X1

    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    Operon::FeasibilityFirstComparison comp([&sce](Operon::Tree const& t) { return sce.Feasible(t); });

    auto feasibleWorseFit = Fixture::MakeIndividual(fx.tree); // satisfies the constraint
    feasibleWorseFit.Fitness = { 10.0F };

    auto infeasibleBetterFit = InfixParser::Parse("X2 - X1", fx.ds); // violates: non-increasing in X1
    auto infeasibleInd = Fixture::MakeIndividual(infeasibleBetterFit);
    infeasibleInd.Fitness = { 0.01F };

    CHECK(comp(feasibleWorseFit, infeasibleInd)); // feasible wins despite worse fitness
    CHECK_FALSE(comp(infeasibleInd, feasibleWorseFit));

    // equal feasibility -> falls back to the wrapped comparator (fitness order)
    auto feasibleBetterFit = Fixture::MakeIndividual(fx.tree);
    feasibleBetterFit.Fitness = { 1.0F };
    CHECK(comp(feasibleBetterFit, feasibleWorseFit));
}

TEST_CASE("ShapeConstrainedEvaluator - Prepare() populates the feasibility cache correctly", "[shape-constraints]")
{
    // The memoization itself lives in ShapeConstrainedEvaluator::Feasible()
    // now (populated by Prepare() and by Evaluate()), not in
    // FeasibilityFirstComparison -- this checks Prepare()'s cache-fill and
    // cache-clear-and-rebuild behavior stays functionally correct, i.e.
    // doesn't silently return a stale/wrong answer for either the
    // just-prepared population or an unrelated tree asked about later.
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
    cs.Constraints.push_back({ .Op = ShapeConstraintOp::FirstDerivative,
        .Variable = "X1",
        .Sign = 1,
        .Bound = std::nullopt }); // true for X1 - X2

    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);

    auto infeasibleTree = InfixParser::Parse("X2 - X1", fx.ds); // false: non-increasing in X1
    std::vector<Operon::Individual> pop { Fixture::MakeIndividual(fx.tree), Fixture::MakeIndividual(infeasibleTree) };
    sce.Prepare(pop);

    CHECK(sce.Feasible(fx.tree));
    CHECK_FALSE(sce.Feasible(infeasibleTree));

    // A second Prepare() with a different population clears and rebuilds
    // the cache; trees from the first population must still resolve
    // correctly afterward (Feasible() computes fresh on a miss, doesn't
    // require having been in the most recent Prepare() call).
    auto other = InfixParser::Parse("X1", fx.ds);
    std::vector<Operon::Individual> pop2 { Fixture::MakeIndividual(other) };
    sce.Prepare(pop2);
    CHECK(sce.Feasible(other));
    CHECK(sce.Feasible(fx.tree));
    CHECK_FALSE(sce.Feasible(infeasibleTree));
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

        auto tree = InfixParser::Parse(model, ds);
        Operon::ShapeConstrainedEvaluator sce(&nmse, &dtable, loaded);
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
    auto tree = InfixParser::Parse(model, ds);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    Operon::Evaluator<DTable> nmse(&problem, &dtable, Operon::NMSE {});
    Operon::ShapeConstrainedEvaluator sce(&nmse, &dtable, loaded);
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
        auto tree = InfixParser::Parse(c.model, ds);
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
        auto const affineRaw = ae.Evaluate(dtree.GetCoefficients()).to_interval();

        Operon::IntervalEvaluator<Operon::Scalar> ie(
            &dtree, Operon::IntervalEvaluator<Operon::Scalar>::DomainMap { domains });
        auto const intervalRaw = ie.Evaluate(dtree.GetCoefficients());

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

} // namespace Operon::Test
