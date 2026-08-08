// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <stdexcept>
#include <string>

#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/constraint.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/linear_scaling.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "shape_constraints_config.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"

namespace Operon::Test {

namespace {

// f(X1, X2) = X1 - X2 on [1,5]x[1,5], 20 rows. Known monotonicity:
// non-decreasing in X1, non-increasing in X2.
struct Fixture {
    static constexpr auto Nrow { 20 };
    static constexpr auto Ncol { 3 }; // X1, X2, y

    Operon::RandomGenerator rng{0};
    Eigen::Array<Operon::Scalar, -1, -1> data{Nrow, Ncol};
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
                std::generate(col.begin(), col.end(), [&]() -> float { return Operon::Random::Uniform(rng, 1.0F, 5.0F); });
            }
            data.col(Ncol - 1) = data.col(0) - data.col(1);
            return Operon::Dataset(gsl::not_null{data.data()}, Nrow, Ncol);
        }())
        , tree(InfixParser::Parse("X1 - X2", ds))
        , problem(&ds)
        , nmse(&problem, &dtable, Operon::NMSE{})
    {
        problem.SetTrainingRange({0, Nrow});
        problem.SetTestRange({0, Nrow});
        problem.SetTarget("X3");
        problem.SetDefaultInputs(); // Problem::SetTarget doesn't refresh GetInputs() on its own -- redo it now that the target is known, matching how the CLI itself sequences this (operon_gp.cpp: SetTarget then SetInputs).
    }

    static auto MakeIndividual(Operon::Tree const& t) -> Operon::Individual {
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

    auto loaded = Operon::LoadShapeConstraints(path.string());
    REQUIRE(loaded);
    REQUIRE(loaded->Domains.size() == 3);
    CHECK(loaded->Domains.at("X1").first == Catch::Approx(1.0));
    CHECK(loaded->Domains.at("X2").second == Catch::Approx(5.0));
    CHECK(loaded->Domains.at("x2").first == Catch::Approx(-2.0));

    REQUIRE(loaded->Constraints.size() == 5);
    CHECK(loaded->Constraints[0].Op == ShapeConstraintOp::Identity);
    REQUIRE(loaded->Constraints[0].Bound);
    CHECK(loaded->Constraints[0].Bound->first == Catch::Approx(-4.0));
    CHECK(loaded->Constraints[1].Op == ShapeConstraintOp::Identity);
    REQUIRE(loaded->Constraints[1].Sign);
    CHECK(*loaded->Constraints[1].Sign == 1);
    CHECK(loaded->Constraints[2].Op == ShapeConstraintOp::FirstDerivative);
    CHECK(loaded->Constraints[2].Variable == "X1");
    CHECK(loaded->Constraints[3].Op == ShapeConstraintOp::SecondDerivative);
    CHECK(loaded->Constraints[3].Variable == "X2");
    REQUIRE(loaded->Constraints[3].Bound);
    CHECK(loaded->Constraints[4].Op == ShapeConstraintOp::SecondDerivative);
    CHECK(loaded->Constraints[4].Variable == "x2"); // unambiguous variable name ending in '2'
}

TEST_CASE("LoadShapeConstraints handles empty paths and JSON schema errors", "[shape-constraints]")
{
    CHECK_FALSE(Operon::LoadShapeConstraints(""));
    auto const missing = std::filesystem::temp_directory_path() / "operon_shape_constraints_missing_file_this_test_should_not_exist.json";
    std::filesystem::remove(missing);
    CHECK_THROWS_AS(Operon::LoadShapeConstraints(missing.string()), std::runtime_error);
    CHECK_THROWS_AS(Operon::LoadShapeConstraints(WriteShapeConfig("malformed", R"json({"domains":)json").string()), std::runtime_error);

    auto throwsConfig = [](std::string const& name, std::string const& json) {
        CHECK_THROWS_AS(Operon::LoadShapeConstraints(WriteShapeConfig(name, json).string()), std::runtime_error);
    };

    throwsConfig("both_sign_bound", R"json({"constraints":[{"op":"id","sign":1,"bound":[0,1]}]})json");
    throwsConfig("neither_sign_bound", R"json({"constraints":[{"op":"id"}]})json");
    throwsConfig("non_integral_sign", R"json({"constraints":[{"op":"id","sign":1.5}]})json");
    throwsConfig("out_of_range_sign", R"json({"constraints":[{"op":"id","sign":0}]})json");
    throwsConfig("bad_order", R"json({"constraints":[{"op":"derivative","variable":"X1","order":3,"sign":1}]})json");
    throwsConfig("non_integral_order", R"json({"constraints":[{"op":"derivative","variable":"X1","order":1.5,"sign":1}]})json");
    throwsConfig("missing_variable", R"json({"constraints":[{"op":"derivative","order":1,"sign":1}]})json");
    throwsConfig("missing_order", R"json({"constraints":[{"op":"derivative","variable":"X1","sign":1}]})json");
    throwsConfig("bad_domain", R"json({"domains":{"X1":[0,1,2]},"constraints":[{"op":"id","sign":1}]})json");
    throwsConfig("domains_not_object", R"json({"domains":"not an object","constraints":[{"op":"id","sign":1}]})json");
    throwsConfig("constraints_not_array", R"json({"constraints":{}})json");
    throwsConfig("constraint_entry_not_object", R"json({"constraints":[7]})json");
    throwsConfig("bound_not_array", R"json({"constraints":[{"op":"id","bound":7}]})json");
    throwsConfig("non_string_op", R"json({"constraints":[{"op":7,"sign":1}]})json");
}

TEST_CASE("ShapeConstrainedEvaluator - correctly-signed constraints are feasible", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt});  // non-decreasing: true
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X2", .Sign = -1, .Bound = std::nullopt}); // non-increasing: true

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

TEST_CASE("ShapeConstrainedEvaluator - wrongly-signed constraint is rejected with WorstValue", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    // f is actually non-decreasing in X1; asserting the opposite must be rejected.
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = -1, .Bound = std::nullopt});

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
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    // f = X1 - X2 over [1,5]x[1,5] has range [-4, 4]; a [-4,4] bound holds, a [-1,1] bound doesn't.
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-4}, Operon::Scalar{4}}});
    Operon::ShapeConstrainedEvaluator wide(&fx.nmse, &fx.dtable, cs);
    CHECK(wide.Feasible(fx.tree));

    cs.Constraints[0].Bound = std::pair{Operon::Scalar{-1}, Operon::Scalar{1}};
    Operon::ShapeConstrainedEvaluator narrow(&fx.nmse, &fx.dtable, cs);
    CHECK_FALSE(narrow.Feasible(fx.tree));
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
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = -1, .Bound = std::nullopt});

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
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    // Raw d(X2-X1)/dX1 is [-1,-1], satisfying this bound.  After the fitted
    // negative scale is applied to the delivered model, the derivative interval
    // becomes [1,1] via LinearScaling::ApplyToDerivativeInterval's endpoint swap,
    // and the bound-arithmetic violation path must reject it.
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-1.5F}, Operon::Scalar{-0.5F}}});

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
    constexpr auto nrow = std::size_t{5};
    constexpr auto ncol = std::size_t{2};
    Eigen::Array<Operon::Scalar, -1, -1> data(nrow, ncol);
    for (std::size_t i = 0; i < nrow; ++i) {
        data(static_cast<Eigen::Index>(i), 0) = static_cast<Operon::Scalar>(i) / static_cast<Operon::Scalar>(nrow - 1);
        data(static_cast<Eigen::Index>(i), 1) = data(static_cast<Eigen::Index>(i), 0) + Operon::Scalar{10};
    }
    Operon::Dataset ds(gsl::not_null{data.data()}, nrow, ncol);
    auto tree = InfixParser::Parse("X1", ds);
    Operon::Problem problem(&ds);
    problem.SetTrainingRange({0, nrow});
    problem.SetTestRange({0, nrow});
    problem.SetTarget("X2");
    problem.SetDefaultInputs();
    Fixture::DTable dtable;
    Operon::Evaluator<Fixture::DTable> nmse(&problem, &dtable, Operon::NMSE{});

    auto scaling = Operon::FitLinearScaling(tree, problem, dtable, problem.TrainingRange());
    REQUIRE(scaling);
    CHECK(scaling->Scale == Catch::Approx(1.0));
    CHECK(scaling->Offset == Catch::Approx(10.0));

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{0}, Operon::Scalar{1}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{0}, Operon::Scalar{1}}});

    Operon::ShapeConstrainedEvaluator scaled(&nmse, &dtable, cs);
    CHECK_FALSE(scaled.Feasible(tree));

    problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstrainedEvaluator raw(&nmse, &dtable, cs);
    CHECK(raw.Feasible(tree));
}

TEST_CASE("ShapeConstrainedEvaluator and ShapeViolationEvaluator report directly fitted scaled bounds", "[shape-constraints]")
{
    Fixture fx;
    auto tree = InfixParser::Parse("2 * (X1 - X2) + 3", fx.ds);
    auto scaling = Operon::FitLinearScaling(tree, fx.problem, fx.dtable, fx.problem.TrainingRange());
    REQUIRE(scaling);

    // The public API does not expose BoundFor directly. For this affine tree/domain
    // we can reconstruct the raw identity enclosure exactly by hand: [2*(-4)+3, 2*4+3].
    auto const [expectedLo, expectedHi] = scaling->ApplyToValueInterval(Operon::Scalar{-5}, Operon::Scalar{11});

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-100}, Operon::Scalar{100}}});

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

TEST_CASE("ShapeConstrainedEvaluator - derivative through an unsupported op is not falsely feasible", "[shape-constraints]")
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
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt});

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
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt});

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
    cs.Domains.insert_or_assign("NotAColumn", std::pair{Operon::Scalar{0}, Operon::Scalar{1}});
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable, cs), std::invalid_argument);
}

TEST_CASE("ShapeConstrainedEvaluator - constraint variable missing from domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    // No domains entry for X1 at all.
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt});
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable, cs), std::invalid_argument);
}

TEST_CASE("ShapeConstrainedEvaluator - problem input variable missing from domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-4}, Operon::Scalar{4}}});

    CHECK_THROWS_WITH(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable, cs), Catch::Matchers::ContainsSubstring("X2"));
    CHECK_THROWS_WITH(Operon::ShapeViolationEvaluator(&fx.problem, &fx.dtable, cs), Catch::Matchers::ContainsSubstring("X2"));
}

TEST_CASE("ShapeConstrainedEvaluator - domain error (e.g. division by zero-containing interval) is treated as infeasible, not a crash", "[shape-constraints]")
{
    // f(x) = 1 / X1, with X1's domain spanning zero -- AffineEvaluator
    // throws std::invalid_argument for this (affine_form::inv is
    // undefined over an interval containing zero); GP generates trees
    // like this constantly, so Feasible() must swallow it as "can't be
    // certified feasible" rather than letting a run crash.
    Fixture fx;
    auto tree = InfixParser::Parse("1 / X1", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{-1}, Operon::Scalar{1}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-100}, Operon::Scalar{100}}});

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    auto const measurement = sce.Measure(tree);
    REQUIRE(measurement.Measurements.size() == 1);
    CHECK_FALSE(measurement.Measurements[0].Certified);
    CHECK(measurement.Measurements[0].Violation == Catch::Approx(1.0));
    CHECK_FALSE(sce.Feasible(tree));
}

TEST_CASE("ShapeConstrainedEvaluator - falls back to interval bounds for constant integer powers", "[shape-constraints]")
{
    // AffineEvaluator represents both operands as affine forms. Its general
    // affine-base/affine-exponent operation rejects a negative base before it
    // discovers that the exponent is the constant integer 2. IntervalEvaluator
    // handles the mathematically valid power directly.
    Fixture fx;
    auto tree = InfixParser::Parse("(-0.91) ^ 2", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt,
                              .Bound = std::pair{Operon::Scalar{-1}, Operon::Scalar{1}}});

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    auto const measurement = sce.Measure(tree);
    REQUIRE(measurement.Measurements.size() == 1);
    CHECK(measurement.Measurements[0].Certified);
    REQUIRE(measurement.Measurements[0].Bound);
    CHECK(measurement.Feasible);
    CHECK(sce.Feasible(tree));
}

TEST_CASE("ShapeConstrainedEvaluator - a NaN bound endpoint (Scale==0 times an unbounded raw interval) is not falsely feasible", "[shape-constraints]")
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
    for (auto i = 0; i < Nrow; ++i) { data(i, 0) = static_cast<Operon::Scalar>(i); } // 0..19 -- training points stay finite under exp(); the domain box below is what overflows
    data.col(1).setConstant(Operon::Scalar{5}); // constant target -> covariance(*, y) == 0
    Operon::Dataset ds(gsl::not_null{data.data()}, Nrow, 2);

    Operon::Problem problem(&ds);
    problem.SetTrainingRange({0, Nrow});
    problem.SetTestRange({0, Nrow});
    problem.SetTarget("X2");
    problem.SetDefaultInputs();
    problem.SetLinearScalingEnabled(true);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    Operon::Evaluator<DTable> nmse(&problem, &dtable, Operon::NMSE{});

    auto tree = InfixParser::Parse("exp(X1)", ds);
    auto const scaling = Operon::FitLinearScaling(tree, problem, dtable, problem.TrainingRange());
    REQUIRE(scaling.has_value());
    CHECK(scaling->Scale == Operon::Scalar{0}); // exact zero, not a fallback

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{0}, Operon::Scalar{1000}}); // exp(1000) overflows double to +inf
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{0}, Operon::Scalar{1}}});

    Operon::ShapeConstrainedEvaluator sce(&nmse, &dtable, cs);
    CHECK_FALSE(sce.Feasible(tree));
}

TEST_CASE("ShapeConstraintPolicy validation covers GP and NSGA2 mode rules", "[shape-constraints]")
{
    using E = Operon::ShapeConstraintEnforcement;

    auto valid = [](E enforcement, bool isNsga2) {
        CHECK_FALSE(Operon::ValidatePolicy({.Enforcement = enforcement, .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{1}}, isNsga2));
    };
    auto invalid = [](E enforcement, bool isNsga2) {
        CHECK(Operon::ValidatePolicy({.Enforcement = enforcement, .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{1}}, isNsga2));
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

    CHECK(Operon::ValidatePolicy({.Enforcement = E::None, .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{1}}, false));
    CHECK(Operon::ValidatePolicy({.Enforcement = static_cast<E>(1U << 9U), .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{1}}, false));
    CHECK(Operon::ValidatePolicy({.Enforcement = E::HardReject, .UnknownViolation = Operon::Scalar{-1}, .PenaltyWeight = Operon::Scalar{1}}, false));
    CHECK(Operon::ValidatePolicy({.Enforcement = E::HardReject, .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{-1}}, false));
    CHECK(Operon::ValidatePolicy({.Enforcement = E::HardReject, .UnknownViolation = std::numeric_limits<Operon::Scalar>::quiet_NaN(), .PenaltyWeight = Operon::Scalar{1}}, false));
    CHECK(Operon::ValidatePolicy({.Enforcement = E::HardReject, .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = std::numeric_limits<Operon::Scalar>::quiet_NaN()}, false));
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
    auto loaded = Operon::LoadShapeConstraints(path.string());
    REQUIRE(loaded);

    auto requireValid = [](Operon::ShapeConstraintPolicy const& policy, bool isNsga2) {
        if (auto error = Operon::ValidatePolicy(policy, isNsga2)) { throw std::invalid_argument(*error); }
    };

    SECTION("GP hard-reject constructs the rejecting evaluator")
    {
        Operon::ShapeConstraintPolicy policy{.Enforcement = Operon::ParseShapeEnforcement("hard-reject"), .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{1}};
        REQUIRE_NOTHROW(requireValid(policy, false));
        Operon::ShapeConstrainedEvaluator gated(&fx.nmse, &fx.dtable, *loaded);
        CHECK(gated.Feasible(fx.tree));
    }

    SECTION("GP penalty constructs the violation evaluator and summed aggregate")
    {
        Operon::ShapeConstraintPolicy policy{
            .Enforcement = Operon::ParseShapeEnforcement("penalty"),
            .UnknownViolation = Operon::Scalar{2},
            .PenaltyWeight = Operon::Scalar{3},
        };
        REQUIRE_NOTHROW(requireValid(policy, false));
        Operon::ShapeViolationEvaluator violation(&fx.problem, &fx.dtable, *loaded, policy.PenaltyWeight, policy.UnknownViolation);
        Operon::MultiEvaluator aggregate(&fx.problem);
        aggregate.Add(&fx.nmse);
        aggregate.Add(&violation);
        aggregate.SetAggregateType(Operon::MultiEvaluator::AggregateType::Sum);
        CHECK(violation.RawViolation(fx.tree) == Catch::Approx(0.0));
    }

    SECTION("GP feasibility-first constructs the comparator-side violation evaluator")
    {
        Operon::ShapeConstraintPolicy policy{.Enforcement = Operon::ParseShapeEnforcement("feasibility-first"), .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{1}};
        REQUIRE_NOTHROW(requireValid(policy, false));
        fx.problem.SetLinearScalingEnabled(false);
        Operon::ShapeViolationEvaluator violation(&fx.problem, &fx.dtable, *loaded, Operon::Scalar{1}, policy.UnknownViolation);
        Operon::FeasibilityFirstComparison comp([&violation](Operon::Tree const& t) { return violation.Measure(t).Feasible; });
        auto feasible = Fixture::MakeIndividual(fx.tree);
        feasible.Fitness = {10.0F};
        auto infeasible = Fixture::MakeIndividual(InfixParser::Parse("X2 - X1", fx.ds));
        infeasible.Fitness = {0.1F};
        CHECK(comp(feasible, infeasible));
    }

    SECTION("NSGA2 extra-objective constructs the added shape objective")
    {
        Operon::ShapeConstraintPolicy policy{.Enforcement = Operon::ParseShapeEnforcement("extra-objective"), .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{1}};
        REQUIRE_NOTHROW(requireValid(policy, true));
        Operon::ShapeViolationEvaluator extra(&fx.problem, &fx.dtable, *loaded, Operon::Scalar{1}, policy.UnknownViolation);
        Operon::MultiEvaluator objectives(&fx.problem);
        objectives.Add(&fx.nmse);
        objectives.Add(&extra);
        CHECK(extra.RawViolation(fx.tree) == Catch::Approx(0.0));
    }

    SECTION("invalid CLI mode combination is rejected by the same validation step")
    {
        Operon::ShapeConstraintPolicy policy{.Enforcement = Operon::ParseShapeEnforcement("penalty,extra-objective"), .UnknownViolation = Operon::Scalar{1}, .PenaltyWeight = Operon::Scalar{1}};
        CHECK_THROWS_AS(requireValid(policy, false), std::invalid_argument);
    }
}

TEST_CASE("ShapeViolationEvaluator - sign constraint violation magnitudes", "[shape-constraints]")
{
    Fixture fx;
    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt});

    auto bad = InfixParser::Parse("X2 - X1", fx.ds);
    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs, Operon::Scalar{3});
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
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-1}, Operon::Scalar{1}}});

    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs);
    auto m = sve.Measure(fx.tree);
    REQUIRE(m.Measurements.size() == 1);
    REQUIRE(m.Measurements[0].Bound);
    CHECK(m.Measurements[0].Bound->first == Catch::Approx(-4.0));
    CHECK(m.Measurements[0].Bound->second == Catch::Approx(4.0));
    CHECK(m.Violation == Catch::Approx(6.0));
    CHECK_FALSE(m.Feasible);
}

TEST_CASE("ShapeViolationEvaluator - identity, first-derivative, and second-derivative measurements", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-4}, Operon::Scalar{4}}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{1}, Operon::Scalar{1}}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::SecondDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt});

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
    secondDerivativeOnly.Constraints.push_back({.Op = ShapeConstraintOp::SecondDerivative, .Variable = "X1", .Sign = -1, .Bound = std::nullopt});
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
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{-1}, Operon::Scalar{1}}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt});

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

    std::vector<Operon::Individual> pop{Fixture::MakeIndividual(fx.tree)};
    sve.Prepare(pop);
    checkSame(first, sve.Measure(fx.tree));
}

TEST_CASE("ShapeViolationEvaluator - unknown violation and empty constraint set", "[shape-constraints]")
{
    Fixture fx;
    auto unknownTree = InfixParser::Parse("abs(X1)", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt});

    Operon::ShapeViolationEvaluator sve(&fx.problem, &fx.dtable, cs, Operon::Scalar{1}, Operon::Scalar{2.5});
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
        cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
        cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
        cs.Constraints.push_back(std::move(c));
        return cs;
    };

    // neither Sign nor Bound set
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable, withOneConstraint({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::nullopt})),
        std::invalid_argument);
    // both set
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable,
        withOneConstraint({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = 1, .Bound = std::pair{Operon::Scalar{0}, Operon::Scalar{1}}})),
        std::invalid_argument);
    // invalid sign
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable, withOneConstraint({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = 2, .Bound = std::nullopt})),
        std::invalid_argument);
    // lo > hi
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, &fx.dtable,
        withOneConstraint({.Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair{Operon::Scalar{5}, Operon::Scalar{1}}})),
        std::invalid_argument);
}

TEST_CASE("FeasibilityFirstComparison - feasible precedes infeasible regardless of fitness", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt}); // true: f is non-decreasing in X1

    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);
    Operon::FeasibilityFirstComparison comp([&sce](Operon::Tree const& t) { return sce.Feasible(t); });

    auto feasibleWorseFit = Fixture::MakeIndividual(fx.tree); // satisfies the constraint
    feasibleWorseFit.Fitness = {10.0F};

    auto infeasibleBetterFit = InfixParser::Parse("X2 - X1", fx.ds); // violates: non-increasing in X1
    auto infeasibleInd = Fixture::MakeIndividual(infeasibleBetterFit);
    infeasibleInd.Fitness = {0.01F};

    CHECK(comp(feasibleWorseFit, infeasibleInd));       // feasible wins despite worse fitness
    CHECK_FALSE(comp(infeasibleInd, feasibleWorseFit));

    // equal feasibility -> falls back to the wrapped comparator (fitness order)
    auto feasibleBetterFit = Fixture::MakeIndividual(fx.tree);
    feasibleBetterFit.Fitness = {1.0F};
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
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt}); // true for X1 - X2

    fx.problem.SetLinearScalingEnabled(false);
    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, &fx.dtable, cs);

    auto infeasibleTree = InfixParser::Parse("X2 - X1", fx.ds); // false: non-increasing in X1
    std::vector<Operon::Individual> pop{Fixture::MakeIndividual(fx.tree), Fixture::MakeIndividual(infeasibleTree)};
    sce.Prepare(pop);

    CHECK(sce.Feasible(fx.tree));
    CHECK_FALSE(sce.Feasible(infeasibleTree));

    // A second Prepare() with a different population clears and rebuilds
    // the cache; trees from the first population must still resolve
    // correctly afterward (Feasible() computes fresh on a miss, doesn't
    // require having been in the most recent Prepare() call).
    auto other = InfixParser::Parse("X1", fx.ds);
    std::vector<Operon::Individual> pop2{Fixture::MakeIndividual(other)};
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
        problem.SetTrainingRange({0, static_cast<std::size_t>(ds.Rows())});
        problem.SetTestRange({0, static_cast<std::size_t>(ds.Rows())});
        problem.SetTarget(target);
        problem.SetDefaultInputs();

        auto path = WriteShapeConfig(label, constraintsJson);
        auto loaded = Operon::LoadShapeConstraints(path.string());
        REQUIRE(loaded);

        using DTable = DispatchTable<Operon::Scalar>;
        DTable dtable;
        Operon::Evaluator<DTable> nmse(&problem, &dtable, Operon::NMSE{});

        auto tree = InfixParser::Parse(model, ds);
        Operon::ShapeConstrainedEvaluator sce(&nmse, &dtable, *loaded);
        auto feasible = sce.Feasible(tree);
        WARN(label << " Feasible()=" << feasible << " (expected false -- independent check found a real violation)");
        CHECK_FALSE(feasible);
    };

    auto const base = std::string{"/home/bogdb/src/operon-workspace/operon-publications/experiments/shape-constraints-reproduction/results/full_run/"};
    auto const flowStressConstraints = R"json({"domains": {"T": [350, 510], "phi": [0, 0.7], "phi_dot": [0.001, 10]}, "constraints": [{"op": "id", "bound": [0, 200]}, {"op": "derivative", "variable": "T", "order": 1, "sign": -1}, {"op": "derivative", "variable": "phi_dot", "order": 1, "sign": 1}, {"op": "derivative", "variable": "phi", "order": 2, "sign": -1}]})json";
    auto const carsConstraints = R"json({"domains": {"cylinders": [3, 8], "displacement": [68, 455], "horsepower": [46, 230], "weight": [1613, 5140], "acceleration": [8, 23.5]}, "constraints": [{"op": "id", "bound": [9.0, 46.6]}, {"op": "derivative", "variable": "displacement", "order": 1, "sign": -1}, {"op": "derivative", "variable": "horsepower", "order": 1, "sign": -1}, {"op": "derivative", "variable": "weight", "order": 1, "sign": -1}]})json";

    SECTION("Case A") {
        runCase("caseA", base + "Flow_stress_data.csv", "kf", flowStressConstraints,
            "((-1656471.375000) + (3.006961 * (((2.070481 * phi) + ((((((-0.107675) * T) + (2.079886 + (1.591433 * phi_dot))) + (((2.379283 * phi_dot) + (sin(((1.591433 * phi_dot) / 1.596721)) + (exp(exp(2.646259)) + (sin((1.591433 * phi_dot)) + (log((1.155026 + (1.591433 * phi_dot))) + (((-0.133379) * T) + log((1.155026 + (1.591433 * phi_dot))))))))) + ((sin((1.766540 * phi_dot)) + exp(2.313587)) / (exp(1.995921) ^ 2)))) * 0.065602) / 1.268149)) * exp(2.079886))))");
    }
    SECTION("Case B") {
        runCase("caseB", base + "Flow_stress_data.csv", "kf", flowStressConstraints,
            "(175.725388 + (0.000001 * (((((1.661356 * phi_dot) ^ 2) ^ 2) + ((((((cos(((3.070189 * phi_dot) / (-3.244338))) / 0.146962) ^ 2) ^ 2) / (-2.819170)) + (((((((2.715578 * phi_dot) ^ 2) + (((3.070189 * phi_dot) / (-3.244338)) / 0.035275)) + ((-0.302006) * T)) + (((1.920551 * phi_dot) ^ 2) + ((((((3.367852 * phi_dot) / (-3.460277)) / (-0.037834)) + ((-1.131171) * T)) / (-0.037834)) + ((1.920551 * phi_dot) ^ 2)))) + ((-1.244043) * T)) / (-0.037834))) / (-0.037834))) / (-0.037834))))");
    }
    SECTION("Case C") {
        runCase("caseC", base + "Flow_stress_data.csv", "kf", flowStressConstraints,
            "((-0.541280) + (0.997766 * (((((((((((((((((-0.000020) * phi) + ((-0.000002) * phi_dot)) + (-1.965610)) + 1.881476) * 2041.673462) + ((-1.431116) * (-119.989891))) + (0.000224 * T)) / 168.553055) * ((-145.032639) + ((1.275833 * phi_dot) + (0.093371 * T)))) + (-1.971097)) + 1.879035) * 2041.673462) + ((-1.337756) * (-112.162704))) + ((-0.049931) * T)) + 159.525269) + (-6.337012))))");
    }
    SECTION("Case D") {
        runCase("caseD", base + "Cars_data.csv", "mpg", carsConstraints,
            "(13634192.000000 + ((-0.001787) * ((((2.305640 * displacement) + ((1.891310 * weight) + ((exp(exp(1.582666)) * ((1.250606 * cylinders) + exp(exp(2.884111)))) + (0.360743 * displacement)))) + ((0.781982 * cylinders) + (exp(0.058231) + ((((-0.539599) * cylinders) + ((0.506669 * displacement) + ((0.506669 * displacement) + 0.303519))) + (exp(exp(1.582666)) + (exp(0.303519) + (exp(1.582666) * 0.058231))))))) + (((1.525301 * weight) * exp((0.058231 + exp(0.303519)))) * 0.058231))))");
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
// bound is unsound here, not just conservative. Run with
// OPERON_SHAPE_DEBUG=1 and this test's tag alone
// (`-t "[shape-constraints-scratch]"`) to see which of TryAffineBound's
// three paths (affine-direct / ill-conditioned-fallback / exception-
// fallback) produced the wrong bound.
TEST_CASE("SCRATCH ind333 sign-wrong derivative bound repro", "[.][shape-constraints-scratch]")
{
    // Must be the real training data (not a degenerate stand-in): the
    // linear-scaling fit (Scale/Offset) that TransformBound applies to the
    // raw derivative bound depends on it, and a degenerate/constant target
    // fits Scale=0, which zeroes out any bound (including a correct one)
    // and produces a misleadingly "feasible" result unrelated to the real bug.
    Operon::Dataset ds("/home/bogdb/src/operon-workspace/operon-publications/experiments/shape-constraints-reproduction/results/full_sweep/II_11_27_without_noise_rep00_data.csv", /*hasHeader=*/true);
    Operon::Problem problem(&ds);
    problem.SetTrainingRange({0, 100});
    problem.SetTestRange({100, 200});
    problem.SetTarget("y");
    problem.SetDefaultInputs();

    auto const constraintsJson = R"json({"domains": {"n": [0, 1], "alpha": [0, 1], "epsilon": [1, 2], "Ef": [1, 2]}, "constraints": [{"op": "derivative", "variable": "n", "order": 1, "sign": 1}, {"op": "derivative", "variable": "alpha", "order": 1, "sign": 1}, {"op": "derivative", "variable": "epsilon", "order": 1, "sign": 1}, {"op": "derivative", "variable": "Ef", "order": 1, "sign": 1}]})json";
    auto const path = WriteShapeConfig("ind333", constraintsJson);
    auto loaded = Operon::LoadShapeConstraints(path.string());
    REQUIRE(loaded);

    auto const model = "(exp((((0.903161883 * n) * (4.700151443 * alpha)) * ((((-4.018970013) * Ef) * (4.741394520 * epsilon)) + (((2.177207947 * n) * (2.639619350 * alpha)) * (((-1.325337768) * Ef) * (1.827161431 * epsilon)))))) + (((-4.018970013) * Ef) * ((0.903161883 * n) * (4.700151443 * alpha))))";
    auto tree = InfixParser::Parse(model, ds);

    using DTable = DispatchTable<Operon::Scalar>;
    DTable dtable;
    Operon::Evaluator<DTable> nmse(&problem, &dtable, Operon::NMSE{});
    Operon::ShapeConstrainedEvaluator sce(&nmse, &dtable, *loaded);
    auto const summary = sce.Measure(tree);
    REQUIRE(summary.Measurements.size() == 4);
    auto const& dn = summary.Measurements[0]; // d/dn, sign >= 0 required
    INFO("d/dn certified=" << dn.Certified << " bound=[" << (dn.Bound ? dn.Bound->first : 0) << ", " << (dn.Bound ? dn.Bound->second : 0) << "] violation=" << dn.Violation);
    WARN("Operon certifies feasible=" << summary.Feasible << " for a tree whose true d/dn is always negative over the domain box (verified by finite-difference sampling) -- expected infeasible.");
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
    Operon::Dataset ds("/home/bogdb/src/operon-workspace/operon-publications/experiments/shape-constraints-reproduction/results/full_sweep/II_11_27_without_noise_rep00_data.csv", /*hasHeader=*/true);

    Operon::Map<Operon::Hash, std::pair<Operon::Scalar, Operon::Scalar>> domains;
    domains.insert_or_assign(ds.GetVariable("n")->Hash, std::pair{Operon::Scalar{0}, Operon::Scalar{1}});
    domains.insert_or_assign(ds.GetVariable("alpha")->Hash, std::pair{Operon::Scalar{0}, Operon::Scalar{1}});
    domains.insert_or_assign(ds.GetVariable("epsilon")->Hash, std::pair{Operon::Scalar{1}, Operon::Scalar{2}});
    domains.insert_or_assign(ds.GetVariable("Ef")->Hash, std::pair{Operon::Scalar{1}, Operon::Scalar{2}});

    struct Case { std::string label; std::string model; Operon::Scalar scale; };
    std::vector<Case> const cases{
        {"ind431", "(exp(exp((((-1.325337768) * Ef) * (1.827161431 * epsilon)))) + (((0.903161883 * n) * (4.700151443 * alpha)) * ((((-4.018970013) * Ef) * (4.741394520 * epsilon)) + (((0.903161883 * n) * (4.700151443 * alpha)) * (((-1.325337768) * Ef) * (1.827161431 * epsilon))))))", Operon::Scalar{-0.0116670932}},
        {"ind450", "(((4.700151443 * alpha) * (((-1.325337768) * Ef) * (1.827161431 * epsilon))) + (((0.903161883 * n) * (4.700151443 * alpha)) * ((((-4.018970013) * Ef) * (4.741394520 * epsilon)) + (((0.903161883 * n) * (4.700151443 * alpha)) * (((-1.325337768) * Ef) * (1.827161431 * epsilon))))))", Operon::Scalar{-0.0104688368}},
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

        Operon::Vector<Operon::Node> sliced(dag.Nodes.begin(), dag.Nodes.begin() + static_cast<std::ptrdiff_t>(root) + 1);
        Operon::Tree dtree(std::move(sliced));
        dtree.UpdateNodes();

        Operon::AffineEvaluator ae(&dtree, domains);
        auto const affineRaw = ae.Evaluate(dtree.GetCoefficients()).to_interval();

        Operon::IntervalEvaluator ie(&dtree, Operon::IntervalEvaluator::DomainMap{domains});
        auto const intervalRaw = ie.Evaluate(dtree.GetCoefficients());

        // Apply the scale factor manually (matches TransformBound's
        // ApplyToDerivativeInterval for a pure scale multiply -- offset drops
        // out of a derivative).
        auto const scaleInterval = [&](Operon::Scalar lo, Operon::Scalar hi) -> std::pair<Operon::Scalar, Operon::Scalar> {
            auto a = c.scale * lo;
            auto b = c.scale * hi;
            return a <= b ? std::pair{a, b} : std::pair{b, a};
        };
        auto const [affLo, affHi] = scaleInterval(affineRaw.inf(), affineRaw.sup());
        auto const [intLo, intHi] = scaleInterval(intervalRaw.inf(), intervalRaw.sup());

        WARN(c.label << " d/dn (scaled): affine=[" << affLo << ", " << affHi << "] "
            << "operon-interval=[" << intLo << ", " << intHi << "]");
    }
}

} // namespace Operon::Test
