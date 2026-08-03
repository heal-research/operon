// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <stdexcept>

#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/constraint.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
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
        , nmse(&problem, &dtable, Operon::NMSE{}, /*linearScaling=*/true)
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

} // namespace

TEST_CASE("ShapeConstrainedEvaluator - correctly-signed constraints are feasible", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1});  // non-decreasing: true
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X2", .Sign = -1}); // non-increasing: true

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, cs);
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
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = -1});

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, cs);
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
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Bound = std::pair{Operon::Scalar{-4}, Operon::Scalar{4}}});
    Operon::ShapeConstrainedEvaluator wide(&fx.nmse, cs);
    CHECK(wide.Feasible(fx.tree));

    cs.Constraints[0].Bound = std::pair{Operon::Scalar{-1}, Operon::Scalar{1}};
    Operon::ShapeConstrainedEvaluator narrow(&fx.nmse, cs);
    CHECK_FALSE(narrow.Feasible(fx.tree));
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
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1});

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, cs);
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
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1});

    Operon::ShapeConstrainedEvaluator nonDecreasing(&fx.nmse, cs);
    CHECK(nonDecreasing.Feasible(tree));

    cs.Constraints[0].Sign = -1;
    Operon::ShapeConstrainedEvaluator nonIncreasing(&fx.nmse, cs);
    CHECK(nonIncreasing.Feasible(tree));
}

TEST_CASE("ShapeConstrainedEvaluator - unknown variable in domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("NotAColumn", std::pair{Operon::Scalar{0}, Operon::Scalar{1}});
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, cs), std::invalid_argument);
}

TEST_CASE("ShapeConstrainedEvaluator - constraint variable missing from domains throws", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    // No domains entry for X1 at all.
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1});
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, cs), std::invalid_argument);
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
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Bound = std::pair{Operon::Scalar{-100}, Operon::Scalar{100}}});

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, cs);
    CHECK_FALSE(sce.Feasible(tree));
}

TEST_CASE("ShapeConstraintPolicy validation covers GP and NSGA2 mode rules", "[shape-constraints]")
{
    using E = Operon::ShapeConstraintEnforcement;

    auto valid = [](E enforcement, bool isNsga2) {
        CHECK_FALSE(Operon::ValidatePolicy({.Enforcement = enforcement}, isNsga2));
    };
    auto invalid = [](E enforcement, bool isNsga2) {
        CHECK(Operon::ValidatePolicy({.Enforcement = enforcement}, isNsga2));
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

    CHECK(Operon::ValidatePolicy({.Enforcement = E::None}, false));
    CHECK(Operon::ValidatePolicy({.Enforcement = static_cast<E>(1U << 9U)}, false));
    CHECK(Operon::ValidatePolicy({.Enforcement = E::HardReject, .UnknownViolation = Operon::Scalar{-1}}, false));
    CHECK(Operon::ValidatePolicy({.Enforcement = E::HardReject, .PenaltyWeight = Operon::Scalar{-1}}, false));
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

TEST_CASE("ShapeViolationEvaluator - sign constraint violation magnitudes", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1});

    auto bad = InfixParser::Parse("X2 - X1", fx.ds);
    Operon::ShapeViolationEvaluator sve(&fx.nmse, cs, Operon::Scalar{3});
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
    Operon::ShapeViolationEvaluator mirror(&fx.nmse, cs);
    CHECK(mirror.RawViolation(fx.tree) == Catch::Approx(1.0));
}

TEST_CASE("ShapeViolationEvaluator - bound constraint violation magnitude", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Bound = std::pair{Operon::Scalar{-1}, Operon::Scalar{1}}});

    Operon::ShapeViolationEvaluator sve(&fx.nmse, cs);
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
    cs.Constraints.push_back({.Op = ShapeConstraintOp::Identity, .Bound = std::pair{Operon::Scalar{-4}, Operon::Scalar{4}}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Bound = std::pair{Operon::Scalar{1}, Operon::Scalar{1}}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::SecondDerivative, .Variable = "X1", .Sign = 1});

    Operon::ShapeViolationEvaluator sve(&fx.nmse, cs);
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
    secondDerivativeOnly.Constraints.push_back({.Op = ShapeConstraintOp::SecondDerivative, .Variable = "X1", .Sign = -1});
    Operon::ShapeViolationEvaluator violated(&fx.nmse, secondDerivativeOnly);
    auto v = violated.Measure(square);
    CHECK_FALSE(v.Feasible);
    CHECK(v.Violation == Catch::Approx(2.0));
}

TEST_CASE("ShapeViolationEvaluator - unknown violation and empty constraint set", "[shape-constraints]")
{
    Fixture fx;
    auto unknownTree = InfixParser::Parse("abs(X1)", fx.ds);

    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1});

    Operon::ShapeViolationEvaluator sve(&fx.nmse, cs, Operon::Scalar{1}, Operon::Scalar{2.5});
    auto m = sve.Measure(unknownTree);
    REQUIRE(m.Measurements.size() == 1);
    CHECK_FALSE(m.Measurements[0].Certified);
    CHECK_FALSE(m.Feasible);
    CHECK(m.Violation == Catch::Approx(2.5));

    cs.Constraints.clear();
    Operon::ShapeViolationEvaluator empty(&fx.nmse, cs);
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
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, withOneConstraint({.Op = ShapeConstraintOp::Identity})),
        std::invalid_argument);
    // both set
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse,
        withOneConstraint({.Op = ShapeConstraintOp::Identity, .Sign = 1, .Bound = std::pair{Operon::Scalar{0}, Operon::Scalar{1}}})),
        std::invalid_argument);
    // invalid sign
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse, withOneConstraint({.Op = ShapeConstraintOp::Identity, .Sign = 2})),
        std::invalid_argument);
    // lo > hi
    CHECK_THROWS_AS(Operon::ShapeConstrainedEvaluator(&fx.nmse,
        withOneConstraint({.Op = ShapeConstraintOp::Identity, .Bound = std::pair{Operon::Scalar{5}, Operon::Scalar{1}}})),
        std::invalid_argument);
}

TEST_CASE("FeasibilityFirstComparison - feasible precedes infeasible regardless of fitness", "[shape-constraints]")
{
    Fixture fx;
    Operon::ShapeConstraintSet cs;
    cs.Domains.insert_or_assign("X1", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Domains.insert_or_assign("X2", std::pair{Operon::Scalar{1}, Operon::Scalar{5}});
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1}); // true: f is non-decreasing in X1

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, cs);
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
    cs.Constraints.push_back({.Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1}); // true for X1 - X2

    Operon::ShapeConstrainedEvaluator sce(&fx.nmse, cs);

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

} // namespace Operon::Test
