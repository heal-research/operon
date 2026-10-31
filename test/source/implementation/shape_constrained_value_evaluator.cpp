// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_approx.hpp>
#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <limits>
#include <memory>
#include <optional>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

#include "operon/core/concepts.hpp"
#include "operon/core/constraint.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/linear_scaling.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/parser/infix.hpp"
#include "operon/random/random.hpp"

#include "fixed_scaling_evaluator.hpp"

namespace Operon::Test {

namespace {

    using CountingGate = ShapeConstrainedValueEvaluator<PredictionCountingEvaluator>;
    using FixedGate = ShapeConstrainedValueEvaluator<FixedScalingEvaluator>;
    using StructuralList = std::vector<gsl::not_null<Operon::TreePropertyEvaluator const*>>;

    // f(X1, X2) = X1 - X2 on [1,5]x[1,5], 20 rows, linear scaling disabled so the raw derivative sign is
    // what the constraints see: non-decreasing in X1, non-increasing in X2.
    struct GateFixture {
        static constexpr auto Nrow { 20 };
        static constexpr auto Ncol { 3 }; // X1, X2, y

        Operon::RandomGenerator rng { 0 };
        Eigen::Array<Operon::Scalar, -1, -1> data { Nrow, Ncol };
        Operon::Dataset ds;
        Operon::Tree tree; // X1 - X2
        ScalarDispatch dtable;
        Operon::Problem problem;

        GateFixture()
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
        {
            problem.SetTrainingRange({ 0, Nrow });
            problem.SetTestRange({ 0, Nrow });
            problem.SetTarget("X3");
            problem.SetLinearScalingEnabled(false);
        }

        [[nodiscard]] auto Buffer() const -> std::vector<Operon::Scalar>
        {
            return std::vector<Operon::Scalar>(problem.TrainingRange().Size());
        }

        [[nodiscard]] auto Parse(std::string const& expr) const -> Operon::Tree
        {
            return InfixParser::ParseOrThrow(expr, ds);
        }

        static auto MakeIndividual(Operon::Tree const& t) -> Operon::Individual
        {
            Operon::Individual ind;
            ind.Genotype = t;
            return ind;
        }

        // The NMSE interpreter evaluator, counting its prediction phase, constrained by the typed adapter.
        [[nodiscard]] auto MakeGate(Operon::ShapeConstraintSet constraints, ShapeConstraintPolicy policy,
            StructuralList structural = {}) -> std::unique_ptr<CountingGate>
        {
            return std::make_unique<CountingGate>(
                ShapeConstraintSpec {
                    .Constraints = std::move(constraints), .Policy = policy, .Structural = std::move(structural) },
                &dtable, &problem, &dtable, Operon::NMSE {});
        }

        // The fixed-transform evaluator constrained by the typed adapter.
        [[nodiscard]] auto MakeFixedGate(Operon::ShapeConstraintSet constraints, ShapeConstraintPolicy policy,
            std::optional<LinearScaling> scaling) -> std::unique_ptr<FixedGate>
        {
            return std::make_unique<FixedGate>(
                ShapeConstraintSpec { .Constraints = std::move(constraints), .Policy = policy }, &dtable, &problem,
                &dtable, scaling);
        }
    };

    auto Domains() -> Operon::ShapeConstraintSet
    {
        Operon::ShapeConstraintSet cs;
        cs.Domains.insert_or_assign("X1", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
        cs.Domains.insert_or_assign("X2", std::pair { Operon::Scalar { 1 }, Operon::Scalar { 5 } });
        return cs;
    }

    // d/dX1 >= 0: true for X1 - X2 (violation 0), false for X2 - X1 (violation exactly 1).
    auto IncreasingInX1() -> Operon::ShapeConstraintSet
    {
        auto cs = Domains();
        cs.Constraints.push_back(
            { .Op = ShapeConstraintOp::FirstDerivative, .Variable = "X1", .Sign = 1, .Bound = std::nullopt });
        return cs;
    }

    auto IdentityBound(Operon::Scalar lo, Operon::Scalar hi) -> Operon::ShapeConstraintSet
    {
        auto cs = Domains();
        cs.Constraints.push_back(
            { .Op = ShapeConstraintOp::Identity, .Variable = "", .Sign = std::nullopt, .Bound = std::pair { lo, hi } });
        return cs;
    }

    auto Policy(ShapeConstraintEnforcement enforcement, Operon::Scalar unknown = 1, Operon::Scalar weight = 3)
        -> ShapeConstraintPolicy
    {
        return ShapeConstraintPolicy {
            .Enforcement = enforcement, .UnknownViolation = unknown, .PenaltyWeight = weight
        };
    }

    // Replaces every coefficient with `value`, as a local search that optimizes all of them to one point.
    class ConstantCoefficientOptimizer final : public OptimizerBase {
    public:
        ConstantCoefficientOptimizer(gsl::not_null<Problem const*> problem, Operon::Scalar value)
            : OptimizerBase(problem)
            , value_(value)
        {
        }

        [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*rng*/, Tree const& tree) const -> FitOutcome override
        {
            FitResult result;
            result.InitialParameters = tree.GetCoefficients();
            result.FinalParameters.assign(result.InitialParameters.size(), value_);
            result.InitialCost = static_cast<Operon::Scalar>(1);
            result.FinalCost = static_cast<Operon::Scalar>(0);
            result.FunctionEvaluations = 1;
            return result;
        }

    private:
        Operon::Scalar value_;
    };

    // A structural objective that wrongly declares two objectives.
    class TwoObjectiveProperty final : public Operon::TreePropertyEvaluator {
    public:
        using Operon::TreePropertyEvaluator::TreePropertyEvaluator;

        [[nodiscard]] auto ObjectiveCount() const -> std::size_t override { return 2; }
    };

    auto Length(Operon::Problem const* problem) -> std::unique_ptr<Operon::TreePropertyEvaluator>
    {
        return std::make_unique<Operon::TreePropertyEvaluator>(
            problem, [](Operon::Tree const& t) { return static_cast<Operon::Scalar>(t.Length()); });
    }

    // Whether the adapter template accepts `T` as its base evaluator.
    template <typename T>
    concept CanBeAdapterBase = requires { typename ShapeConstrainedValueEvaluator<T>; };

} // namespace

// ── type structure and access boundary ──────────────────────────────────────────

// The adapter is a derived typed evaluator, not a wrapper: it *is* its base evaluator and an EvaluatorBase.
static_assert(std::is_base_of_v<PredictionCountingEvaluator, CountingGate>);
static_assert(std::is_base_of_v<Evaluator<ScalarDispatch>, CountingGate>);
static_assert(std::is_base_of_v<EvaluatorBase, CountingGate>);
static_assert(std::is_base_of_v<detail::ShapeConstraintEnforcer, CountingGate>);
static_assert(std::is_final_v<CountingGate>);
static_assert(!std::is_final_v<PredictionCountingEvaluator>);
static_assert(Concepts::EvaluatorCallable<CountingGate>);

// The phase hooks and the restoration hook stay unreachable from outside the hierarchy.
static_assert(!CanCallEvaluatePhase<CountingGate>);
static_assert(!CanCallScorePhase<CountingGate>);
static_assert(!CanCallRestorationHook<CountingGate>);

// Nested constrained adapters are rejected at compile time: an adapter is not a value evaluator, and neither is a
// composition, a user-defined objective or a structural objective.
static_assert(Concepts::ValueEvaluator<PredictionCountingEvaluator>);
static_assert(Concepts::ValueEvaluator<FixedScalingEvaluator>);
static_assert(Concepts::ValueEvaluator<Evaluator<ScalarDispatch>>);
static_assert(Concepts::ShapeConstrainedEvaluator<CountingGate>);
static_assert(!Concepts::ShapeConstrainedEvaluator<PredictionCountingEvaluator>);
static_assert(!Concepts::ValueEvaluator<CountingGate>);
static_assert(!Concepts::ValueEvaluator<FixedGate>);
static_assert(!Concepts::ValueEvaluator<MultiEvaluator>);
static_assert(!Concepts::ValueEvaluator<UserDefinedEvaluator>);
static_assert(!Concepts::ValueEvaluator<TreePropertyEvaluator>);
static_assert(CanBeAdapterBase<PredictionCountingEvaluator>);
static_assert(!CanBeAdapterBase<CountingGate>);
static_assert(!CanBeAdapterBase<ShapeConstrainedValueEvaluator<Evaluator<ScalarDispatch>>>);
static_assert(!CanBeAdapterBase<MultiEvaluator>);
static_assert(!CanBeAdapterBase<TreePropertyEvaluator>);

// A post-objective is a TreePropertyEvaluator: neither a value evaluator nor an adapter can be passed as one.
static_assert(!std::is_convertible_v<Evaluator<ScalarDispatch>*, gsl::not_null<TreePropertyEvaluator const*>>);
static_assert(!std::is_convertible_v<MultiEvaluator*, gsl::not_null<TreePropertyEvaluator const*>>);
static_assert(!std::is_convertible_v<CountingGate*, gsl::not_null<TreePropertyEvaluator const*>>);

TEST_CASE("ShapeConstrainedValueEvaluator construction rejects policies and structural objectives that conflict",
    "[shape-adapter]")
{
    GateFixture fx;
    auto const make = [&](ShapeConstraintPolicy policy, StructuralList structural = {}) {
        return fx.MakeGate(IncreasingInX1(), policy, std::move(structural));
    };
    auto const hard = Policy(ShapeConstraintEnforcement::HardReject);

    SECTION("structural post-objectives must be single-objective")
    {
        auto const length = Length(&fx.problem);
        CHECK_NOTHROW(make(hard, { length.get() }));

        TwoObjectiveProperty const wide { &fx.problem,
            [](Operon::Tree const& t) { return static_cast<Operon::Scalar>(t.Length()); } };
        CHECK_THROWS_AS(make(hard, { &wide }), std::invalid_argument);
    }

    SECTION("every representable enforcement mode is accepted by the adapter")
    {
        CHECK_NOTHROW(make(Policy(ShapeConstraintEnforcement::HardReject)));
        CHECK_NOTHROW(make(Policy(ShapeConstraintEnforcement::Penalty)));
        CHECK_NOTHROW(make(Policy(ShapeConstraintEnforcement::ExtraObjective)));
        CHECK_NOTHROW(make(Policy(ShapeConstraintEnforcement::PenaltyAndExtraObjective)));
    }

    SECTION("the spec's worst value is applied and an invalid bound depth is rejected at construction")
    {
        ShapeConstraintSpec spec { .Constraints = IncreasingInX1(),
            .Policy = hard,
            .WorstValue = 9.0,
            .Bound = { .Mode = ShapeBoundMode::Interval, .BisectionDepth = 5 } };
        CountingGate const gate { std::move(spec), &fx.dtable, &fx.problem, &fx.dtable, Operon::NMSE {} };
        CHECK(gate.WorstValue() == 9.0);

        ShapeConstraintSpec bad { .Constraints = IncreasingInX1(), .Policy = hard, .Bound = { .BisectionDepth = 21 } };
        CHECK_THROWS_AS((CountingGate { std::move(bad), &fx.dtable, &fx.problem, &fx.dtable, Operon::NMSE {} }),
            std::invalid_argument);
    }

    SECTION("objective widths follow the declared structure")
    {
        auto const length = Length(&fx.problem);
        auto const gp = make(hard);
        CHECK(gp->ObjectiveCount() == 1);
        auto const nsgp = make(hard, { length.get() });
        CHECK(nsgp->ObjectiveCount() == 2);
        auto const extra = make(Policy(ShapeConstraintEnforcement::ExtraObjective), { length.get() });
        CHECK(extra->ObjectiveCount() == 3);
        CHECK_FALSE(gp->StructuralFitnessReusable());
    }
}

TEST_CASE("MakeValueEvaluator constructs the typed adapter before type erasure", "[shape-adapter]")
{
    GateFixture fx;
    auto const spec = std::optional { ShapeConstraintSpec {
        .Constraints = IncreasingInX1(), .Policy = Policy(ShapeConstraintEnforcement::HardReject) } };

    SECTION("with a spec the erased evaluator is the constrained adapter over the requested leaf")
    {
        auto const erased
            = MakeValueEvaluator<Evaluator<ScalarDispatch>>(spec, &fx.dtable, &fx.problem, &fx.dtable, Operon::NMSE {});
        REQUIRE(erased != nullptr);
        CHECK(dynamic_cast<detail::ShapeConstraintEnforcer const*>(erased.get()) != nullptr);
        CHECK(dynamic_cast<Evaluator<ScalarDispatch> const*>(erased.get()) != nullptr);
        CHECK(dynamic_cast<ShapeConstrainedValueEvaluator<Evaluator<ScalarDispatch>> const*>(erased.get()) != nullptr);
        CHECK_FALSE(erased->StructuralFitnessReusable());
    }

    SECTION("without a spec the leaf is returned unconstrained")
    {
        auto const erased = MakeValueEvaluator<Evaluator<ScalarDispatch>>(
            std::nullopt, &fx.dtable, &fx.problem, &fx.dtable, Operon::NMSE {});
        REQUIRE(erased != nullptr);
        CHECK(dynamic_cast<detail::ShapeConstraintEnforcer const*>(erased.get()) == nullptr);
        CHECK(erased->StructuralFitnessReusable());
    }

    SECTION("the closed set of likelihood-based leaves can be constrained")
    {
        auto const mdl
            = MakeValueEvaluator<MinimumDescriptionLengthEvaluator<ScalarDispatch, GaussianLikelihood<Operon::Scalar>>>(
                spec, &fx.dtable, &fx.problem, &fx.dtable);
        CHECK(dynamic_cast<detail::ShapeConstraintEnforcer const*>(mdl.get()) != nullptr);
        auto const fbf
            = MakeValueEvaluator<FractionalBayesFactorEvaluator<ScalarDispatch, GaussianLikelihood<Operon::Scalar>>>(
                spec, &fx.dtable, &fx.problem, &fx.dtable);
        CHECK(dynamic_cast<detail::ShapeConstraintEnforcer const*>(fbf.get()) != nullptr);
        auto const gauss = MakeValueEvaluator<GaussianLikelihoodEvaluator<ScalarDispatch>>(
            spec, &fx.dtable, &fx.problem, &fx.dtable);
        CHECK(dynamic_cast<detail::ShapeConstraintEnforcer const*>(gauss.get()) != nullptr);

        auto buf = fx.Buffer();
        auto const ind = GateFixture::MakeIndividual(fx.tree);
        CHECK((*mdl)(fx.rng, ind, buf).size() == 1);
    }
}

// ── one prediction pass in every mode, objective order and width ────────────────

TEST_CASE("The adapter performs exactly one prediction pass per call in every enforcement mode", "[shape-adapter]")
{
    GateFixture fx;
    auto const length = Length(&fx.problem);
    auto const feasible = GateFixture::MakeIndividual(fx.tree);
    auto const infeasible = GateFixture::MakeIndividual(fx.Parse("X2 - X1"));
    auto buf = fx.Buffer();

    using E = ShapeConstraintEnforcement;
    std::vector<std::pair<std::string, E>> const modes {
        { "hard-reject", E::HardReject },
        { "penalty", E::Penalty },
        { "extra-objective", E::ExtraObjective },
        { "penalty,extra-objective", E::PenaltyAndExtraObjective },
    };
    for (auto const& [name, enforcement] : modes) {
        for (bool const withStructural : { false, true }) {
            INFO(name << " structural=" << withStructural);
            StructuralList structural;
            if (withStructural) {
                structural.emplace_back(length.get());
            }
            auto const holder = fx.MakeGate(IncreasingInX1(), Policy(enforcement), structural);
            auto const& gate = *holder;

            for (auto const* ind : { &feasible, &infeasible }) {
                auto const before = gate.Predictions.load();
                auto const calls = gate.CallCount.load();
                auto const fit = gate(fx.rng, *ind, buf);
                CHECK(gate.Predictions.load() == before + 1);
                CHECK(gate.CallCount.load() == calls + 1);
                CHECK(fit.size() == gate.ObjectiveCount());
            }
        }
    }
}

TEST_CASE("The adapter's objective vector order and values", "[shape-adapter]")
{
    GateFixture fx;
    std::atomic_int lengthCalls { 0 };
    Operon::TreePropertyEvaluator length { &fx.problem, [&](Operon::Tree const& t) {
                                              ++lengthCalls;
                                              return static_cast<Operon::Scalar>(t.Length());
                                          } };
    Operon::Evaluator<ScalarDispatch> plain { &fx.problem, &fx.dtable, Operon::NMSE {} };
    auto buf = fx.Buffer();

    auto const feasible = GateFixture::MakeIndividual(fx.tree);
    auto const infeasible = GateFixture::MakeIndividual(fx.Parse("X2 - X1"));
    auto const errFeasible = plain(fx.rng, feasible, buf)[0];
    auto const errInfeasible = plain(fx.rng, infeasible, buf)[0];
    auto const lenFeasible = static_cast<Operon::Scalar>(feasible.Genotype.Length());
    auto const lenInfeasible = static_cast<Operon::Scalar>(infeasible.Genotype.Length());
    REQUIRE(errInfeasible > Operon::Scalar { 0 });
    lengthCalls = 0;

    using E = ShapeConstraintEnforcement;
    auto const run = [&](E enforcement, Operon::Individual const& ind) {
        auto const gate = fx.MakeGate(IncreasingInX1(), Policy(enforcement), { &length });
        gate->SetWorstValue(7);
        return (*gate)(fx.rng, ind, buf);
    };

    SECTION("hard-reject: error and length when feasible, WorstValue on every objective when not")
    {
        auto const ok = run(E::HardReject, feasible);
        REQUIRE(ok.size() == 2);
        CHECK(ok[0] == Catch::Approx(errFeasible).margin(1e-6));
        CHECK(ok[1] == lenFeasible);
        CHECK(lengthCalls == 1);

        lengthCalls = 0;
        auto const rejected = run(E::HardReject, infeasible);
        REQUIRE(rejected.size() == 2);
        CHECK(rejected[0] == Operon::Scalar { 7 });
        CHECK(rejected[1] == Operon::Scalar { 7 });
        CHECK(lengthCalls == 0); // structural post-objectives are not evaluated on a hard reject
    }

    SECTION("penalty folds the weighted violation into the error")
    {
        auto const ok = run(E::Penalty, feasible);
        REQUIRE(ok.size() == 2);
        CHECK(ok[0] == Catch::Approx(errFeasible).margin(1e-6));
        auto const penalized = run(E::Penalty, infeasible);
        REQUIRE(penalized.size() == 2);
        CHECK(penalized[0] == Catch::Approx(errInfeasible + Operon::Scalar { 3 }).margin(1e-5));
        CHECK(penalized[1] == lenInfeasible);
    }

    SECTION("extra-objective appends the violation after the structural objectives")
    {
        auto const ok = run(E::ExtraObjective, feasible);
        REQUIRE(ok.size() == 3);
        CHECK(ok[0] == Catch::Approx(errFeasible).margin(1e-6));
        CHECK(ok[1] == lenFeasible);
        CHECK(ok[2] == Operon::Scalar { 0 });

        auto const bad = run(E::ExtraObjective, infeasible);
        REQUIRE(bad.size() == 3);
        CHECK(bad[0] == Catch::Approx(errInfeasible).margin(1e-6));
        CHECK(bad[1] == lenInfeasible);
        CHECK(bad[2] == Catch::Approx(1.0).margin(1e-6));
    }

    SECTION("penalty with extra-objective preserves [error + penalty, length, violation]")
    {
        auto const bad = run(E::PenaltyAndExtraObjective, infeasible);
        REQUIRE(bad.size() == 3);
        CHECK(bad[0] == Catch::Approx(errInfeasible + Operon::Scalar { 3 }).margin(1e-5));
        CHECK(bad[1] == lenInfeasible);
        CHECK(bad[2] == Catch::Approx(1.0).margin(1e-6));
    }

    SECTION("GP shapes without structural objectives")
    {
        auto const hard = fx.MakeGate(IncreasingInX1(), Policy(E::HardReject));
        hard->SetWorstValue(7);
        auto const rejected = (*hard)(fx.rng, infeasible, buf);
        REQUIRE(rejected.size() == 1);
        CHECK(rejected[0] == Operon::Scalar { 7 });
        CHECK(hard->Violations() == 1);

        auto const penalty = fx.MakeGate(IncreasingInX1(), Policy(E::Penalty));
        auto const penalized = (*penalty)(fx.rng, infeasible, buf);
        REQUIRE(penalized.size() == 1);
        CHECK(penalized[0] == Catch::Approx(errInfeasible + Operon::Scalar { 3 }).margin(1e-5));
        CHECK(penalty->Violations() == 0);
    }
}

// ── failures ────────────────────────────────────────────────────────────────────

TEST_CASE("The adapter's failure outputs span the whole objective vector", "[shape-adapter]")
{
    GateFixture fx;
    auto const length = Length(&fx.problem);
    auto buf = fx.Buffer();
    using E = ShapeConstraintEnforcement;
    auto const holder = fx.MakeGate(IncreasingInX1(), Policy(E::PenaltyAndExtraObjective), { length.get() });
    auto& gate = *holder;
    REQUIRE(gate.ObjectiveCount() == 3);

    SECTION("a failed prediction returns ErrMax for every objective and counts once")
    {
        auto ind = GateFixture::MakeIndividual(fx.Parse("X1"));
        ind.Genotype.Nodes().front().HashValue = static_cast<Operon::Hash>(0xDEADBEEFDEADBEEFULL);

        auto const calls = gate.CallCount.load();
        auto const fit = gate(fx.rng, ind, buf);
        REQUIRE(fit.size() == gate.ObjectiveCount());
        for (auto v : fit) {
            CHECK(v == EvaluatorBase::ErrMax);
        }
        CHECK(gate.CallCount.load() == calls + 1);
    }

    SECTION("a failed score returns ErrMax for every objective")
    {
        // log of a negative argument: every prediction is NaN, the value evaluator scores ErrMax.
        auto const ind = GateFixture::MakeIndividual(fx.Parse("log(X1 - 10)"));
        auto const fit = gate(fx.rng, ind, buf);
        REQUIRE(fit.size() == 3);
        for (auto v : fit) {
            CHECK(v == EvaluatorBase::ErrMax);
        }
    }
}

TEST_CASE("UnknownViolation of zero still yields an infeasible verdict", "[shape-adapter]")
{
    GateFixture fx;
    // abs has no differentiation rule, so the derivative bound cannot be certified.
    auto const ind = GateFixture::MakeIndividual(fx.Parse("abs(X1)"));
    auto buf = fx.Buffer();
    Operon::Evaluator<ScalarDispatch> plain { &fx.problem, &fx.dtable, Operon::NMSE {} };
    auto const err = plain(fx.rng, ind, buf)[0];
    using E = ShapeConstraintEnforcement;

    SECTION("measurement")
    {
        ShapeCertifier certifier { &fx.problem, &fx.dtable, IncreasingInX1(), Operon::Scalar { 0 } };
        auto const summary = certifier.MeasureCanonical(ind.Genotype);
        REQUIRE(summary.Measurements.size() == 1);
        CHECK_FALSE(summary.Measurements[0].Certified);
        CHECK(summary.Violation == Operon::Scalar { 0 });
        CHECK_FALSE(summary.Feasible);
    }

    SECTION("penalty adds nothing for an uncertified model")
    {
        auto const gate = fx.MakeGate(IncreasingInX1(), Policy(E::Penalty, /*unknown=*/0, /*weight=*/5));
        auto const fit = (*gate)(fx.rng, ind, buf);
        REQUIRE(fit.size() == 1);
        CHECK(fit[0] == Catch::Approx(err).margin(1e-6));
    }

    SECTION("hard-reject rejects an uncertified model")
    {
        auto const gate = fx.MakeGate(IncreasingInX1(), Policy(E::HardReject, /*unknown=*/0));
        gate->SetWorstValue(7);
        auto const fit = (*gate)(fx.rng, ind, buf);
        REQUIRE(fit.size() == 1);
        CHECK(fit[0] == Operon::Scalar { 7 });
        CHECK(gate->Violations() == 1);
    }
}

// ── exact transform ────────────────────────────────────────────────────────────

TEST_CASE("The adapter certifies against the exact transform its base evaluator applies", "[shape-adapter][scaling]")
{
    GateFixture fx;
    auto const ind = GateFixture::MakeIndividual(fx.tree);
    auto buf = fx.Buffer();
    using E = ShapeConstraintEnforcement;

    SECTION("the evaluator's pair rejects a model the canonical fit accepts")
    {
        // Value bound [-4.5, 4.5]: the raw (canonical, scaling disabled) [-4, 4] is feasible; the evaluator's
        // (-2, 3) pair gives [-5, 11], which is not.
        auto const gate
            = fx.MakeFixedGate(IdentityBound(-4.5F, 4.5F), Policy(E::HardReject), Operon::LinearScaling { -2.0, 3.0 });
        gate->SetWorstValue(7);

        REQUIRE(gate->Certifier().MeasureCanonical(fx.tree).Feasible);
        auto const fit = (*gate)(fx.rng, ind, buf);
        REQUIRE(fit.size() == 1);
        CHECK(fit[0] == Operon::Scalar { 7 });
        CHECK(gate->Predictions.load() == 1);
        CHECK(gate->Violations() == 1);
        CHECK_FALSE(gate->Certifier().Measure(fx.tree, Operon::LinearScaling { -2.0, 3.0 }).Feasible);
    }

    SECTION("the evaluator's pair accepts a model the canonical fit rejects")
    {
        auto const gate
            = fx.MakeFixedGate(IdentityBound(-3.0F, 3.0F), Policy(E::HardReject), Operon::LinearScaling { 0.5, 0.0 });
        gate->SetWorstValue(7);

        REQUIRE_FALSE(gate->Certifier().MeasureCanonical(fx.tree).Feasible);
        auto const fit = (*gate)(fx.rng, ind, buf);
        REQUIRE(fit.size() == 1);
        CHECK(fit[0] != Operon::Scalar { 7 });
        CHECK(std::isfinite(fit[0]));
        CHECK(gate->Violations() == 0);
        CHECK(gate->Certifier().Measure(fx.tree, Operon::LinearScaling { 0.5, 0.0 }).Feasible);
    }

    SECTION("a disabled transform is certified raw")
    {
        auto const gate = fx.MakeFixedGate(IdentityBound(-4.5F, 4.5F), Policy(E::HardReject), std::nullopt);
        CHECK(std::isfinite((*gate)(fx.rng, ind, buf)[0]));
        CHECK(gate->Certifier().Measure(fx.tree, std::nullopt).Feasible);
    }

    SECTION("the scratch buffer may be larger than the training range and need not be clean")
    {
        auto const gate
            = fx.MakeFixedGate(IdentityBound(-3.0F, 3.0F), Policy(E::HardReject), Operon::LinearScaling { 0.5, 0.0 });
        std::vector<Operon::Scalar> large(
            3 * static_cast<std::size_t>(GateFixture::Nrow), std::numeric_limits<Operon::Scalar>::quiet_NaN());
        auto const fit = (*gate)(fx.rng, ind, large);
        REQUIRE(fit.size() == 1);
        CHECK(std::isfinite(fit[0]));
        CHECK(gate->Certifier().Measure(fx.tree, Operon::LinearScaling { 0.5, 0.0 }).Feasible);
    }
}

TEST_CASE("ShapeCertifier measurements are identified by coefficients, bound mode and exact transform",
    "[shape-adapter][scaling]")
{
    GateFixture fx;
    ShapeCertifier certifier { &fx.problem, &fx.dtable, IdentityBound(0.0F, 1.0F) };

    auto const raw = certifier.Measure(fx.tree, std::nullopt);
    auto const scaled = certifier.Measure(fx.tree, Operon::LinearScaling { -2.0, 3.0 });
    // Raw X1 - X2 over [1,5]^2 is [-4, 4] -> violation 4 + 3; scaled by (-2, 3) it is [-5, 11] -> 5 + 10.
    CHECK(raw.Violation == Catch::Approx(7.0).margin(1e-4));
    CHECK(scaled.Violation == Catch::Approx(15.0).margin(1e-4));
    CHECK(certifier.Measure(fx.tree, Operon::LinearScaling { -2.0, 3.0 }).Violation == scaled.Violation);

    // A different coefficient set changes the measured bound.
    auto changed = fx.tree;
    changed.SetCoefficients(std::array<Operon::Scalar, 2> { Operon::Scalar { 2 }, Operon::Scalar { 1 } });
    CHECK(certifier.Measure(changed, std::nullopt).Violation != raw.Violation);
}

// ── non-Lamarckian restoration ─────────────────────────────────────────────────

TEST_CASE("Non-Lamarckian restoration does not re-certify the restored genotype", "[shape-adapter][lifecycle]")
{
    GateFixture fx;
    using E = ShapeConstraintEnforcement;
    auto buf = fx.Buffer();

    // Coefficient order follows the postfix node array (X2 precedes X1), so address the X1 weight by identity.
    // f = w1*X1 - w2*X2, so d/dX1 = w1: a stored w1 = -0.5 violates d/dX1 >= 0 by 0.5.
    auto const x1Hash = fx.ds.GetVariable("X1")->Hash;
    auto const x1Weight = [x1Hash](Operon::Tree const& t) {
        for (auto const& n : t.Nodes()) {
            if (n.IsVariable() && n.HashValue == x1Hash) {
                return n.Value;
            }
        }
        return Operon::Scalar { 0 };
    };
    auto ind = GateFixture::MakeIndividual(fx.tree);
    for (auto& n : ind.Genotype.Nodes()) {
        if (n.IsVariable() && n.HashValue == x1Hash) {
            n.Value = Operon::Scalar { -0.5 };
        }
    }
    REQUIRE(ind.Genotype.GetCoefficients().size() == 2);
    REQUIRE(x1Weight(ind.Genotype) == Operon::Scalar { -0.5 });

    ConstantCoefficientOptimizer optimizer { &fx.problem, Operon::Scalar { 1 } }; // optimized f = X1 - X2: error 0
    Operon::CoefficientOptimizer coeffOptimizer { &optimizer };
    Operon::RandomGenerator rng { 1 };

    auto const holder = fx.MakeGate(IncreasingInX1(), Policy(E::HardReject));
    auto& gate = *holder;
    gate.SetWorstValue(7);
    Operon::OptimizerAccounting accounting;
    ScoreIndividual(rng, ind, gate, &coeffOptimizer, 1.0, 0.0, Operon::Span<Operon::Scalar> { buf }, accounting);

    // The fitness describes the evaluated (optimized-coefficient) model: feasible, error 0, one prediction pass.
    CHECK(gate.Predictions.load() == 1);
    CHECK(gate.Violations() == 0);
    REQUIRE(ind.Fitness.size() == 1);
    CHECK(ind.Fitness[0] == Catch::Approx(0.0).margin(1e-5));

    // The stored (infeasible) coefficients are restored; certifying that model is an explicit measurement.
    CHECK(x1Weight(ind.Genotype) == Operon::Scalar { -0.5 });
    auto const summary = gate.Certifier().MeasureCanonical(ind.Genotype);
    CHECK_FALSE(summary.Feasible);
    CHECK(summary.Violation == Catch::Approx(0.5).margin(1e-5));
}

// ── composition and counters ───────────────────────────────────────────────────

TEST_CASE("MultiEvaluator rejects a shape-constrained adapter as a child", "[shape-adapter][composition]")
{
    GateFixture fx;
    using E = ShapeConstraintEnforcement;
    auto const length = Length(&fx.problem);
    auto const holder = fx.MakeGate(IncreasingInX1(), Policy(E::Penalty));
    auto const fixedHolder = fx.MakeFixedGate(IdentityBound(-4.5F, 4.5F), Policy(E::HardReject), std::nullopt);
    auto const spec
        = std::optional { ShapeConstraintSpec { .Constraints = IncreasingInX1(), .Policy = Policy(E::HardReject) } };

    MultiEvaluator composition { &fx.problem };

    SECTION("the adapter is rejected at registration, whatever its base type or erasure, and nothing is registered")
    {
        CHECK_THROWS_AS(composition.Add(holder.get()), std::invalid_argument);
        CHECK_THROWS_AS(composition.Add(fixedHolder.get()), std::invalid_argument);
        auto const erased
            = MakeValueEvaluator<Evaluator<ScalarDispatch>>(spec, &fx.dtable, &fx.problem, &fx.dtable, Operon::NMSE {});
        CHECK_THROWS_AS(composition.Add(erased.get()), std::invalid_argument);
        CHECK(composition.Evaluators().empty());

        composition.Add(length.get());
        CHECK_THROWS_AS(composition.Add(holder.get()), std::invalid_argument); // at any position
        CHECK(composition.Evaluators().size() == 1);
    }

    SECTION("the error names the unsupported composition")
    {
        try {
            composition.Add(holder.get());
            FAIL("MultiEvaluator::Add accepted a ShapeConstrainedValueEvaluator");
        } catch (std::invalid_argument const& e) {
            auto const message = std::string { e.what() };
            CHECK(message.find("ShapeConstrainedValueEvaluator") != std::string::npos);
        }
    }

    SECTION("a composition holding the adapter cannot be built indirectly through nesting")
    {
        MultiEvaluator inner { &fx.problem };
        CHECK_THROWS_AS(inner.Add(holder.get()), std::invalid_argument);
        MultiEvaluator outer { &fx.problem };
        CHECK_NOTHROW(outer.Add(&inner)); // an empty, adapter-free inner composition is an ordinary child
    }

    SECTION("the adapter itself is the evaluator handed to the algorithm and owns the whole vector")
    {
        auto const withLength = fx.MakeGate(IncreasingInX1(), Policy(E::PenaltyAndExtraObjective), { length.get() });
        auto buf = fx.Buffer();
        auto const infeasible = GateFixture::MakeIndividual(fx.Parse("X2 - X1"));
        auto const fit = (*withLength)(fx.rng, infeasible, buf);
        CHECK(fit.size() == 3); // [error + penalty, length, violation]
        CHECK(withLength->Predictions.load() == 1);
    }
}

TEST_CASE("MultiEvaluator still dispatches ordinary children through operator()", "[shape-adapter][composition]")
{
    GateFixture fx;
    auto const length = Length(&fx.problem);
    auto const ind = GateFixture::MakeIndividual(fx.Parse("X2 - X1"));
    auto buf = fx.Buffer();
    PredictionCountingEvaluator counting { &fx.problem, &fx.dtable, Operon::NMSE {} };

    MultiEvaluator valueFirst { &fx.problem };
    valueFirst.Add(&counting);
    valueFirst.Add(length.get());
    MultiEvaluator valueLast { &fx.problem };
    valueLast.Add(length.get());
    valueLast.Add(&counting);

    auto const a = valueFirst(fx.rng, ind, buf);
    auto const b = valueLast(fx.rng, ind, buf);
    REQUIRE(a.size() == 2);
    REQUIRE(b.size() == 2);
    CHECK(a[0] == b[1]); // a child's objective does not depend on its position
    CHECK(a[1] == b[0]);
    CHECK(counting.Predictions.load() == 2); // one prediction pass per composed call
    CHECK(counting.CallCount.load() == 2);
    CHECK(valueFirst.StructuralFitnessReusable());
}

TEST_CASE("The adapter aggregates its structural objectives' accounting", "[shape-adapter]")
{
    GateFixture fx;
    using E = ShapeConstraintEnforcement;
    auto const length = Length(&fx.problem);
    auto const ind = GateFixture::MakeIndividual(fx.tree);
    auto buf = fx.Buffer();
    auto const holder = fx.MakeGate(IncreasingInX1(), Policy(E::HardReject), { length.get() });
    auto& gate = *holder;
    gate.SetBudget(3);

    CHECK_FALSE(gate.BudgetExhausted());
    static_cast<void>(gate(fx.rng, ind, buf));
    static_cast<void>(gate(fx.rng, ind, buf));
    auto const [residual, jacobian, calls] = gate.Stats();
    CHECK(residual == 2);
    CHECK(jacobian == 0);
    CHECK(calls == 2);
    CHECK_FALSE(gate.BudgetExhausted());
    static_cast<void>(gate(fx.rng, ind, buf));
    CHECK(gate.BudgetExhausted());
}

TEST_CASE("The adapter's scoring stats and Reset cover its structural objectives", "[shape-adapter][stats]")
{
    GateFixture fx;
    using E = ShapeConstraintEnforcement;
    auto const length = Length(&fx.problem);
    auto const ind = GateFixture::MakeIndividual(fx.tree);
    auto buf = fx.Buffer();
    auto const holder = fx.MakeGate(IncreasingInX1(), Policy(E::HardReject), { length.get() });
    auto& gate = *holder;

    static_cast<void>(gate(fx.rng, ind, buf)); // one scoring prediction on the adapter's base evaluator
    // Charge the structural objective with scoring work, as an evaluator that takes a residual pass and a Jacobian
    // in its own scoring would.
    length->ResidualEvaluations += 4;
    length->JacobianEvaluations += 3;

    auto const [residual, jacobian, calls] = gate.Stats();
    CHECK(residual == 5); // 1 prediction + the structural objective's 4
    CHECK(jacobian == 3);
    CHECK(calls == 1); // the adapter's own, not the structural objectives'
    CHECK(gate.TotalEvaluations() == 1); // own counters only

    gate.Reset();
    CHECK(gate.Stats() == std::tuple<std::size_t, std::size_t, std::size_t> { 0, 0, 0 });
    CHECK(length->Stats() == std::tuple<std::size_t, std::size_t, std::size_t> { 0, 0, 0 });
    CHECK(gate.TotalEvaluations() == 0);
}

TEST_CASE("Optimizer work of a shape-constrained fit is recorded once, by the orchestration", "[shape-adapter][stats]")
{
    GateFixture fx;
    using E = ShapeConstraintEnforcement;
    auto const length = Length(&fx.problem);
    auto ind = GateFixture::MakeIndividual(fx.tree);
    auto buf = fx.Buffer();
    auto const holder = fx.MakeGate(IncreasingInX1(), Policy(E::HardReject), { length.get() });
    auto& gate = *holder;
    ConstantCoefficientOptimizer optimizer { &fx.problem, Operon::Scalar { 1 } };
    Operon::CoefficientOptimizer coeffOptimizer { &optimizer };
    Operon::OptimizerAccounting accounting;
    Operon::RandomGenerator rng { 1 };

    ScoreIndividual(rng, ind, gate, &coeffOptimizer, 1.0, 1.0, Operon::Span<Operon::Scalar> { buf }, accounting);

    // The fit did 1 residual evaluation: the run's accounting has it, neither the adapter nor its objectives do.
    auto const work = accounting.Snapshot();
    CHECK(work.Residual == 1);
    CHECK(work.Jacobian == 0);
    auto const [residual, jacobian, calls] = gate.Stats();
    CHECK(residual == 1); // the adapter's single prediction pass
    CHECK(jacobian == 0);
    CHECK(calls == 1);
    CHECK(std::get<0>(length->Stats()) == 0);
}

} // namespace Operon::Test
