// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Phase 9: NSGA2 integration test — full evolutionary run + pappus bounds check.

#include <catch2/catch_test_macros.hpp>

#include <cmath>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/affine_evaluator.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"

namespace Operon::Test {

using S  = Operon::Scalar;
using IE = IntervalEvaluator;
using AE = AffineEvaluator;

// ---------------------------------------------------------------------------
// Phase 9: NSGA2 integration — full evolutionary run, Pareto-front bounds check
// ---------------------------------------------------------------------------
// Runs NSGA2 with a TypeCoherent primitive set (arithmetic + Exp, Log, Sin,
// Cos, Pow, Square) and tree sizes representative of a real NSGP run.
// After the run, evaluates every Pareto-front individual with IntervalEvaluator
// and checks that each result is either:
//   - a valid enclosure  (inf <= sup), or
//   - an empty interval  (NaN bounds — valid for out-of-domain operations such
//     as log of a negative interval).

TEST_CASE("NSGA2 Pareto front: interval bounds are valid", "[pappus][nsgp]")
{
    // Friedman-I: Y = 10*sin(pi*X1*X2) + 20*(X3-0.5)^2 + 10*X4 + 5*X5
    // All inputs uniform in [0,1].  Hard enough to require transcendentals and
    // multi-term trees — representative of real NSGP output.
    constexpr int NRows = 200;
    constexpr int NVars = 5;
    Operon::RandomGenerator dsRng(1234);
    std::uniform_real_distribution<Operon::Scalar> uDist(S{0}, S{1});

    std::vector<std::vector<Operon::Scalar>> cols(NVars + 1, std::vector<Operon::Scalar>(NRows));
    for (int r = 0; r < NRows; ++r) {
        for (int c = 0; c < NVars; ++c) { cols[c][r] = uDist(dsRng); }
        auto const& x1 = cols[0][r]; auto const& x2 = cols[1][r];
        auto const& x3 = cols[2][r]; auto const& x4 = cols[3][r];
        auto const& x5 = cols[4][r];
        cols[NVars][r] = S{10} * std::sin(S{3.14159265358979323846L} * x1 * x2)
                       + S{20} * (x3 - S{0.5}) * (x3 - S{0.5})
                       + S{10} * x4 + S{5} * x5;
    }
    Operon::Dataset ds({"X1", "X2", "X3", "X4", "X5", "Y"}, cols);

    Operon::Problem problem{gsl::not_null<Operon::Dataset*>(&ds)};
    problem.SetTrainingRange({0, NRows});
    problem.SetTestRange({0, NRows});
    problem.SetTarget("Y");
    problem.ConfigurePrimitiveSet(Operon::PrimitiveSet::TypeCoherent);

    std::vector<Operon::Hash> inputs{
        ds.GetVariable("X1")->Hash, ds.GetVariable("X2")->Hash,
        ds.GetVariable("X3")->Hash, ds.GetVariable("X4")->Hash,
        ds.GetVariable("X5")->Hash,
    };
    problem.SetInputs(inputs);

    using DTable = Operon::DispatchTable<Operon::Scalar>;
    DTable dtable;

    constexpr std::size_t MaxLength = 50;
    constexpr std::size_t MaxDepth  = 10;

    Operon::Evaluator<DTable> rmseEval(&problem, &dtable);
    Operon::LengthEvaluator   lenEval(&problem, MaxLength);
    Operon::MultiEvaluator    multiEval(&problem);
    multiEval.Add(&rmseEval);
    multiEval.Add(&lenEval);

    Operon::SubtreeCrossover crossover{0.9, MaxDepth, MaxLength};
    Operon::MultiMutation    mutator;

    Operon::NormalCoefficientInitializer coeffInit;
    auto& pset = problem.GetPrimitiveSet();
    Operon::BalancedTreeCreator     creator(&pset, inputs, 0.0, MaxLength);
    Operon::UniformTreeInitializer  treeInit(&creator);
    treeInit.ParameterizeDistribution(2U, MaxLength);
    treeInit.SetMaxDepth(MaxDepth);

    Operon::ChangeVariableMutation    changeVar{inputs};
    Operon::ChangeFunctionMutation    changeFunc{pset};
    Operon::RemoveSubtreeMutation     removeSubtree{pset};
    Operon::InsertSubtreeMutation     insertSubtree{&creator, &coeffInit, MaxDepth, MaxLength};
    Operon::ReplaceSubtreeMutation    replaceSubtree{&creator, &coeffInit, MaxDepth, MaxLength};
    Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> onePoint;
    mutator.Add(&onePoint,      1.0);
    mutator.Add(&changeVar,     1.0);
    mutator.Add(&changeFunc,    1.0);
    mutator.Add(&removeSubtree, 1.0);
    mutator.Add(&insertSubtree, 1.0);
    mutator.Add(&replaceSubtree, 1.0);

    Operon::CrowdedComparison       comp;
    Operon::TournamentSelector      femSel{comp};
    Operon::TournamentSelector      maleSel{comp};
    Operon::BasicOffspringGenerator generator{&multiEval, &crossover, &mutator, &femSel, &maleSel};
    Operon::KeepBestReinserter      reinserter{comp};
    Operon::EfficientBinarySorter   sorter;

    Operon::GeneticAlgorithmConfig cfg;
    cfg.PopulationSize       = 1000;
    cfg.PoolSize             = 1000;
    cfg.Generations          = 100;
    cfg.CrossoverProbability = 0.9;
    cfg.MutationProbability  = 0.25;

    Operon::NSGA2 nsga2{cfg, &problem, &treeInit, &coeffInit, &generator, &reinserter, &sorter};

    Operon::RandomGenerator rng(42);
    nsga2.Run(rng, nullptr, /*threads=*/1);

    auto const front = nsga2.Best();
    REQUIRE(!front.empty());

    IE::DomainMap dom;
    for (auto const& name : {"X1", "X2", "X3", "X4", "X5"}) {
        dom[ds.GetVariable(name)->Hash] = {S{0}, S{1}};
    }

    // Affine term budget: 5 input vars + a small headroom for nonlinear ops.
    // This is intentionally below the unbounded term count for complex trees to
    // exercise the Chebyshev condensation path.
    constexpr std::size_t TermBudget = 8;

    for (auto const& ind : front) {
        REQUIRE(ind.Genotype.Length() > 0);

        // --- IntervalEvaluator ---
        IE iev(&ind.Genotype, IE::DomainMap{dom});
        auto const enc = iev.Evaluate(ind.Genotype.GetCoefficients());
        // Empty (NaN bounds) is valid for out-of-domain ops (e.g. log of a
        // negative interval). Non-empty must satisfy inf <= sup.
        auto const ieEmpty = std::isnan(enc.inf()) || std::isnan(enc.sup());
        INFO("IE enclosure: [" << enc.inf() << ", " << enc.sup() << "]");
        REQUIRE((ieEmpty || enc.inf() <= enc.sup()));

        // --- AffineEvaluator (unbounded) — measure raw term growth ---
        // AE can throw std::exception for out-of-domain inputs (e.g. log of a
        // negative affine form); that is documented behaviour, not a bug.
        try {
            AE aev(&ind.Genotype, AE::DomainMap{dom});
            auto const r     = aev.Evaluate(ind.Genotype.GetCoefficients());
            auto const terms = aev.TermCount();
            auto const aiv   = r.to_interval();
            auto const aeEmpty = std::isnan(aiv.inf()) || std::isnan(aiv.sup());
            INFO("AE (unbounded) terms=" << terms
                 << " enclosure: [" << aiv.inf() << ", " << aiv.sup() << "]"
                 << " length=" << ind.Genotype.Length());
            REQUIRE((aeEmpty || aiv.inf() <= aiv.sup()));
        } catch (std::exception const& e) {
            INFO("AE (unbounded) threw (domain error): " << e.what());
        }

        // --- AffineEvaluator (bounded) — verify condensation cap is respected ---
        try {
            AE aevB(&ind.Genotype, AE::DomainMap{dom}, TermBudget);
            auto const rB     = aevB.Evaluate(ind.Genotype.GetCoefficients());
            auto const termsB = aevB.TermCount();
            auto const aivB   = rB.to_interval();
            auto const aeEmptyB = std::isnan(aivB.inf()) || std::isnan(aivB.sup());
            INFO("AE (cap=" << TermBudget << ") terms=" << termsB
                 << " enclosure: [" << aivB.inf() << ", " << aivB.sup() << "]");
            REQUIRE(termsB <= TermBudget);
            REQUIRE((aeEmptyB || aivB.inf() <= aivB.sup()));
        } catch (std::exception const& e) {
            INFO("AE (bounded) threw (domain error): " << e.what());
        }
    }
}

} // namespace Operon::Test
