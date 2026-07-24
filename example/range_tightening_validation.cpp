// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
//
// Empirical validation of TightenRange against real evolved trees, not
// just synthetic hand-built ones. Runs a GP evolution on Poly-10, then for
// each individual in the final population compares naive vs. TightenRange
// enclosure width over the variable domains, and checks soundness against
// dense point sampling (tightened must never exclude a value the tree
// actually attains).
//
// Usage: range_tightening_validation <path-to-Poly-10.csv> [domain-shrink]
//   domain-shrink (default 1.0): scales each variable's domain box around
//   its midpoint. Mean-value form is a local (first-order) approximation,
//   so its benefit over naive shrinks as the box widens - pass e.g. 0.1 or
//   0.01 to see the effect on a narrower box.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <fmt/core.h>
#include <limits>
#include <random>
#include <string>
#include <taskflow/taskflow.hpp>
#include <thread>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/tree_diff.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/interpreter/interval_evaluator.hpp"
#include "operon/interpreter/range_tightening.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/optimizer/optimizer.hpp"

namespace {

using DT = Operon::ScalarDispatch;
using Interp = Operon::Interpreter<Operon::Scalar, DT>;
using IE = Operon::IntervalEvaluator;

auto MakeDomains(Operon::Dataset const& ds, Operon::Range range, std::vector<Operon::Hash> const& inputs, float shrink = 1.0F) -> IE::DomainMap
{
    IE::DomainMap domains;
    for (auto h : inputs) {
        auto vals = ds.GetValues(h).subspan(range.Start(), range.Size());
        auto [lo, hi] = std::ranges::minmax(vals);
        auto const mid = (lo + hi) / 2;
        auto const halfWidth = (hi - lo) / 2 * shrink;
        domains[h] = {mid - halfWidth, mid + halfWidth};
    }
    return domains;
}

struct Stats {
    std::size_t total{};
    std::size_t empty{};
    std::size_t unbounded{}; // naive diameter itself non-finite
    std::size_t fellBackToNaive{};
    std::size_t counted{}; // contributes to totalWidthReductionPct
    std::size_t soundnessViolations{};
    double totalWidthReductionPct{};
    double naiveNanosTotal{};
    double tightenedNanosTotal{};
};

} // namespace

auto main(int argc, char** argv) -> int // NOLINT(bugprone-exception-escape)
{
    if (argc < 2) {
        fmt::print(stderr, "usage: {} <path-to-Poly-10.csv>\n", argv[0]);
        return 1;
    }

    Operon::Dataset const dataset(argv[1], /*hasHeader=*/true);

    Operon::Problem problem(std::make_unique<Operon::Dataset>(dataset));
    problem.SetTrainingRange({ 0, 250 });
    problem.SetTestRange({ 250, 500 });
    problem.SetTarget("Y");

    auto inputs = dataset.VariableHashes();
    std::erase(inputs, dataset.GetVariable("Y").value().Hash);
    problem.SetInputs(inputs);

    problem.ConfigurePrimitiveSet(
        Operon::NodeType::Constant | Operon::NodeType::Variable |
        Operon::BuiltinOp::Add | Operon::BuiltinOp::Sub |
        Operon::BuiltinOp::Mul | Operon::BuiltinOp::Div);

    DT dtable;
    auto& pset = problem.GetPrimitiveSet();

    constexpr std::size_t MaxLength = 40;
    constexpr std::size_t MaxDepth  = 8;

    auto [arityMin, arityMax] = pset.FunctionArityLimits();

    Operon::BalancedTreeCreator creator { &pset, problem.GetInputs(), /* bias= */ 0.0, MaxLength };

    Operon::NormalCoefficientInitializer coeffInit;
    coeffInit.ParameterizeDistribution(Operon::Scalar{0}, Operon::Scalar{1});

    Operon::UniformTreeInitializer treeInit { &creator };
    treeInit.ParameterizeDistribution(arityMin + 1, MaxLength);
    treeInit.SetMinDepth(1);
    treeInit.SetMaxDepth(MaxDepth);

    Operon::SubtreeCrossover crossover { 0.9, MaxDepth, MaxLength };

    Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> onePoint;
    Operon::ChangeFunctionMutation changeFunc { pset };
    Operon::ChangeVariableMutation changeVar  { problem.GetInputs() };
    Operon::RemoveSubtreeMutation  removeSub  { &creator, &coeffInit, MaxDepth };
    Operon::InsertSubtreeMutation  insertSub  { &creator, &coeffInit, MaxDepth, MaxLength };

    Operon::MultiMutation mutator;
    mutator.Add(&onePoint,   1.0);
    mutator.Add(&changeFunc, 1.0);
    mutator.Add(&changeVar,  1.0);
    mutator.Add(&removeSub,  1.0);
    mutator.Add(&insertSub,  1.0);

    Operon::Evaluator<DT> evaluator { &problem, &dtable, Operon::MSE{}, /*linearScaling=*/true };
    evaluator.SetBudget(std::numeric_limits<std::size_t>::max());

    Operon::LevenbergMarquardtOptimizer<DT, Operon::OptimizerType::Eigen> lmOptimizer {
        &dtable, &problem
    };
    Operon::CoefficientOptimizer const coeffOpt { &lmOptimizer };

    auto comp = [](auto const& a, auto const& b) -> auto { return a[0] < b[0]; };
    Operon::TournamentSelector femaleSelector { comp };
    Operon::TournamentSelector   maleSelector { comp };

    Operon::BasicOffspringGenerator generator {
        &evaluator, &crossover, &mutator, &femaleSelector, &maleSelector, &coeffOpt
    };
    Operon::ReplaceWorstReinserter reinserter { comp };

    Operon::GeneticAlgorithmConfig config {
        .Generations    = 50,
        .Evaluations    = std::numeric_limits<std::size_t>::max(),
        .Iterations     = 2,
        .PopulationSize = 200,
        .PoolSize       = 200,
        .Seed           = 42,
    };

    Operon::RandomGenerator rng { config.Seed };
    Operon::GeneticProgrammingAlgorithm gp {
        config, &problem, &treeInit, &coeffInit, &generator, &reinserter
    };

    tf::Executor executor(std::thread::hardware_concurrency());
    fmt::print("Running GP ({} generations, population {}) on Poly-10...\n",
        config.Generations, config.PopulationSize);
    gp.Run(executor, rng, nullptr);
    fmt::print("Done. Validating TightenRange against the final population ({} trees)...\n\n",
        config.PopulationSize);

    float const shrink = argc > 2 ? std::stof(argv[2]) : 1.0F;
    auto domains = MakeDomains(dataset, problem.TrainingRange(), inputs, shrink);

    Stats stats;
    constexpr int nSamples = 200;
    std::uniform_real_distribution<Operon::Scalar> unit(0.F, 1.F);

    int debugPrinted = 0;
    for (auto const& ind : gp.Individuals()) {
        auto const& tree  = ind.Genotype;
        auto const coeff  = tree.GetCoefficients();

        auto const t0 = std::chrono::steady_clock::now();
        auto const naive = IE(&tree, domains).Evaluate(coeff);
        auto const t1 = std::chrono::steady_clock::now();
        auto const tightened = Operon::TightenRange(tree, domains, coeff);
        auto const t2 = std::chrono::steady_clock::now();

        if (debugPrinted < 3) {
            auto gdag = Operon::BuildVariableGradientDag(tree, coeff);
            fmt::print("[sample] len={:3d} vars={} naive=[{:.4f},{:.4f}] tightened=[{:.4f},{:.4f}]\n",
                tree.Length(), gdag.Variables.size(), naive.inf(), naive.sup(),
                tightened.inf(), tightened.sup());
            ++debugPrinted;
        }

        ++stats.total;
        stats.naiveNanosTotal += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t1 - t0).count());
        stats.tightenedNanosTotal += static_cast<double>(std::chrono::duration_cast<std::chrono::nanoseconds>(t2 - t1).count());

        if (naive.is_empty()) { ++stats.empty; continue; }

        auto const naiveWidth = naive.diameter();
        auto const tightWidth = tightened.diameter();
        if (!std::isfinite(naiveWidth) || naiveWidth <= 0) { ++stats.unbounded; }
        else {
            ++stats.counted;
            auto const reduction = 100.0 * (1.0 - static_cast<double>(tightWidth) / static_cast<double>(naiveWidth));
            if (reduction < 1e-6) { ++stats.fellBackToNaive; }
            stats.totalWidthReductionPct += reduction;
        }

        // Dense random sampling within the domain box as an approximate
        // ground truth: TightenRange must never exclude a value the tree
        // actually attains somewhere in the box.
        std::vector<std::vector<Operon::Scalar>> pointData(inputs.size(), std::vector<Operon::Scalar>(nSamples));
        for (std::size_t vi = 0; vi < inputs.size(); ++vi) {
            auto const [lo, hi] = domains[inputs[vi]];
            for (auto& v : pointData[vi]) { v = lo + unit(rng) * (hi - lo); }
        }
        std::vector<std::string> names;
        names.reserve(inputs.size());
        for (auto h : inputs) { names.push_back(dataset.GetVariable(h)->Name); }
        Operon::Dataset const sampleDs(names, pointData);
        auto const values = Interp::Evaluate(tree, sampleDs, Operon::Range{0, nSamples}, Operon::Span<Operon::Scalar const>(coeff));

        for (auto v : values) {
            if (!std::isfinite(v)) { continue; }
            if (v < tightened.inf() - 1e-3F || v > tightened.sup() + 1e-3F) {
                ++stats.soundnessViolations;
            }
        }
    }

    auto const avgReduction = stats.totalWidthReductionPct / static_cast<double>(std::max(stats.counted, std::size_t{1}));
    auto const naiveAvgUs = stats.naiveNanosTotal / static_cast<double>(std::max(stats.total, std::size_t{1})) / 1000.0;
    auto const tightAvgUs = stats.tightenedNanosTotal / static_cast<double>(std::max(stats.total, std::size_t{1})) / 1000.0;

    fmt::print("=== TightenRange empirical validation (Poly-10, real evolved trees) ===\n");
    fmt::print("Trees evaluated:          {}\n", stats.total);
    fmt::print("Naive enclosure empty:    {}\n", stats.empty);
    fmt::print("Naive enclosure unbounded/degenerate (excluded from reduction avg): {}\n", stats.unbounded);
    fmt::print("Counted (finite, nonzero naive width): {}\n", stats.counted);
    fmt::print("  of which fell back to naive (no improvement): {} ({:.1f}%)\n",
        stats.fellBackToNaive, 100.0 * static_cast<double>(stats.fellBackToNaive) / static_cast<double>(std::max(stats.counted, std::size_t{1})));
    fmt::print("Average enclosure width reduction (counted trees): {:.2f}%\n", avgReduction);
    fmt::print("Avg naive eval time:      {:.3f} us\n", naiveAvgUs);
    fmt::print("Avg TightenRange time:    {:.3f} us  ({:.1f}x naive)\n", tightAvgUs, tightAvgUs / std::max(naiveAvgUs, 1e-9));
    fmt::print("Soundness violations:     {}  (must be 0)\n", stats.soundnessViolations);

    return stats.soundnessViolations == 0 ? 0 : 1;
}
