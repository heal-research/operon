// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research
//
// Demonstration: symbolic regression using runtime-registered primitives.
//
// All mathematical functions beyond the basic arithmetic set are registered
// at runtime via RegisterUnaryFunction / RegisterBinaryFunction.  The example
// shows that user-defined callables participate in tree generation, evaluation,
// and coefficient optimisation on equal footing with built-in operators.
//
// Custom primitives added in this example:
//   sincos(x)   = sin(x) + cos(x)    (unary, explicit derivative)
//   gaussian(x) = exp(-x * x)        (unary, Jet auto-diff fallback)
//   hypot(a, b) = sqrt(a^2 + b^2)    (binary, Jet auto-diff fallback)
//
// Usage:
//   custom_primitives <path-to-Poly-10.csv>
//
// The Poly-10 dataset ships with the repository under data/Poly-10.csv.

#include <algorithm>
#include <cmath>
#include <fmt/core.h>
#include <limits>
#include <taskflow/taskflow.hpp>
#include <thread>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/symbol_library.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/hash/hash.hpp"
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

auto main(int argc, char** argv) -> int
{
    if (argc < 2) {
        fmt::print(stderr, "usage: {} <path-to-Poly-10.csv>\n", argv[0]);
        fmt::print(stderr, "  (dataset ships with the repo under data/Poly-10.csv)\n");
        return 1;
    }

    Operon::Dataset dataset(argv[1], /*hasHeader=*/true);

    Operon::Problem problem(std::make_unique<Operon::Dataset>(dataset));
    problem.SetTrainingRange({ 0, 250 });
    problem.SetTestRange({ 250, 500 });
    problem.SetTarget("Y");

    auto inputs = dataset.VariableHashes();
    std::erase(inputs, dataset.GetVariable("Y")->Hash);
    problem.SetInputs(inputs);

    // Arithmetic built-ins form the structural backbone of expressions.
    // No transcendental functions from NodeType — those come from our
    // runtime-registered callables below.
    problem.ConfigurePrimitiveSet(
        Operon::NodeType::Constant | Operon::NodeType::Variable |
        Operon::NodeType::Add | Operon::NodeType::Sub |
        Operon::NodeType::Mul | Operon::NodeType::Div);

    // register custom primitives
    using DT = Operon::DefaultDispatch;
    DT dtable;
    auto& pset = problem.GetPrimitiveSet();

    // sincos(x) = sin(x) + cos(x) — explicit derivative provided
    Operon::RegisterUnaryFunction<DT, Operon::Scalar>(dtable, pset,
        { .Hash = Operon::Hasher{}("sincos"),
          .Name = "sincos", .Desc = "sin(x) + cos(x)",
          .Arity = 1, .Frequency = 2 },
        [](auto v) { using std::sin, std::cos; return sin(v) + cos(v); },
        [](auto v) { using std::sin, std::cos; return cos(v) - sin(v); });

    // gaussian(x) = exp(-x²) — derivative via Jet<T,1> auto-diff
    Operon::RegisterUnaryFunction<DT, Operon::Scalar>(dtable, pset,
        { .Hash = Operon::Hasher{}("gaussian"),
          .Name = "gaussian", .Desc = "exp(-x * x)",
          .Arity = 1, .Frequency = 2 },
        [](auto v) { using std::exp; return exp(-v * v); });

    // hypot(a, b) = sqrt(a² + b²) — derivative via Jet<T,1> auto-diff
    Operon::RegisterBinaryFunction<DT, Operon::Scalar>(dtable, pset,
        { .Hash = Operon::Hasher{}("hypot"),
          .Name = "hypot", .Desc = "sqrt(a^2 + b^2)",
          .Arity = 2, .Frequency = 1 },
        [](auto a, auto b) { using std::sqrt; return sqrt((a * a) + (b * b)); });

    fmt::print("Primitives in use:\n");
    for (auto const& node : pset.EnabledPrimitives()) {
        fmt::print("  {:<12}  arity={}\n", node.Name(), node.Arity);
    }
    fmt::print("\n");

    // set up GP operators
    constexpr size_t MaxLength = 50;
    constexpr size_t MaxDepth  = 10;

    auto [arityMin, arityMax] = pset.FunctionArityLimits();

    Operon::BalancedTreeCreator creator { &pset, problem.GetInputs() };

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
    Operon::RemoveSubtreeMutation  removeSub  { pset };
    Operon::InsertSubtreeMutation  insertSub  { &creator, &coeffInit, MaxDepth, MaxLength };

    Operon::MultiMutation mutator;
    mutator.Add(&onePoint,   1.0);
    mutator.Add(&changeFunc, 1.0);
    mutator.Add(&changeVar,  1.0);
    mutator.Add(&removeSub,  1.0);
    mutator.Add(&insertSub,  1.0);

    // evaluator, coefficient optimizer, selectors, offspring generator
    Operon::Evaluator<DT> evaluator { &problem, &dtable, Operon::MSE{}, /*linearScaling=*/true };
    evaluator.SetBudget(std::numeric_limits<size_t>::max()); // generations is the only stop criterion

    Operon::LevenbergMarquardtOptimizer<DT, Operon::OptimizerType::Eigen> lmOptimizer {
        &dtable, &problem
    };
    Operon::CoefficientOptimizer coeffOpt { &lmOptimizer };

    auto comp = [](auto const& a, auto const& b) { return a[0] < b[0]; };
    Operon::TournamentSelector femaleSelector { comp };
    Operon::TournamentSelector   maleSelector { comp };

    Operon::BasicOffspringGenerator generator {
        &evaluator, &crossover, &mutator, &femaleSelector, &maleSelector, &coeffOpt
    };
    Operon::ReplaceWorstReinserter reinserter { comp };

    Operon::GeneticAlgorithmConfig config {
        .Generations    = 100,
        .Evaluations    = std::numeric_limits<size_t>::max(),
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

    size_t gen = 0;
    auto report = [&]() {
        ++gen;
        if (gen % 10 != 0) { return; }
        auto const* first = gp.Individuals().data();
        auto const* last  = first + config.PopulationSize;
        auto const* best  = std::min_element(first, last,
            [](auto const& a, auto const& b) { return a[0] < b[0]; });
        fmt::print("  gen {:4d}  MSE(train)={:.6f}  len={}\n",
            gen, best->Fitness[0], best->Genotype.Length());
    };

    fmt::print("Running GP ({} generations, population {})...\n\n",
        config.Generations, config.PopulationSize);

    gp.Run(executor, rng, report);

    auto const* first = gp.Individuals().data();
    auto const* last  = first + config.PopulationSize;
    auto const* best  = std::min_element(first, last,
        [](auto const& a, auto const& b) { return a[0] < b[0]; });

    fmt::print("\nBest model (MSE={:.6f}, length={}):\n  {}\n",
        best->Fitness[0],
        best->Genotype.Length(),
        Operon::InfixFormatter::Format(best->Genotype, dataset, 6));

    return 0;
}
