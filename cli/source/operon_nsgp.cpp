// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <memory>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/taskflow.hpp>
#include <thread>

#include "operon/algorithms/nsga2.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/version.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/hash/hash.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/solvers/sgd.hpp"

#include "operator_factory.hpp"
#include "pareto_front.hpp"
#include "reporter.hpp"
#include "util.hpp"

namespace {
auto MakeCoeffAndMutation(bool symbolic)
    -> std::pair<std::unique_ptr<Operon::CoefficientInitializerBase>, std::unique_ptr<Operon::MutatorBase>>
{
    if (symbolic) {
        using Dist = std::uniform_int_distribution<int>;
        auto ci = std::make_unique<Operon::CoefficientInitializer<Dist>>();
        int constexpr range { 5 };
        dynamic_cast<Operon::CoefficientInitializer<Dist>*>(ci.get())->ParameterizeDistribution(-range, +range);
        auto op = std::make_unique<Operon::OnePointMutation<Dist>>();
        dynamic_cast<Operon::OnePointMutation<Dist>*>(op.get())->ParameterizeDistribution(-range, +range);
        return { std::move(ci), std::move(op) };
    }
    using Dist = std::normal_distribution<Operon::Scalar>;
    auto ci = std::make_unique<Operon::CoefficientInitializer<Dist>>();
    dynamic_cast<Operon::NormalCoefficientInitializer*>(ci.get())->ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
    auto op = std::make_unique<Operon::OnePointMutation<Dist>>();
    dynamic_cast<Operon::OnePointMutation<Dist>*>(op.get())->ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
    return { std::move(ci), std::move(op) };
}
} // anonymous namespace

auto main(int argc, char** argv) -> int // NOLINT(bugprone-exception-escape)
{
    auto opts = Operon::InitOptions("operon_gp", "Genetic programming symbolic regression");
    auto result = Operon::ParseOptions(std::move(opts), argc, argv);

    // parse and set default values
    Operon::GeneticAlgorithmConfig config {};
    config.Generations = result["generations"].as<size_t>();
    config.PopulationSize = result["population-size"].as<size_t>();
    config.PoolSize = result["pool-size"].as<size_t>();
    config.Epsilon = result["epsilon"].as<Operon::Scalar>();
    config.Evaluations = result["evaluations"].as<size_t>();
    config.Iterations = result["iterations"].as<size_t>();
    config.CrossoverProbability = result["crossover-probability"].as<Operon::Scalar>();
    config.MutationProbability = result["mutation-probability"].as<Operon::Scalar>();
    config.LocalSearchProbability = result["local-search-probability"].as<Operon::Scalar>();
    config.LamarckianProbability = result["lamarckian-probability"].as<Operon::Scalar>();
    config.TimeLimit = result["timelimit"].as<size_t>();
    config.Seed = std::random_device {}();

    Operon::Range trainingRange;
    Operon::Range testRange;
    std::unique_ptr<Operon::Dataset> dataset;
    std::string targetName;
    bool showPrimitiveSet = false;
    auto threads = std::thread::hardware_concurrency();
    auto primitiveSetConfig = Operon::PrimitiveSet::Arithmetic;

    auto maxLength = result["maxlength"].as<size_t>();
    auto maxDepth = result["maxdepth"].as<size_t>();
    auto const crossoverInternalProbability = result["crossover-internal-probability"].as<Operon::Scalar>();
    auto const symbolic = result["symbolic"].as<bool>();

    // Apply overrides from parsed options
    dataset = std::make_unique<Operon::Dataset>(result["dataset"].as<std::string>(), /*hasHeader=*/true);
    ENSURE(!dataset->IsView());
    if (result.contains("seed"))             { config.Seed = result["seed"].as<size_t>(); }
    if (result.contains("train"))            { trainingRange = Operon::ParseRange(result["train"].as<std::string>()); }
    if (result.contains("test"))             { testRange = Operon::ParseRange(result["test"].as<std::string>()); }
    if (result.contains("target"))           { targetName = result["target"].as<std::string>(); }
    if (result.contains("enable-symbols"))   { primitiveSetConfig |= Operon::ParsePrimitiveSetConfig(result["enable-symbols"].as<std::string>()); }
    if (result.contains("disable-symbols"))  { primitiveSetConfig &= ~Operon::ParsePrimitiveSetConfig(result["disable-symbols"].as<std::string>()); }
    if (result.contains("threads"))          { threads = static_cast<decltype(threads)>(result["threads"].as<size_t>()); }
    if (result.contains("show-primitives"))  { showPrimitiveSet = true; }

    try {
        if (showPrimitiveSet) {
            Operon::PrintPrimitives(primitiveSetConfig);
            return EXIT_SUCCESS;
        }

        // set the target
        auto res = dataset->GetVariable(targetName);
        if (!res) {
            fmt::print(stderr, "error: target variable {} does not exist in the dataset.", targetName);
            return EXIT_FAILURE;
        }
        auto const& target = *res;
        auto const rows { dataset->Rows<std::size_t>() };

        Operon::SetupRanges(result, *dataset, trainingRange, testRange);

        // validate training range
        if (trainingRange.Start() >= rows || trainingRange.End() > rows) {
            fmt::print(stderr, "error: the training range {}:{} exceeds the available data range ({} rows)\n", trainingRange.Start(), trainingRange.End(), dataset->Rows());
            return EXIT_FAILURE;
        }
        if (trainingRange.Start() > trainingRange.End()) {
            fmt::print(stderr, "error: invalid training range {}:{}\n", trainingRange.Start(), trainingRange.End());
            return EXIT_FAILURE;
        }

        auto inputs = Operon::BuildInputs(result, *dataset, target.Hash);

        Operon::Problem problem(std::move(dataset));
        problem.SetTrainingRange(trainingRange);
        problem.SetTestRange(testRange);
        problem.SetTarget(target.Hash);
        problem.SetInputs(inputs);
        problem.ConfigurePrimitiveSet(primitiveSetConfig);

        auto creator = ParseCreator(result["creator"].as<std::string>(), problem.GetPrimitiveSet(), problem.GetInputs(), maxLength);

        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        Operon::UniformTreeInitializer treeInitializer(creator.get());

        auto const initialMinDepth = result["creator-mindepth"].as<std::size_t>();
        auto const initialMaxDepth = result["creator-maxdepth"].as<std::size_t>();
        auto const initialMaxLength = result["creator-maxlength"].as<std::size_t>();
        treeInitializer.ParameterizeDistribution(amin + 1, initialMaxLength);
        treeInitializer.SetMinDepth(initialMinDepth);
        treeInitializer.SetMaxDepth(initialMaxDepth); // NOLINT

        auto [coeffInitializer, onePoint] = MakeCoeffAndMutation(symbolic);

        Operon::SubtreeCrossover crossover { crossoverInternalProbability, maxDepth, maxLength };
        Operon::MultiMutation mutator {};

        Operon::ChangeVariableMutation changeVar { problem.GetInputs() };
        Operon::ChangeFunctionMutation changeFunc { problem.GetPrimitiveSet() };
        Operon::ReplaceSubtreeMutation replaceSubtree { creator.get(), coeffInitializer.get(), maxDepth, maxLength };
        Operon::InsertSubtreeMutation insertSubtree { creator.get(), coeffInitializer.get(), maxDepth, maxLength };
        Operon::RemoveSubtreeMutation removeSubtree { problem.GetPrimitiveSet() };
        Operon::DiscretePointMutation discretePoint;
        for (auto v : Operon::Math::Constants) {
            discretePoint.Add(static_cast<Operon::Scalar>(v), 1);
        }
        mutator.Add(onePoint.get(), 1.0);
        mutator.Add(&changeVar, 1.0);
        mutator.Add(&changeFunc, 1.0);
        mutator.Add(&replaceSubtree, 1.0);
        mutator.Add(&insertSubtree, 1.0);
        mutator.Add(&removeSubtree, 1.0);
        mutator.Add(&discretePoint, 1.0);

        Operon::ScalarDispatch dtable;
        auto const scale = result["linear-scaling"].as<bool>();
        auto errorEvaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable, scale);
        errorEvaluator->SetBudget(config.Evaluations);

        auto optimizer = std::make_unique<Operon::LevenbergMarquardtOptimizer<decltype(dtable), Operon::OptimizerType::Eigen>>(&dtable, &problem);
        optimizer->SetIterations(config.Iterations);
        Operon::LengthEvaluator lengthEvaluator(&problem, maxLength);

        Operon::MultiEvaluator evaluator(&problem);
        evaluator.SetBudget(config.Evaluations);
        evaluator.Add(errorEvaluator.get());
        evaluator.Add(&lengthEvaluator);

        EXPECT(problem.TrainingRange().Size() > 0);

        Operon::CrowdedComparison const comp;

        auto femaleSelector = Operon::ParseSelector(result["female-selector"].as<std::string>(), comp);
        auto maleSelector = Operon::ParseSelector(result["male-selector"].as<std::string>(), comp);
        Operon::CoefficientOptimizer const cOpt { optimizer.get() };

        auto generator = Operon::ParseGenerator(result["offspring-generator"].as<std::string>(), evaluator, crossover, mutator, *femaleSelector, *maleSelector, &cOpt);
        auto reinserter = Operon::ParseReinserter(result["reinserter"].as<std::string>(), comp);

        std::unique_ptr<Operon::Zobrist> cache;
        if (result["transposition-cache"].as<bool>()) {
            Operon::RandomGenerator cacheRng(config.Seed);
            cache = std::make_unique<Operon::Zobrist>(cacheRng, static_cast<int>(maxLength));
            config.Cache = cache.get();
        }

        Operon::RandomGenerator random(config.Seed);
        if (result["shuffle"].as<bool>()) { problem.GetDataset()->Shuffle(random); }
        if (result["standardize"].as<bool>()) { problem.StandardizeData(problem.TrainingRange()); }

        tf::Executor executor(threads);
        Operon::RankIntersectSorter sorter;
        Operon::NSGA2 gp { config, &problem, &treeInitializer, coeffInitializer.get(), generator.get(), reinserter.get(), &sorter };

        auto const* ptr = dynamic_cast<Operon::Evaluator<decltype(dtable)> const*>(errorEvaluator.get());
        Operon::Reporter<Operon::Evaluator<decltype(dtable)>> reporter(ptr);
        gp.Run(executor, random, [&]() -> void { reporter(executor, gp); });
        auto best = reporter.GetBest();
        fmt::print("{}\n", Operon::InfixFormatter::Format(best.Genotype, *problem.GetDataset(), std::numeric_limits<Operon::Scalar>::digits));
        if (result.contains("pareto-front")) {
            Operon::WriteParetoFront(result["pareto-front"].as<std::string>(), gp.Individuals(), dtable, problem, scale);
        }
    } catch (std::exception& e) {
        fmt::print(stderr, "error: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return 0;
}
