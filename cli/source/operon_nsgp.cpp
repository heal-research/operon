// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research


#include "operon/hash/hash.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/solvers/sgd.hpp"
#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdlib>

#include <fmt/core.h>
#include <fmt/ranges.h>

#include <memory>
#include <thread>
#include <taskflow/taskflow.hpp>
#if TF_MINOR_VERSION > 2
#include <taskflow/algorithm/reduce.hpp>
#endif
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/version.hpp"
#include "operon/core/problem.hpp"
#include "operon/formatter/formatter.hpp"
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
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/solvers/sgd.hpp"

#include "util.hpp"
#include "operator_factory.hpp"
#include "reporter.hpp"

auto main(int argc, char** argv) -> int
{
    auto opts = Operon::InitOptions("operon_gp", "Genetic programming symbolic regression");
    auto result = Operon::ParseOptions(std::move(opts), argc, argv);

    // parse and set default values
    Operon::GeneticAlgorithmConfig config{};
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

    // parse remaining config options
    Operon::Range trainingRange;
    Operon::Range testRange;
    std::unique_ptr<Operon::Dataset> dataset;
    std::string targetName;
    bool showPrimitiveSet = false;
    auto threads = std::thread::hardware_concurrency();
    Operon::NodeType primitiveSetConfig = Operon::PrimitiveSet::Arithmetic;

    auto maxLength = result["maxlength"].as<size_t>();
    auto maxDepth = result["maxdepth"].as<size_t>();
    auto crossoverInternalProbability = result["crossover-internal-probability"].as<Operon::Scalar>();

    auto symbolic = result["symbolic"].as<bool>();

    try {
        for (const auto& kv : result.arguments()) {
            const auto& key = kv.key();
            const auto& value = kv.value();

            if (key == "dataset") {
                dataset = std::make_unique<Operon::Dataset>(value, true);
                ENSURE(!dataset->IsView());
            }
            if (key == "seed") {
                config.Seed = kv.as<size_t>();
            }
            if (key == "train") {
                trainingRange = Operon::ParseRange(value);
            }
            if (key == "test") {
                testRange = Operon::ParseRange(value);
            }
            if (key == "target") {
                targetName = value;
            }
            if (key == "maxlength") {
                maxLength = kv.as<size_t>();
            }
            if (key == "maxdepth") {
                maxDepth = kv.as<size_t>();
            }
            if (key == "enable-symbols") {
                auto mask = Operon::ParsePrimitiveSetConfig(value);
                primitiveSetConfig |= mask;
            }
            if (key == "disable-symbols") {
                auto mask = ~Operon::ParsePrimitiveSetConfig(value);
                primitiveSetConfig &= mask;
            }
            if (key == "threads") {
                threads = static_cast<decltype(threads)>(kv.as<size_t>());
            }
            if (key == "show-primitives") {
                showPrimitiveSet = true;
            }
        }

        if (showPrimitiveSet) {
            Operon::PrintPrimitives(primitiveSetConfig);
            return EXIT_SUCCESS;
        }

        // set the target
        Operon::Variable target;
        auto res = dataset->GetVariable(targetName);
        if (!res) {
            fmt::print(stderr, "error: target variable {} does not exist in the dataset.", targetName);
            return EXIT_FAILURE;
        }
        target = *res;
        auto const rows{ dataset->Rows<std::size_t>() };
        if (result.count("train") == 0) {
            trainingRange = Operon::Range{ 0, 2 * rows / 3 }; // by default use 66% of the data as training
        }
        if (result.count("test") == 0) {
            // if no test range is specified, we try to infer a reasonable range based on the trainingRange
            if (trainingRange.Start() > 0) {
                testRange = Operon::Range{ 0, trainingRange.Start() };
            } else if (trainingRange.End() < rows) {
                testRange = Operon::Range{ trainingRange.End(), dataset->Rows<std::size_t>() };
            } else {
                testRange = Operon::Range{ 0, 1};
            }
        }
        // validate training range
        if (trainingRange.Start() >= rows || trainingRange.End() > rows) {
            fmt::print(stderr, "error: the training range {}:{} exceeds the available data range ({} rows)\n", trainingRange.Start(), trainingRange.End(), dataset->Rows());
            return EXIT_FAILURE;
        }

        if (trainingRange.Start() > trainingRange.End()) {
            fmt::print(stderr, "error: invalid training range {}:{}\n", trainingRange.Start(), trainingRange.End());
            return EXIT_FAILURE;
        }

        std::vector<Operon::Hash> inputs;
        if (result.count("inputs") == 0) {
            inputs = dataset->VariableHashes();
            std::erase(inputs, target.Hash);
        } else {
            auto str = result["inputs"].as<std::string>();
            auto tokens = Operon::Split(str, ',');

            for (auto const& tok : tokens) {
                if (auto res = dataset->GetVariable(tok); res.has_value()) {
                    inputs.push_back(res->Hash);
                } else {
                    fmt::print(stderr, "error: variable {} does not exist in the dataset.", tok);
                    return EXIT_FAILURE;
                }
            }
        }
        Operon::Problem problem(std::move(dataset));
        problem.SetTrainingRange(trainingRange);
        problem.SetTestRange(testRange);
        problem.SetTarget(target.Hash);
        problem.SetInputs(inputs);
        problem.ConfigurePrimitiveSet(primitiveSetConfig);

        std::unique_ptr<Operon::CreatorBase> creator;
        creator = ParseCreator(result["creator"].as<std::string>(), problem.GetPrimitiveSet(), problem.GetInputs());

        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        Operon::UniformTreeInitializer treeInitializer(creator.get());

        auto const initialMinDepth = result["creator-mindepth"].as<std::size_t>();
        auto const initialMaxDepth = result["creator-maxdepth"].as<std::size_t>();
        auto const initialMaxLength = result["creator-maxlength"].as<std::size_t>();
        treeInitializer.ParameterizeDistribution(amin+1, initialMaxLength);
        treeInitializer.SetMinDepth(initialMinDepth);
        treeInitializer.SetMaxDepth(initialMaxDepth); // NOLINT

        std::unique_ptr<Operon::CoefficientInitializerBase> coeffInitializer;
        std::unique_ptr<Operon::MutatorBase> onePoint;
        if (symbolic) {
            using Dist = std::uniform_int_distribution<int>;
            coeffInitializer = std::make_unique<Operon::CoefficientInitializer<Dist>>();
            int constexpr range{5};
            dynamic_cast<Operon::CoefficientInitializer<Dist>*>(coeffInitializer.get())->ParameterizeDistribution(-range, +range);
            onePoint = std::make_unique<Operon::OnePointMutation<Dist>>();
            dynamic_cast<Operon::OnePointMutation<Dist>*>(onePoint.get())->ParameterizeDistribution(-range, +range);
        } else {
            using Dist = std::normal_distribution<Operon::Scalar>;
            coeffInitializer = std::make_unique<Operon::CoefficientInitializer<Dist>>();
            dynamic_cast<Operon::NormalCoefficientInitializer*>(coeffInitializer.get())->ParameterizeDistribution(Operon::Scalar{0}, Operon::Scalar{1});
            onePoint = std::make_unique<Operon::OnePointMutation<Dist>>();
            dynamic_cast<Operon::OnePointMutation<Dist>*>(onePoint.get())->ParameterizeDistribution(Operon::Scalar{0}, Operon::Scalar{1});
        }
        auto useTranspositionCrossover = result["use-transposition-aware-crossover"].as<bool>();
        std::unique_ptr<Operon::CrossoverBase> crossover;
        crossover.reset(useTranspositionCrossover
                ? static_cast<Operon::CrossoverBase*>(new Operon::TranspositionAwareCrossover(crossoverInternalProbability, maxDepth, maxLength))
                : static_cast<Operon::CrossoverBase*>(new Operon::SubtreeCrossover(crossoverInternalProbability, maxDepth, maxLength)));

        bool useTranspositionMutation = result["use-transposition-aware-mutation"].as<bool>();
        std::unique_ptr<Operon::MutatorBase> mutator;
        mutator.reset(useTranspositionMutation
                ? static_cast<Operon::MutatorBase*>(new Operon::TranspositionAwareMutation)
                : static_cast<Operon::MutatorBase*>(new Operon::MultiMutation));

        Operon::ChangeVariableMutation changeVar { problem.GetInputs() };
        Operon::ChangeFunctionMutation changeFunc { problem.GetPrimitiveSet() };
        Operon::ReplaceSubtreeMutation replaceSubtree { creator.get(), coeffInitializer.get(), maxDepth, maxLength };
        Operon::InsertSubtreeMutation insertSubtree { creator.get(), coeffInitializer.get(), maxDepth, maxLength };
        Operon::RemoveSubtreeMutation removeSubtree { problem.GetPrimitiveSet() };
        Operon::DiscretePointMutation discretePoint;
        for (auto v : Operon::Math::Constants) {
            discretePoint.Add(static_cast<Operon::Scalar>(v), 1);
        }
        dynamic_cast<Operon::MultiMutation*>(mutator.get())->Add(onePoint.get(), 1.0);
        dynamic_cast<Operon::MultiMutation*>(mutator.get())->Add(&changeVar, 1.0);
        dynamic_cast<Operon::MultiMutation*>(mutator.get())->Add(&changeFunc, 1.0);
        dynamic_cast<Operon::MultiMutation*>(mutator.get())->Add(&replaceSubtree, 1.0);
        dynamic_cast<Operon::MultiMutation*>(mutator.get())->Add(&insertSubtree, 1.0);
        dynamic_cast<Operon::MultiMutation*>(mutator.get())->Add(&removeSubtree, 1.0);
        dynamic_cast<Operon::MultiMutation*>(mutator.get())->Add(&discretePoint, 1.0);

        Operon::DefaultDispatch dtable;
        // DynamicPrimitives::Saxpy<Operon::Scalar, Operon::Backend::BatchSize<Operon::Scalar>> f{};
        // dtable.RegisterCallable(12345UL, f, f);

        auto scale = result["linear-scaling"].as<bool>();
        auto errorEvaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable, scale);
        errorEvaluator->SetBudget(config.Evaluations);

        auto sigma = result["sigma"].as<Operon::Scalar>();

        auto optimizer = std::make_unique<Operon::LevenbergMarquardtOptimizer<decltype(dtable), Operon::OptimizerType::Eigen>>(&dtable, &problem);
        optimizer->SetIterations(config.Iterations);
        Operon::LengthEvaluator lengthEvaluator(&problem, maxLength);
        // Operon::EntropyEvaluator entropyEvaluator(&problem);

        Operon::MultiEvaluator evaluator(&problem);
        evaluator.SetBudget(config.Evaluations);
        evaluator.Add(errorEvaluator.get());
        evaluator.Add(&lengthEvaluator);
        // evaluator.Add(&entropyEvaluator);

        EXPECT(problem.TrainingRange().Size() > 0);

        Operon::CrowdedComparison comp;

        auto femaleSelector = Operon::ParseSelector(result["female-selector"].as<std::string>(), comp);
        auto maleSelector = Operon::ParseSelector(result["male-selector"].as<std::string>(), comp);
        Operon::CoefficientOptimizer cOpt{optimizer.get()};

        auto generator = Operon::ParseGenerator(result["offspring-generator"].as<std::string>(), evaluator, *crossover, *mutator, *femaleSelector, *maleSelector, &cOpt);
        generator->UseTranspositionCache(result["use-transposition-cache"].as<bool>());

        auto reinserter = Operon::ParseReinserter(result["reinserter"].as<std::string>(), comp);

        Operon::RandomGenerator random(config.Seed);
        Operon::Zobrist::Construct(random, maxLength);

        if (result["shuffle"].as<bool>()) {
            problem.GetDataset()->Shuffle(random);
        }
        if (result["standardize"].as<bool>()) {
            problem.StandardizeData(problem.TrainingRange());
        }
        tf::Executor executor(threads);
        Operon::RankIntersectSorter sorter;
        // Operon::RankOrdinalSorter sorter;
        // Operon::MergeSorter sorter;
        // Operon::BestOrderSorter sorter;
        // Operon::EfficientBinarySorter sorter;
        Operon::NSGA2 gp { config, &problem, &treeInitializer, coeffInitializer.get(), generator.get(), reinserter.get(), &sorter };

        //Operon::Reporter<decltype(dtable)> reporter(&dtable, &evaluator, ';', '\n');
        Operon::Reporter<decltype(dtable)> reporter(&dtable, &evaluator);
        reporter.SetModelCriterion(decltype(reporter)::ModelCriterion::MinimumDescriptionLength);
        reporter.SetSigma(sigma);
        gp.Run(executor, random, [&](){ reporter(executor, gp); });
        auto best = reporter.GetBest();
        fmt::print("{}\n", Operon::InfixFormatter::Format(best.Genotype, *problem.GetDataset(), std::numeric_limits<Operon::Scalar>::digits));
    } catch (std::exception& e) {
        fmt::print(stderr, "error: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return 0;
}
