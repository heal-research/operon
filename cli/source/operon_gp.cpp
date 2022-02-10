// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include <chrono>
#include <cmath>
#include <cstdlib>

#include <cxxopts.hpp>
#include <fmt/core.h>

#include <memory>
#include <thread>
#include <taskflow/taskflow.hpp>
#if TF_MINOR_VERSION > 2
#include <taskflow/algorithm/reduce.hpp>
#endif
#include "operon/algorithms/gp.hpp"
#include "operon/core/format.hpp"
#include "operon/core/version.hpp"
#include "operon/core/problem.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"

#include "util.hpp"
#include "operator_factory.hpp"

auto main(int argc, char** argv) -> int
{
    auto opts = Operon::InitOptions("operon_gp", "Genetic programming symbolic regression");
    auto result = Operon::ParseOptions(std::move(opts), argc, argv);

    // parse and set default values
    Operon::GeneticAlgorithmConfig config;
    config.Generations = result["generations"].as<size_t>();
    config.PopulationSize = result["population-size"].as<size_t>();
    config.PoolSize = result["pool-size"].as<size_t>();
    config.Evaluations = result["evaluations"].as<size_t>();
    config.Iterations = result["iterations"].as<size_t>();
    config.CrossoverProbability = result["crossover-probability"].as<Operon::Scalar>();
    config.MutationProbability = result["mutation-probability"].as<Operon::Scalar>();
    config.TimeLimit = result["timelimit"].as<size_t>();
    config.Seed = std::random_device {}();

    // parse remaining config options
    Operon::Range trainingRange;
    Operon::Range testRange;
    std::unique_ptr<Operon::Dataset> dataset;
    std::string target;
    bool showPrimitiveSet = false;
    auto threads = std::thread::hardware_concurrency();
    Operon::NodeType primitiveSetConfig = Operon::PrimitiveSet::Arithmetic;

    auto maxLength = result["maxlength"].as<size_t>();
    auto maxDepth = result["maxdepth"].as<size_t>();
    auto crossoverInternalProbability = result["crossover-internal-probability"].as<Operon::Scalar>();

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
                target = value;
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
        if (auto res = dataset->GetVariable(target); !res.has_value()) {
            fmt::print(stderr, "error: target variable {} does not exist in the dataset.", target);
            return EXIT_FAILURE;
        }
        if (result.count("train") == 0) {
            trainingRange = Operon::Range{ 0, 2 * dataset->Rows() / 3 }; // by default use 66% of the data as training
        }
        if (result.count("test") == 0) {
            // if no test range is specified, we try to infer a reasonable range based on the trainingRange
            if (trainingRange.Start() > 0) {
                testRange = Operon::Range{ 0, trainingRange.Start() };
            } else if (trainingRange.End() < dataset->Rows()) {
                testRange = Operon::Range{ trainingRange.End(), dataset->Rows() };
            } else {
                testRange = Operon::Range{ 0, 1};
            }
        }
        // validate training range
        if (trainingRange.Start() >= dataset->Rows() || trainingRange.End() > dataset->Rows()) {
            fmt::print(stderr, "error: the training range {}:{} exceeds the available data range ({} rows)\n", trainingRange.Start(), trainingRange.End(), dataset->Rows());
            return EXIT_FAILURE;
        }

        if (trainingRange.Start() > trainingRange.End()) {
            fmt::print(stderr, "error: invalid training range {}:{}\n", trainingRange.Start(), trainingRange.End());
            return EXIT_FAILURE;
        }

        std::vector<Operon::Variable> inputs;
        if (result.count("inputs") == 0) {
            auto variables = dataset->Variables();
            std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto const& var) { return var.Name != target; });
        } else {
            auto str = result["inputs"].as<std::string>();
            auto tokens = Operon::Split(str, ',');

            for (auto const& tok : tokens) {
                if (auto res = dataset->GetVariable(tok); res.has_value()) {
                    inputs.push_back(res.value());
                } else {
                    fmt::print(stderr, "error: variable {} does not exist in the dataset.", tok);
                    return EXIT_FAILURE;
                }
            }
        }

        auto problem = Operon::Problem(*dataset).Inputs(inputs).Target(target).TrainingRange(trainingRange).TestRange(testRange);
        problem.GetPrimitiveSet().SetConfig(primitiveSetConfig);

        std::unique_ptr<Operon::CreatorBase> creator;
        creator = ParseCreator(result["tree-creator"].as<std::string>(), problem.GetPrimitiveSet(), problem.InputVariables());

        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        Operon::UniformTreeInitializer treeInitializer(*creator);
        treeInitializer.ParameterizeDistribution(amin+1, maxLength);
        treeInitializer.SetMinDepth(1);
        treeInitializer.SetMaxDepth(1000); // NOLINT

        Operon::NormalCoefficientInitializer coeffInitializer;
        coeffInitializer.ParameterizeDistribution(Operon::Scalar{0.0}, Operon::Scalar{1.0});

        auto crossover = Operon::SubtreeCrossover { crossoverInternalProbability, maxDepth, maxLength };
        auto mutator = Operon::MultiMutation {};
        Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> onePoint;
        onePoint.ParameterizeDistribution(Operon::Scalar{0}, Operon::Scalar{1});

        auto changeVar = Operon::ChangeVariableMutation { problem.InputVariables() };
        auto changeFunc = Operon::ChangeFunctionMutation { problem.GetPrimitiveSet() };
        auto replaceSubtree = Operon::ReplaceSubtreeMutation { *creator, maxDepth, maxLength };
        auto insertSubtree = Operon::InsertSubtreeMutation { *creator, maxDepth, maxLength, problem.GetPrimitiveSet() };
        auto removeSubtree = Operon::RemoveSubtreeMutation { problem.GetPrimitiveSet() };
        mutator.Add(onePoint, 1.0);
        mutator.Add(changeVar, 1.0);
        mutator.Add(changeFunc, 1.0);
        mutator.Add(replaceSubtree, 1.0);
        mutator.Add(insertSubtree, 1.0);
        mutator.Add(removeSubtree, 1.0);

        auto const& [error, scale] = Operon::ParseErrorMetric(result["error-metric"].as<std::string>());

        Operon::Interpreter interpreter;
        Operon::Evaluator evaluator(problem, interpreter, *error, scale);

        evaluator.SetLocalOptimizationIterations(config.Iterations);
        evaluator.SetBudget(config.Evaluations);

        EXPECT(problem.TrainingRange().Size() > 0);

        auto comp = [](auto const& lhs, auto const& rhs) { return lhs[0] < rhs[0]; };

        auto femaleSelector = Operon::ParseSelector(result["female-selector"].as<std::string>(), comp);
        auto maleSelector = Operon::ParseSelector(result["male-selector"].as<std::string>(), comp);

        auto generator = Operon::ParseGenerator(result["offspring-generator"].as<std::string>(), evaluator, crossover, mutator, *femaleSelector, *maleSelector);
        auto reinserter = Operon::ParseReinserter(result["reinserter"].as<std::string>(), comp);

        Operon::RandomGenerator random(config.Seed);
        if (result["shuffle"].as<bool>()) {
            problem.GetDataset().Shuffle(random);
        }
        if (result["standardize"].as<bool>()) {
            problem.StandardizeData(problem.TrainingRange());
        }

        tf::Executor executor(threads);

        auto t0 = std::chrono::high_resolution_clock::now();

        Operon::GeneticProgrammingAlgorithm gp { problem, config, treeInitializer, coeffInitializer, *generator, *reinserter };

        auto targetValues = problem.TargetValues();
        auto targetTrain = targetValues.subspan(trainingRange.Start(), trainingRange.Size());
        auto targetTest = targetValues.subspan(testRange.Start(), testRange.Size());

        // some boilerplate for reporting results
        const size_t idx { 0 };
        auto getBest = [&](Operon::Span<Operon::Individual const> pop) -> Operon::Individual {
            const auto *minElem = std::min_element(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) { return lhs[idx] < rhs[idx]; });
            return *minElem;
        };

        Operon::Individual best(1);
        //auto const& pop = gp.Parents();

        auto getSize = [](Operon::Individual const& ind) { return sizeof(ind) + sizeof(ind.Genotype) + sizeof(Operon::Node) * ind.Genotype.Nodes().capacity(); };

        tf::Executor exe(threads);

        auto report = [&]() {
            auto const& pop = gp.Parents();
            auto const& off = gp.Offspring();

            best = getBest(pop);

            Operon::Vector<Operon::Scalar> estimatedTrain;
            Operon::Vector<Operon::Scalar> estimatedTest;

            tf::Taskflow taskflow;

            auto evalTrain = taskflow.emplace([&]() {
                estimatedTrain = interpreter.Evaluate<Operon::Scalar>(best.Genotype, problem.GetDataset(), trainingRange);
            });

            auto evalTest = taskflow.emplace([&]() {
                estimatedTest = interpreter.Evaluate<Operon::Scalar>(best.Genotype, problem.GetDataset(), testRange);
            });

            // scale values
            Operon::Scalar a{};
            Operon::Scalar b{};
            auto linearScaling = taskflow.emplace([&]() {
                auto [a_, b_] = Operon::FitLeastSquares(estimatedTrain, targetTrain);
                a = static_cast<Operon::Scalar>(a_);
                b = static_cast<Operon::Scalar>(b_);
            });

            double r2Train{};
            double r2Test{};
            double nmseTrain{};
            double nmseTest{};
            double maeTrain{};
            double maeTest{};

            auto scaleTrain = taskflow.emplace([&]() {
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> estimated(estimatedTrain.data(), estimatedTrain.size());
                estimated = estimated * a + b;
            });

            auto scaleTest = taskflow.emplace([&]() {
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> estimated(estimatedTest.data(), estimatedTest.size());
                estimated = estimated * a + b;
            });

            auto calcStats = taskflow.emplace([&]() {
                // negate the R2 because this is an internal fitness measure (minimization) which we here repurpose
                r2Train = -Operon::R2{}(estimatedTrain, targetTrain);
                r2Test = -Operon::R2{}(estimatedTest, targetTest);

                nmseTrain = Operon::NMSE{}(estimatedTrain, targetTrain);
                nmseTest = Operon::NMSE{}(estimatedTest, targetTest);

                maeTrain = Operon::MAE{}(estimatedTrain, targetTrain);
                maeTest = Operon::MAE{}(estimatedTest, targetTest);
            });

            double avgLength = 0;
            double avgQuality = 0;
            double totalMemory = 0;

            auto calculateLength = taskflow.transform_reduce(pop.begin(), pop.end(), avgLength, std::plus<double>{}, [](auto const& ind) { return ind.Genotype.Length(); });
            auto calculateQuality = taskflow.transform_reduce(pop.begin(), pop.end(), avgQuality, std::plus<double>{}, [idx=idx](auto const& ind) { return ind[idx]; });
            auto calculatePopMemory = taskflow.transform_reduce(pop.begin(), pop.end(), totalMemory, std::plus{}, [&](auto const& ind) { return getSize(ind); });
            auto calculateOffMemory = taskflow.transform_reduce(off.begin(), off.end(), totalMemory, std::plus{}, [&](auto const& ind) { return getSize(ind); });

            // define task graph
            linearScaling.succeed(evalTrain, evalTest);
            linearScaling.precede(scaleTrain, scaleTest);
            calcStats.succeed(scaleTrain, scaleTest);
            calcStats.precede(calculateLength, calculateQuality, calculatePopMemory, calculateOffMemory);

            exe.run(taskflow).wait();

            avgLength /= static_cast<double>(pop.size());
            avgQuality /= static_cast<double>(pop.size());

            auto t1 = std::chrono::high_resolution_clock::now();
            auto elapsed = static_cast<double>(std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count()) / 1e6;

            fmt::print("{:.3f}\t{}\t", elapsed, gp.Generation());
            fmt::print("{:.3f}\t{:.3f}\t{:.3g}\t{:.3g}\t{:.3g}\t{:.3g}\t", r2Train, r2Test, maeTrain, maeTest, nmseTrain, nmseTest);
            fmt::print("{:.3g}\t{:.1f}\t{:.3f}\t{:.3f}\t{}\t{}\t{}\t", avgQuality, avgLength, /* shape */ 0.0, /* diversity */ 0.0, evaluator.FitnessEvaluations(), evaluator.LocalEvaluations(), evaluator.TotalEvaluations());
            fmt::print("{}\t{}\n", totalMemory, config.Seed);
        };

        gp.Run(executor, random, report);
        fmt::print("{}\n", Operon::InfixFormatter::Format(best.Genotype, problem.GetDataset(), 20));
    } catch (std::exception& e) {
        fmt::print(stderr, "error: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return 0;
}

