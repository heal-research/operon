// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include <chrono>
#include <cstdlib>

#include <cxxopts.hpp>
#include <fmt/core.h>

#include <thread>

#include "algorithms/gp.hpp"
#include "algorithms/nsga2.hpp"

#include "core/format.hpp"
#include "core/metrics.hpp"
#include "core/version.hpp"
#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/evaluator.hpp"
#include "operators/generator.hpp"
#include "operators/initializer.hpp"
#include "operators/mutation.hpp"
#include "operators/reinserter/keepbest.hpp"
#include "operators/reinserter/replaceworst.hpp"
#include "operators/selection.hpp"

#include "util.hpp"

using namespace Operon;

int main(int argc, char** argv)
{
    cxxopts::Options opts("operon_cli", "C++ large-scale genetic programming");

    opts.add_options()
        ("dataset", "Dataset file name (csv) (required)", cxxopts::value<std::string>())
        ("shuffle", "Shuffle the input data", cxxopts::value<bool>()->default_value("false"))
        ("standardize", "Standardize the training partition (zero mean, unit variance)", cxxopts::value<bool>()->default_value("false"))
        ("train", "Training range specified as start:end (required)", cxxopts::value<std::string>())
        ("test", "Test range specified as start:end", cxxopts::value<std::string>())
        ("target", "Name of the target variable (required)", cxxopts::value<std::string>())
        ("inputs", "Comma-separated list of input variables", cxxopts::value<std::string>())
        ("error-metric", "The error metric used for calculating fitness", cxxopts::value<std::string>()->default_value("r2"))
        ("population-size", "Population size", cxxopts::value<size_t>()->default_value("1000"))
        ("pool-size", "Recombination pool size (how many generated offspring per generation)", cxxopts::value<size_t>()->default_value("1000"))
        ("seed", "Random number seed", cxxopts::value<Operon::RandomGenerator::result_type>()->default_value("0"))
        ("generations", "Number of generations", cxxopts::value<size_t>()->default_value("1000"))
        ("evaluations", "Evaluation budget", cxxopts::value<size_t>()->default_value("1000000"))
        ("iterations", "Local optimization iterations", cxxopts::value<size_t>()->default_value("0"))
        ("selection-pressure", "Selection pressure", cxxopts::value<size_t>()->default_value("100"))
        ("maxlength", "Maximum length", cxxopts::value<size_t>()->default_value("50"))
        ("maxdepth", "Maximum depth", cxxopts::value<size_t>()->default_value("10"))
        ("crossover-probability", "The probability to apply crossover", cxxopts::value<Operon::Scalar>()->default_value("1.0"))
        ("crossover-internal-probability", "Crossover bias towards swapping function nodes", cxxopts::value<Operon::Scalar>()->default_value("0.9"))
        ("mutation-probability", "The probability to apply mutation", cxxopts::value<Operon::Scalar>()->default_value("0.25"))
        ("tree-creator", "Tree creator operator to initialize the population with.", cxxopts::value<std::string>())
        ("female-selector", "Female selection operator, with optional parameters separated by : (eg, --selector tournament:5)", cxxopts::value<std::string>())
        ("male-selector", "Male selection operator, with optional parameters separated by : (eg, --selector tournament:5)", cxxopts::value<std::string>())
        ("offspring-generator", "OffspringGenerator operator, with optional parameters separated by : (eg --offspring-generator brood:10:10)", cxxopts::value<std::string>())
        ("reinserter", "Reinsertion operator merging offspring in the recombination pool back into the population", cxxopts::value<std::string>())
        ("enable-symbols", "Comma-separated list of enabled symbols (add, sub, mul, div, exp, log, sin, cos, tan, sqrt, cbrt)", cxxopts::value<std::string>())
        ("disable-symbols", "Comma-separated list of disabled symbols (add, sub, mul, div, exp, log, sin, cos, tan, sqrt, cbrt)", cxxopts::value<std::string>())
        ("show-primitives", "Display the primitive set used by the algorithm")
        ("threads", "Number of threads to use for parallelism", cxxopts::value<size_t>()->default_value("0"))
        ("timelimit", "Time limit after which the algorithm will terminate", cxxopts::value<size_t>()->default_value(fmt::format("{}", std::numeric_limits<size_t>::max())))
        ("debug", "Debug mode (more information displayed)")
        ("help", "Print help")
        ("version", "Print version and program information");

    auto result = opts.parse(argc, argv);
    if (result.arguments().empty() || result.count("help") > 0) {
        fmt::print("{}\n", opts.help());
        exit(EXIT_SUCCESS);
    }

    if (result.count("version") > 0) {
        fmt::print("{}\n", Version());
        exit(EXIT_SUCCESS);
    }

    // parse and set default values
    GeneticAlgorithmConfig config;
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
    Range trainingRange;
    Range testRange;
    std::unique_ptr<Dataset> dataset;
    std::string target;
    bool showPrimitiveSet = false;
    auto threads = std::thread::hardware_concurrency();
    NodeType primitiveSetConfig = PrimitiveSet::Arithmetic;

    auto maxLength = result["maxlength"].as<size_t>();
    auto maxDepth = result["maxdepth"].as<size_t>();
    auto crossoverInternalProbability = result["crossover-internal-probability"].as<Operon::Scalar>();

    try {
        for (auto kv : result.arguments()) {
            auto& key = kv.key();
            auto& value = kv.value();

            if (key == "dataset") {
                dataset.reset(new Dataset(value, true));
                ENSURE(!dataset->IsView());
            }
            if (key == "seed") {
                config.Seed = kv.as<size_t>();
            }
            if (key == "train") {
                trainingRange = ParseRange(value);
            }
            if (key == "test") {
                testRange = ParseRange(value);
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
                auto mask = ParsePrimitiveSetConfig(value);
                primitiveSetConfig |= mask;
            }
            if (key == "disable-symbols") {
                auto mask = ~ParsePrimitiveSetConfig(value);
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
            PrimitiveSet tmpSet;
            tmpSet.SetConfig(primitiveSetConfig);
            fmt::print("Built-in primitives:\n");
            fmt::print("{:<8}\t{:<50}\t{:>7}\t\t{:>9}\n", "Symbol", "Description", "Enabled", "Frequency");
            for (size_t i = 0; i < NodeTypes::Count; ++i) {
                auto type = static_cast<NodeType>(1u << i);
                auto hash = Node(type).HashValue;
                auto enabled = tmpSet.IsEnabled(hash);
                auto freq = enabled ? tmpSet.GetFrequency(hash) : 0;
                Node node(type);
                fmt::print("{:<8}\t{:<50}\t{:>7}\t\t{:>9}\n", node.Name(), node.Desc(), enabled, freq ? std::to_string(freq) : "-");
            }
            return 0;
        }

        if (!dataset) {
            fmt::print(stderr, "{}\n{}\n", "Error: no dataset given.", opts.help());
            exit(EXIT_FAILURE);
        }
        if (result.count("target") == 0) {
            fmt::print(stderr, "{}\n{}\n", "Error: no target variable given.", opts.help());
            exit(EXIT_FAILURE);
        }
        if (result.count("train") == 0) {
            trainingRange = { 0, 2 * dataset->Rows() / 3 }; // by default use 66% of the data as training
        }
        if (result.count("test") == 0) {
            // if no test range is specified, we try to infer a reasonable range based on the trainingRange
            if (trainingRange.Start() > 0) {
                testRange = { 0, trainingRange.Start() };
            } else if (trainingRange.End() < dataset->Rows()) {
                testRange = { trainingRange.End(), dataset->Rows() };
            } else {
                testRange = { 0, 1 };
            }
        }
        // validate training range
        if (trainingRange.Start() >= dataset->Rows() || trainingRange.End() > dataset->Rows()) {
            fmt::print(stderr, "The training range {}:{} exceeds the available data range ({} rows)\n", trainingRange.Start(), trainingRange.End(), dataset->Rows());
            exit(EXIT_FAILURE);
        }

        if (trainingRange.Start() > trainingRange.End()) {
            fmt::print(stderr, "Invalid training range {}:{}\n", trainingRange.Start(), trainingRange.End());
            exit(EXIT_FAILURE);
        }

        std::vector<Variable> inputs;
        if (result.count("inputs") == 0) {
            auto variables = dataset->Variables();
            std::copy_if(variables.begin(), variables.end(), std::back_inserter(inputs), [&](auto const& var) { return var.Name != target; });
        } else {
            auto str = result["inputs"].as<std::string>();
            auto tokens = Split(str, ',');

            for (auto const& tok : tokens) {
                if (auto res = dataset->GetVariable(tok); res.has_value()) {
                    inputs.push_back(res.value());
                } else {
                    fmt::print(stderr, "Variable {} does not exist in the dataset.", tok);
                    exit(EXIT_FAILURE);
                }
            }
        }

        auto problem = Problem(*dataset).Inputs(inputs).Target(target).TrainingRange(trainingRange).TestRange(testRange);
        problem.GetPrimitiveSet().SetConfig(primitiveSetConfig);

        using Reinserter = ReinserterBase;
        using OffspringGenerator = OffspringGeneratorBase;

        std::unique_ptr<CreatorBase> creator;

        if (result.count("tree-creator") == 0) {
            creator.reset(new BalancedTreeCreator(problem.GetPrimitiveSet(), problem.InputVariables(), 0.0));
        } else {
            auto value = result["tree-creator"].as<std::string>();

            if (value == "ptc2") {
                creator.reset(new ProbabilisticTreeCreator(problem.GetPrimitiveSet(), problem.InputVariables()));
            } else if (value == "grow") {
                creator.reset(new GrowTreeCreator(problem.GetPrimitiveSet(), problem.InputVariables()));
            } else {
                auto tokens = Split(value, ':');
                double irregularityBias = 0.0;
                if (tokens.size() > 1) {
                    if (auto [val, ok] = ParseDouble(tokens[1]); ok) {
                        irregularityBias = val;
                    } else {
                        fmt::print(stderr, "{}\n{}\n", "Error: could not parse BTC bias argument.", opts.help());
                        exit(EXIT_FAILURE);
                    }
                }
                creator.reset(new BalancedTreeCreator(problem.GetPrimitiveSet(), problem.InputVariables(), irregularityBias));
            }
        }

        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        std::uniform_int_distribution<size_t> sizeDistribution(amin + 1, maxLength);
        UniformInitializer initializer(*creator, sizeDistribution);
        initializer.MinDepth(1);
        initializer.MaxDepth(1000);
        auto crossover = SubtreeCrossover { crossoverInternalProbability, maxDepth, maxLength };
        auto mutator = MultiMutation {};
        auto onePoint = OnePointMutation {};
        auto changeVar = ChangeVariableMutation { problem.InputVariables() };
        auto changeFunc = ChangeFunctionMutation { problem.GetPrimitiveSet() };
        auto replaceSubtree = ReplaceSubtreeMutation { *creator, maxDepth, maxLength };
        auto insertSubtree = InsertSubtreeMutation { *creator, maxDepth, maxLength, problem.GetPrimitiveSet() };
        auto removeSubtree = RemoveSubtreeMutation { problem.GetPrimitiveSet() };
        mutator.Add(onePoint, 1.0);
        mutator.Add(changeVar, 1.0);
        mutator.Add(changeFunc, 1.0);
        mutator.Add(replaceSubtree, 1.0);
        mutator.Add(insertSubtree, 1.0);
        mutator.Add(removeSubtree, 1.0);

        // initialize an interpreter using a default dispatch table
        // (the dispatch table is by default initialized with the common operations, but they can be overriden)
        DispatchTable ft;
        Interpreter interpreter(ft);

        std::unique_ptr<EvaluatorBase> evaluator;
        auto errorMetric = result["error-metric"].as<std::string>();
        if (errorMetric == "r2") {
            evaluator.reset(new RSquaredEvaluator(problem, interpreter));
        } else if (errorMetric == "nmse") {
            evaluator.reset(new NormalizedMeanSquaredErrorEvaluator(problem, interpreter));
        } else if (errorMetric == "mse") {
            evaluator.reset(new MeanSquaredErrorEvaluator(problem, interpreter));
        } else if (errorMetric == "rmse") {
            evaluator.reset(new RootMeanSquaredErrorEvaluator(problem, interpreter));
        } else if (errorMetric == "mae") {
            evaluator.reset(new MeanAbsoluteErrorEvaluator(problem, interpreter));
        } else if (errorMetric == "l2") {
            evaluator.reset(new L2NormEvaluator(problem, interpreter));
        } else {
            throw std::runtime_error(fmt::format("Unknown metric {}\n", errorMetric));
        }

        evaluator->SetLocalOptimizationIterations(config.Iterations);
        evaluator->SetBudget(config.Evaluations);

        EXPECT(problem.TrainingRange().Size() > 0);

        auto comp = [](auto const& lhs, auto const& rhs) { return lhs[0] < rhs[0]; };

        auto parseSelector = [&](const std::string& name) -> SelectorBase* {
            if (result.count(name) == 0) {
                auto sel = new TournamentSelector(comp);
                sel->SetTournamentSize(5);
                return sel;
            } else {
                auto value = result[name].as<std::string>();
                auto tokens = Split(value, ':');
                if (tokens[0] == "tournament") {
                    size_t tSize = 5;
                    if (tokens.size() > 1) {
                        if (auto [p, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), tSize); ec != std::errc()) {
                            fmt::print(stderr, "{}\n{}\n", "Error: could not parse tournament size argument.", opts.help());
                            exit(EXIT_FAILURE);
                        }
                    }
                    auto sel = new TournamentSelector(comp);
                    sel->SetTournamentSize(tSize);
                    return sel;
                } else if (tokens[0] == "proportional") {
                    auto sel = new ProportionalSelector(comp);
                    sel->SetObjIndex(0);
                    return sel;
                } else if (tokens[0] == "rank") {
                    size_t tSize = 5;
                    if (tokens.size() > 1) {
                        if (auto [p, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), tSize); ec != std::errc()) {
                            fmt::print(stderr, "{}\n{}\n", "Error: could not parse tournament size argument.", opts.help());
                            exit(EXIT_FAILURE);
                        }
                    }
                    auto sel = new RankTournamentSelector(comp);
                    sel->SetTournamentSize(tSize);
                    return sel;
                } else if (tokens[0] == "random") {
                    return new RandomSelector();
                }
            }
            auto sel = new TournamentSelector(comp);
            sel->SetTournamentSize(5);
            return sel;
        };

        std::unique_ptr<SelectorBase> femaleSelector;
        std::unique_ptr<SelectorBase> maleSelector;

        femaleSelector.reset(parseSelector("female-selector"));
        maleSelector.reset(parseSelector("male-selector"));

        std::unique_ptr<OffspringGenerator> generator;
        if (result.count("offspring-generator") == 0) {
            generator.reset(new BasicOffspringGenerator(*evaluator, crossover, mutator, *femaleSelector, *maleSelector));
        } else {
            auto value = result["offspring-generator"].as<std::string>();
            auto tokens = Split(value, ':');
            if (tokens[0] == "basic") {
                generator.reset(new BasicOffspringGenerator(*evaluator, crossover, mutator, *femaleSelector, *maleSelector));
            } else if (tokens[0] == "brood") {
                size_t broodSize = 10;
                if (tokens.size() > 1) {
                    if (auto [p, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), broodSize); ec != std::errc()) {
                        fmt::print(stderr, "{}\n{}\n", "Error: could not parse brood size argument.", opts.help());
                        exit(EXIT_FAILURE);
                    }
                }
                auto ptr = new BroodOffspringGenerator(*evaluator, crossover, mutator, *femaleSelector, *maleSelector);
                ptr->BroodSize(broodSize);
                generator.reset(ptr);
            } else if (tokens[0] == "poly") {
                size_t broodSize = 10;
                if (tokens.size() > 1) {
                    if (auto [p, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), broodSize); ec != std::errc()) {
                        fmt::print(stderr, "{}\n{}\n", "Error: could not parse brood size argument.", opts.help());
                        exit(EXIT_FAILURE);
                    }
                }
                auto ptr = new PolygenicOffspringGenerator(*evaluator, crossover, mutator, *femaleSelector, *maleSelector);
                ptr->PolygenicSize(broodSize);
                generator.reset(ptr);
            } else if (tokens[0] == "os") {
                size_t selectionPressure = 100;
                double comparisonFactor = 1.0;
                if (tokens.size() > 1) {
                    if (auto [p, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), selectionPressure); ec != std::errc()) {
                        fmt::print(stderr, "{}\n{}\n", "Error: could not parse maximum selection pressure argument.", opts.help());
                        exit(EXIT_FAILURE);
                    }
                }
                if (tokens.size() > 2) {
                    if (auto [val, ok] = ParseDouble(tokens[2]); ok) {
                        comparisonFactor = val;
                    } else {
                        fmt::print(stderr, "{}\n{}\n", "Error: could not parse comparison factor argument.", opts.help());
                        exit(EXIT_FAILURE);
                    }
                }
                auto ptr = new OffspringSelectionGenerator(*evaluator, crossover, mutator, *femaleSelector, *maleSelector);
                ptr->MaxSelectionPressure(selectionPressure);
                ptr->ComparisonFactor(comparisonFactor);
                generator.reset(ptr);
            }
        }
        std::unique_ptr<Reinserter> reinserter;
        if (result.count("reinserter") == 0) {
            reinserter.reset(new ReplaceWorstReinserter(comp));
        } else {
            auto value = result["reinserter"].as<std::string>();
            if (value == "keep-best") {
                reinserter.reset(new KeepBestReinserter(comp));
            } else if (value == "replace-worst") {
                reinserter.reset(new ReplaceWorstReinserter(comp));
            }
        }

        Operon::RandomGenerator random(config.Seed);
        if (result["shuffle"].as<bool>()) {
            problem.GetDataset().Shuffle(random);
        }
        if (result["standardize"].as<bool>()) {
            problem.StandardizeData(problem.TrainingRange());
        }

        tf::Executor executor(threads);

        auto t0 = std::chrono::high_resolution_clock::now();

        GeneticProgrammingAlgorithm gp { problem, config, initializer, *generator, *reinserter };

        auto targetValues = problem.TargetValues();
        auto targetTrain = targetValues.subspan(trainingRange.Start(), trainingRange.Size());
        auto targetTest = targetValues.subspan(testRange.Start(), testRange.Size());

        // some boilerplate for reporting results
        const size_t idx { 0 };
        auto getBest = [&](const Operon::Span<const Individual> pop) -> Individual {
            auto minElem = std::min_element(pop.begin(), pop.end(), [&](auto const& lhs, auto const& rhs) { return lhs[idx] < rhs[idx]; });
            return *minElem;
        };

        Individual best(1);
        //auto const& pop = gp.Parents();

        auto getSize = [](const Individual& ind) { return sizeof(ind) + sizeof(ind.Genotype) + sizeof(Node) * ind.Genotype.Nodes().capacity(); };

        tf::Executor exe(threads);

        auto report = [&]() {
            auto const& pop = gp.Parents();
            auto const& off = gp.Offspring();

            best = getBest(pop);

            Operon::Vector<Operon::Scalar> estimatedTrain, estimatedTest;

            tf::Taskflow taskflow;

            auto evalTrain = taskflow.emplace([&]() {
                estimatedTrain = interpreter.Evaluate<Operon::Scalar>(best.Genotype, problem.GetDataset(), trainingRange);
            });

            auto evalTest = taskflow.emplace([&]() {
                estimatedTest = interpreter.Evaluate<Operon::Scalar>(best.Genotype, problem.GetDataset(), testRange);
            });

            // scale values
            Operon::Scalar a, b;
            auto linearScaling = taskflow.emplace([&]() {
                auto stats = bivariate::accumulate<double>(estimatedTrain.data(), targetTrain.data(), estimatedTrain.size());
                a = static_cast<Operon::Scalar>(stats.covariance / stats.variance_x);
                if (!std::isfinite(a)) { a = 1; }
                b = static_cast<Operon::Scalar>(stats.mean_y - a * stats.mean_x);
            });

            double r2Train, r2Test, nmseTrain, nmseTest, maeTrain, maeTest;

            auto scaleTrain = taskflow.emplace([&]() {
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> estimated(estimatedTrain.data(), estimatedTrain.size());
                estimated = estimated * a + b;
            });

            auto scaleTest = taskflow.emplace([&]() {
                Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1>> estimated(estimatedTest.data(), estimatedTest.size());
                estimated = estimated * a + b;
            });

            auto calcStats = taskflow.emplace([&]() {
                r2Train = RSquared<Operon::Scalar>(estimatedTrain, targetTrain);
                r2Test = RSquared<Operon::Scalar>(estimatedTest, targetTest);

                nmseTrain = NormalizedMeanSquaredError<Operon::Scalar>(estimatedTrain, targetTrain);
                nmseTest = NormalizedMeanSquaredError<Operon::Scalar>(estimatedTest, targetTest);

                maeTrain = MeanAbsoluteError<Operon::Scalar>(estimatedTrain, targetTrain);
                maeTest = MeanAbsoluteError<Operon::Scalar>(estimatedTest, targetTest);
            });

            double avgLength = 0;
            double avgQuality = 0;
            double totalMemory = 0;

            EXPECT(std::all_of(pop.begin(), pop.end(), [](auto const& ind) { return ind.Genotype.Length() > 0; }));

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
            auto elapsed = (double)std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count() / 1e6;

            fmt::print("{:.4f}\t{}\t", elapsed, gp.Generation());
            fmt::print("{:.4f}\t{:.4f}\t{:.4g}\t{:.4g}\t{:.4g}\t{:.4g}\t", r2Train, r2Test, maeTrain, maeTest, nmseTrain, nmseTest);
            fmt::print("{:.4f}\t{:.1f}\t{:.3f}\t{:.3f}\t{}\t{}\t{}\t", avgQuality, avgLength, /* shape */ 0.0, /* diversity */ 0.0, evaluator->FitnessEvaluations(), evaluator->LocalEvaluations(), evaluator->TotalEvaluations());
            fmt::print("{}\t{}\n", totalMemory, config.Seed);
        };

        gp.Run(executor, random, report);
        fmt::print("{}\n", InfixFormatter::Format(best.Genotype, problem.GetDataset(), 20));
    } catch (std::exception& e) {
        fmt::print("{}\n", e.what());
        std::exit(EXIT_FAILURE);
    }

    return 0;
}

