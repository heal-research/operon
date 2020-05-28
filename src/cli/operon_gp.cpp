/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#include <cstdlib>

#include <cxxopts.hpp>
#include <fmt/core.h>

#include <tbb/task_scheduler_init.h>
#include <tbb/global_control.h>

#include "algorithms/gp.hpp"

#include "core/common.hpp"
#include "core/format.hpp"
#include "core/metrics.hpp"
#include "operators/initializer.hpp"
#include "operators/creator.hpp"
#include "operators/crossover.hpp"
#include "operators/evaluator.hpp"
#include "operators/mutation.hpp"
#include "operators/generator.hpp"
#include "operators/selection.hpp"
#include "operators/reinserter/keepbest.hpp"
#include "operators/reinserter/replaceworst.hpp"
#include "stat/linearscaler.hpp"

#include "util.hpp"

using namespace Operon;

int main(int argc, char* argv[])
{
    cxxopts::Options opts("operon_cli", "C++ large-scale genetic programming");

    opts.add_options()
        ("dataset", "Dataset file name (csv) (required)", cxxopts::value<std::string>())
        ("shuffle", "Shuffle the input data", cxxopts::value<bool>()->default_value("false"))
        ("standardize", "Standardize the training partition (zero mean, unit variance)", cxxopts::value<bool>()->default_value("false"))
        ("train", "Training range specified as start:end (required)", cxxopts::value<std::string>())
        ("test", "Test range specified as start:end", cxxopts::value<std::string>())
        ("target", "Name of the target variable (required)", cxxopts::value<std::string>())
        ("population-size", "Population size", cxxopts::value<size_t>()->default_value("1000"))
        ("pool-size", "Recombination pool size (how many generated offspring per generation)", cxxopts::value<size_t>()->default_value("1000"))
        ("seed", "Random number seed", cxxopts::value<Operon::Random::result_type>()->default_value("0"))
        ("generations", "Number of generations", cxxopts::value<size_t>()->default_value("1000"))
        ("evaluations", "Evaluation budget", cxxopts::value<size_t>()->default_value("1000000"))
        ("iterations", "Local optimization iterations", cxxopts::value<size_t>()->default_value("50"))
        ("selection-pressure", "Selection pressure", cxxopts::value<size_t>()->default_value("100"))
        ("maxlength", "Maximum length", cxxopts::value<size_t>()->default_value("50"))
        ("maxdepth", "Maximum depth", cxxopts::value<size_t>()->default_value("10"))
        ("crossover-probability", "The probability to apply crossover", cxxopts::value<Operon::Scalar>()->default_value("1.0"))
        ("mutation-probability", "The probability to apply mutation", cxxopts::value<Operon::Scalar>()->default_value("0.25"))
        ("tree-creator", "Tree creator operator to initialize the population with.", cxxopts::value<std::string>())
        ("female-selector", "Female selection operator, with optional parameters separated by : (eg, --selector tournament:5)", cxxopts::value<std::string>())
        ("male-selector", "Male selection operator, with optional parameters separated by : (eg, --selector tournament:5)", cxxopts::value<std::string>())
        ("offspring-generator", "OffspringGenerator operator, with optional parameters separated by : (eg --offspring-generator brood:10:10)", cxxopts::value<std::string>())
        ("reinserter", "Reinsertion operator merging offspring in the recombination pool back into the population", cxxopts::value<std::string>())
        ("enable-symbols", "Comma-separated list of enabled symbols (add, sub, mul, div, exp, log, sin, cos, tan, sqrt, cbrt)", cxxopts::value<std::string>())
        ("disable-symbols", "Comma-separated list of disabled symbols (add, sub, mul, div, exp, log, sin, cos, tan, sqrt, cbrt)", cxxopts::value<std::string>())
        ("show-grammar", "Show grammar (primitive set) used by the algorithm")
        ("threads", "Number of threads to use for parallelism", cxxopts::value<size_t>()->default_value("0"))
        ("debug", "Debug mode (more information displayed)")("help", "Print help");

    auto result = opts.parse(argc, argv);
    if (result.arguments().empty() || result.count("help") > 0) {
        fmt::print("{}\n", opts.help());
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
    config.Seed = std::random_device{}();

    // parse remaining config options
    Range trainingRange;
    Range testRange;
    std::unique_ptr<Dataset> dataset;
    std::string fileName; // data file name
    std::string target;
    bool showGrammar = false;
    auto threads = tbb::task_scheduler_init::default_num_threads();
    GrammarConfig grammarConfig = Grammar::Arithmetic;

    auto maxLength = result["maxlength"].as<size_t>();
    auto maxDepth = result["maxdepth"].as<size_t>();

    try {
        for (auto kv : result.arguments()) {
            auto& key = kv.key();
            auto& value = kv.value();
            if (key == "dataset") {
                fileName = value;
                dataset.reset(new Dataset(fileName, true));
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
                auto mask = ParseGrammarConfig(value);
                grammarConfig |= mask;
            }
            if (key == "disable-symbols") {
                auto mask = ~ParseGrammarConfig(value);
                grammarConfig &= mask;
            }
            if (key == "threads") {
                threads = kv.as<size_t>();
            }
            if (key == "show-grammar") {
                showGrammar = true;
            }
        }

        if (showGrammar) {
            Grammar tmpGrammar;
            tmpGrammar.SetConfig(grammarConfig);
            for (auto i = 0u; i < NodeTypes::Count; ++i) {
                auto type = static_cast<NodeType>(1u << i);
                auto n = Node(type);
                if (tmpGrammar.IsEnabled(type)) {
                    fmt::print("{}\t{}\n", n.Name(), tmpGrammar.GetFrequency(type));
                }
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
                testRange = { 0, 0 };
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

        auto variables = dataset->Variables();
        auto problem = Problem(*dataset, variables, target, trainingRange, testRange);
        problem.GetGrammar().SetConfig(grammarConfig);

        const gsl::index idx { 0 };
        using Ind                = Individual<1>;
        using Evaluator          = RSquaredEvaluator<Ind>;
        using Selector           = SelectorBase<Ind, idx>;
        using Reinserter         = ReinserterBase<Ind, idx>;
        using OffspringGenerator = OffspringGeneratorBase<Evaluator, SubtreeCrossover, MultiMutation, Selector, Selector>;

        std::unique_ptr<CreatorBase> creator;

        if (result.count("tree-creator") == 0) {
            creator.reset(new BalancedTreeCreator(problem.GetGrammar(), problem.InputVariables(), 0.0));
        } else {
            auto value = result["tree-creator"].as<std::string>();

            if (value == "ptc2") {
                creator.reset(new ProbabilisticTreeCreator(problem.GetGrammar(), problem.InputVariables()));
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
                creator.reset(new BalancedTreeCreator(problem.GetGrammar(), problem.InputVariables(), irregularityBias));
            }
        }

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        //auto creator             = BalancedTreeCreator { problem.GetGrammar(), problem.InputVariables() };
        auto initializer         = Initializer { *creator, sizeDistribution };
        auto crossover           = SubtreeCrossover { 0.9, maxDepth, maxLength };
        auto mutator             = MultiMutation {};
        auto onePoint            = OnePointMutation {};
        auto changeVar           = ChangeVariableMutation { problem.InputVariables() };
        auto changeFunc          = ChangeFunctionMutation { problem.GetGrammar() };
        mutator.Add(onePoint, 1.0);
        mutator.Add(changeVar, 1.0);
        mutator.Add(changeFunc, 1.0);

        Evaluator evaluator(problem);
        evaluator.LocalOptimizationIterations(config.Iterations);
        evaluator.Budget(config.Evaluations);

        Expects(problem.TrainingRange().Size() > 0);

        auto parseSelector = [&](const std::string& name) -> Selector* {
            if (result.count(name) == 0) {
                return new TournamentSelector<Individual<1>, idx> { 5u };
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
                    return new TournamentSelector<Ind, 0> { tSize };
                } else if (tokens[0] == "proportional") {
                    return new ProportionalSelector<Ind, 0> {};
                } else if (tokens[0] == "rank") {
                    size_t tSize = 5;
                    if (tokens.size() > 1) {
                        if (auto [p, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), tSize); ec != std::errc()) {
                            fmt::print(stderr, "{}\n{}\n", "Error: could not parse tournament size argument.", opts.help());
                            exit(EXIT_FAILURE);
                        }
                    }
                    return new RankTournamentSelector<Ind, 0> { tSize };
                } else if (tokens[0] == "random") {
                    return new RandomSelector<Ind, 0> {};
                }
            }
            return new TournamentSelector<Individual<1>, idx> { 5u };
        };

        std::unique_ptr<Selector> femaleSelector;
        std::unique_ptr<Selector> maleSelector;

        femaleSelector.reset(parseSelector("female-selector"));
        maleSelector.reset(parseSelector("male-selector"));

        std::unique_ptr<OffspringGenerator> generator;
        if (result.count("offspring-generator") == 0) {
            generator.reset(new BasicOffspringGenerator(evaluator, crossover, mutator, *femaleSelector, *maleSelector));
        } else {
            auto value = result["offspring-generator"].as<std::string>();
            auto tokens = Split(value, ':');
            if (tokens[0] == "basic") {
                generator.reset(new BasicOffspringGenerator(evaluator, crossover, mutator, *femaleSelector, *maleSelector));
            } else if (tokens[0] == "brood") {
                size_t broodSize = 10;
                if (tokens.size() > 1) {
                    if (auto [p, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), broodSize); ec != std::errc()) {
                        fmt::print(stderr, "{}\n{}\n", "Error: could not parse brood size argument.", opts.help());
                        exit(EXIT_FAILURE);
                    }
                }
                auto ptr = new BroodOffspringGenerator(evaluator, crossover, mutator, *femaleSelector, *maleSelector);
                ptr->BroodSize(broodSize);
                generator.reset(ptr);
            } else if (tokens[0] == "os") {
                size_t selectionPressure = 100;
                if (tokens.size() > 1) {
                    if (auto [p, ec] = std::from_chars(tokens[1].data(), tokens[1].data() + tokens[1].size(), selectionPressure); ec != std::errc()) {
                        fmt::print(stderr, "{}\n{}\n", "Error: could not parse brood size argument.", opts.help());
                        exit(EXIT_FAILURE);
                    }
                }
                auto ptr = new OffspringSelectionGenerator(evaluator, crossover, mutator, *femaleSelector, *maleSelector);
                ptr->MaxSelectionPressure(selectionPressure);
                generator.reset(ptr);
            }
        }
        std::unique_ptr<Reinserter> reinserter;
        if (result.count("reinserter") == 0) {
            reinserter.reset(new ReplaceWorstReinserter<Ind, idx>());
        } else {
            auto value = result["reinserter"].as<std::string>();
            if (value == "keep-best") {
                reinserter.reset(new KeepBestReinserter<Ind, idx>());
            } else if (value == "replace-worst") {
                reinserter.reset(new ReplaceWorstReinserter<Ind, idx>());
            }
        }

        Operon::Random random(config.Seed);
        if (result["shuffle"].as<bool>()) 
        {
            problem.GetDataset().Shuffle(random);
        }
        if (result["standardize"].as<bool>())
        {
            problem.StandardizeData(problem.TrainingRange());
        }

        tbb::global_control c(tbb::global_control::max_allowed_parallelism, threads);

        auto t0 = std::chrono::high_resolution_clock::now();

        GeneticProgrammingAlgorithm gp { problem, config, initializer, *generator, *reinserter };

        auto targetValues = problem.TargetValues();
        auto trainingRange = problem.TrainingRange();
        auto testRange = problem.TestRange();
        auto targetTrain = targetValues.subspan(trainingRange.Start(), trainingRange.Size());
        auto targetTest = targetValues.subspan(testRange.Start(), testRange.Size());

        // some boilerplate for reporting results
        auto getBest = [&](const gsl::span<const Ind> pop) -> Ind {
            auto [minElem, maxElem] = std::minmax_element(pop.begin(), pop.end(), [&](const auto& lhs, const auto& rhs) { return lhs.Fitness[idx] < rhs.Fitness[idx]; });
            return *minElem;
        };

        Ind best;

        auto report = [&]() {
            auto pop = gp.Parents();
            best = getBest(pop);

            //fmt::print("best: {}\n", InfixFormatter::Format(best.Genotype, *dataset));

            auto estimatedTrain = Evaluate<Operon::Scalar>(best.Genotype, problem.GetDataset(), trainingRange);
            auto estimatedTest = Evaluate<Operon::Scalar>(best.Genotype, problem.GetDataset(), testRange);

            // scale values
            auto [a, b] = LinearScalingCalculator::Calculate(estimatedTrain.begin(), estimatedTrain.end(), targetTrain.begin());
            std::transform(estimatedTrain.begin(), estimatedTrain.end(), estimatedTrain.begin(), [a = a, b = b](Operon::Scalar v) { return b * v + a; });
            std::transform(estimatedTest.begin(), estimatedTest.end(), estimatedTest.begin(), [a = a, b = b](Operon::Scalar v) { return b * v + a; });

            auto r2Train = RSquared(estimatedTrain, targetTrain);
            auto r2Test = RSquared(estimatedTest, targetTest);

            auto nmseTrain = NormalizedMeanSquaredError(estimatedTrain, targetTrain);
            auto nmseTest = NormalizedMeanSquaredError(estimatedTest, targetTest);

            auto rmseTrain = RootMeanSquaredError(estimatedTrain, targetTrain);
            auto rmseTest = RootMeanSquaredError(estimatedTest, targetTest);

            auto avgLength = std::transform_reduce(std::execution::par_unseq, pop.begin(), pop.end(), 0.0, std::plus<> {}, [](const auto& ind) { return ind.Genotype.Length(); }) / pop.size();
            auto avgQuality = std::transform_reduce(std::execution::par_unseq, pop.begin(), pop.end(), 0.0, std::plus<> {}, [=](const auto& ind) { return ind[idx]; }) / pop.size();

            auto t1 = std::chrono::high_resolution_clock::now();
            auto elapsed = std::chrono::duration_cast<std::chrono::milliseconds>(t1 - t0).count() / 1000.0;

            auto getSize = [](const Ind& ind) { return sizeof(ind) + sizeof(Node) * ind.Genotype.Nodes().capacity(); };

            // calculate memory consumption
            size_t totalMemory = std::transform_reduce(std::execution::par_unseq, pop.begin(), pop.end(), 0U, std::plus<Operon::Scalar>{}, getSize);
            auto off = gp.Offspring();
            totalMemory += std::transform_reduce(std::execution::par_unseq, off.begin(), off.end(), 0U, std::plus<Operon::Scalar>{}, getSize);

            fmt::print("{:.4f}\t{}\t", elapsed, gp.Generation() + 1);
            fmt::print("{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t", best[idx], r2Train, r2Test, rmseTrain, rmseTest, nmseTrain, nmseTest);
            fmt::print("{:.4f}\t{:.1f}\t{}\t{}\t{}\t", avgQuality, avgLength, evaluator.FitnessEvaluations(), evaluator.LocalEvaluations(), evaluator.TotalEvaluations());
            fmt::print("{}\t{}\n", totalMemory, config.Seed); 

            //fmt::print("best: {}\n", InfixFormatter::Format(best.Genotype, *dataset, 6));
        };

        gp.Run(random, report);
        //report();
    } catch (std::exception& e) {
        fmt::print("{}\n", e.what());
        std::exit(EXIT_FAILURE);
    }

    return 0;
}
