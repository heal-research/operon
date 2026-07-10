// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <cstdlib>
#include <fmt/core.h>
#include <memory>

#include "operon/algorithms/enumeration.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/grammar.hpp"
#include "operon/core/problem.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/optimizer/optimizer.hpp"

#include "operator_factory.hpp"
#include "util.hpp"

auto main(int argc, char** argv) -> int // NOLINT(bugprone-exception-escape)
{
    // InitOptions registers the full shared option set (population-size,
    // crossover-probability, etc.) used by all CLIs (operon_gp/operon_nsgp/
    // operon_parse_model) - most of those are GP-specific and don't apply to
    // this non-population-based algorithm. operon_enum only reads dataset/
    // train/test/target/inputs/enable-symbols/disable-symbols/show-primitives/
    // objective/linear-scaling/iterations/seed from it, plus its own
    // max-complexity/top-k below; everything else shown in --help is inert
    // here. Trimming InitOptions itself would mean restructuring a utility
    // shared by every existing CLI - out of scope for this addition.
    auto opts = Operon::InitOptions("operon_enum", "Exhaustive grammar enumeration symbolic regression");
    opts.add_options()
        ("max-complexity", "Maximum expression complexity (count of all non-Constant nodes)", cxxopts::value<std::size_t>()->default_value("20"))
        ("top-k", "Number of best-fitness models to report", cxxopts::value<std::size_t>()->default_value("5"));
    auto result = Operon::ParseOptions(std::move(opts), argc, argv);

    Operon::Range trainingRange;
    Operon::Range testRange;
    std::unique_ptr<Operon::Dataset> dataset;
    std::string targetName;
    auto primitiveSetConfig = Operon::PrimitiveSet::Arithmetic;

    dataset = std::make_unique<Operon::Dataset>(result["dataset"].as<std::string>(), /*hasHeader=*/true);
    if (result.contains("target"))          { targetName = result["target"].as<std::string>(); }
    if (result.contains("train"))           { trainingRange = Operon::ParseRange(result["train"].as<std::string>()); }
    if (result.contains("test"))            { testRange = Operon::ParseRange(result["test"].as<std::string>()); }
    if (result.contains("enable-symbols"))  { primitiveSetConfig |= Operon::ParsePrimitiveSetConfig(result["enable-symbols"].as<std::string>()); }
    if (result.contains("disable-symbols")) { primitiveSetConfig &= ~Operon::ParsePrimitiveSetConfig(result["disable-symbols"].as<std::string>()); }
    if (result.contains("show-primitives")) { Operon::PrintPrimitives(primitiveSetConfig); return EXIT_SUCCESS; }

    try {
        auto res = dataset->GetVariable(targetName);
        if (!res) {
            fmt::print(stderr, "error: target variable {} does not exist in the dataset.", targetName);
            return EXIT_FAILURE;
        }
        auto const& target = *res;
        auto const rows { dataset->Rows<std::size_t>() };

        Operon::SetupRanges(result, *dataset, trainingRange, testRange);

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

        Operon::Grammar grammar(problem.GetPrimitiveSet().Config(), problem.GetInputs());

        Operon::EnumerationConfig config;
        config.MaxComplexity = result["max-complexity"].as<std::size_t>();
        config.TopK = result["top-k"].as<std::size_t>();
        if (config.TopK == 0) {
            fmt::print(stderr, "error: --top-k must be at least 1\n");
            return EXIT_FAILURE;
        }

        Operon::ScalarDispatch dtable;
        Operon::LevenbergMarquardtOptimizer<decltype(dtable), Operon::OptimizerType::Eigen> optimizer{ &dtable, &problem };
        // Enumeration always needs to fit coefficients (unlike GP, where
        // --iterations 0 sensibly means "no local search on top of the
        // evolved structure") - default to a reasonable non-zero iteration
        // count if the user didn't override it, since an unfit weighted-sum
        // Expression is not a meaningful result here.
        auto const iterations = result["iterations"].as<std::size_t>();
        optimizer.SetIterations(iterations == 0 ? 50 : iterations);

        // optimizer only drives CoefficientOptimizer's internal fit - ranking
        // is via evaluator (same --objective option GP/NSGP expose), so
        // --objective actually changes which models are reported here.
        auto evaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable, result["linear-scaling"].as<bool>());

        auto seed = result["seed"].as<Operon::RandomGenerator::result_type>();
        if (seed == 0) { seed = std::random_device{}(); }
        Operon::RandomGenerator rng(seed);

        Operon::GrammarEnumerationAlgorithm algo(config, std::move(grammar), &optimizer, evaluator.get(), rng);

        algo.Run(rng, [&]() -> bool {
            auto best = algo.BestTrees();
            if (!best.empty()) {
                fmt::print("best fitness so far: {:.6g}\n", best.front().first);
            }
            return false;
        });

        auto best = algo.BestTrees();
        if (best.empty()) {
            fmt::print(stderr, "no expression found within max-complexity={}\n", config.MaxComplexity);
            return EXIT_FAILURE;
        }
        for (auto const& [fitness, tree] : best) {
            fmt::print("fitness={:.6g}\t{}\n", fitness, Operon::InfixFormatter::Format(tree, *problem.GetDataset(), 6));
        }
    } catch (std::exception& e) {
        fmt::print(stderr, "error: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
