// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <cmath>
#include <cstdlib>
#include <fmt/core.h>
#include <limits>
#include <memory>

#include "operon/algorithms/enumeration.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/grammar.hpp"
#include "operon/core/problem.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/optimizer/optimizer.hpp"

#include "operator_factory.hpp"
#include "util.hpp"

namespace {
// Prints every EnumerationFunction's canonical name and whether `functions` enables it - the
// --function-set analogue of Operon::PrintPrimitives, since EnumerationFunction is a distinct
// vocabulary (RecurringFactor recipes, not raw PrimitiveSetConfig bits - see grammar.hpp).
auto PrintEnumerationFunctions(Operon::EnumerationFunctionSet functions) -> void
{
    fmt::print("Enumeration functions (RecurringFactor recipes):\n");
    fmt::print("{:<10}\t{:>7}\n", "Symbol", "Enabled");
    for (std::size_t i = 0; i < Operon::EnumerationFunctions::Count; ++i) {
        auto const fn = static_cast<Operon::EnumerationFunction>(i);
        fmt::print("{:<10}\t{:>7}\n", Operon::EnumerationFunctionName(fn), functions.Test(i));
    }
}
} // namespace

auto main(int argc, char** argv) -> int // NOLINT(bugprone-exception-escape)
{
    // InitOptions registers the full shared option set (population-size,
    // crossover-probability, etc.) used by all CLIs (operon_gp/operon_nsgp/
    // operon_parse_model) - most of those are GP-specific and don't apply to
    // this non-population-based algorithm. operon_enum only reads dataset/
    // train/test/target/inputs/enable-symbols/disable-symbols/show-primitives/
    // objective/mdl-likelihood/linear-scaling/iterations/seed/threads from it,
    // plus its own max-complexity/top-k/function-set/ranking below; everything
    // else shown in --help is inert here. Trimming InitOptions itself would
    // mean restructuring a utility shared by every existing CLI - out of scope
    // for this addition.
    auto opts = Operon::InitOptions("operon_enum", "Exhaustive grammar enumeration symbolic regression");
    opts.add_options()
        ("max-complexity", "Maximum expression complexity (count of all non-Constant nodes)", cxxopts::value<std::size_t>()->default_value("20"))
        ("top-k", "Number of best-fitness models to report", cxxopts::value<std::size_t>()->default_value("5"))
        ("function-set", "Enumeration function set: custom (default, from --enable-symbols/--disable-symbols), "
                          "keep_duplicates, core_maths, ext_maths, osc_maths, base10_maths, base_e_maths",
            cxxopts::value<std::string>()->default_value("custom"))
        ("ranking", "Ranking criterion: mdl (default, uses --mdl-likelihood) or objective (uses --objective)",
            cxxopts::value<std::string>()->default_value("mdl"));
    auto result = Operon::ParseOptions(std::move(opts), argc, argv);

    // --- function-set resolution: no dataset access yet, so --show-primitives can exit before any
    // dataset work (unknown function-set/symbol names also fail here, before dataset work). ---
    auto const functionSetName = result["function-set"].as<std::string>();
    auto const preset = Operon::ParseEnumerationPreset(functionSetName);
    if (functionSetName != "custom" && !preset) {
        fmt::print(stderr, "error: unknown --function-set '{}'\n", functionSetName);
        return EXIT_FAILURE;
    }

    Operon::EnumerationFunctionSet functions{};
    if (preset) {
        if (result.contains("enable-symbols") || result.contains("disable-symbols")) {
            fmt::print(stderr, "error: --function-set {} conflicts with --enable-symbols/--disable-symbols\n", functionSetName);
            return EXIT_FAILURE;
        }
        functions = Operon::PresetFunctions(*preset);
    } else {
        try {
            if (result.contains("enable-symbols")) {
                for (auto const& s : Operon::Split(result["enable-symbols"].as<std::string>(), ',')) {
                    auto fn = Operon::ParseEnumerationFunction(s);
                    if (!fn) { throw std::runtime_error(fmt::format("unrecognized symbol '{}'", s)); }
                    functions |= *fn;
                }
            }
            if (result.contains("disable-symbols")) {
                for (auto const& s : Operon::Split(result["disable-symbols"].as<std::string>(), ',')) {
                    auto fn = Operon::ParseEnumerationFunction(s);
                    if (!fn) { throw std::runtime_error(fmt::format("unrecognized symbol '{}'", s)); }
                    functions &= ~Operon::ToFunctionSet(*fn);
                }
            }
        } catch (std::exception& e) {
            fmt::print(stderr, "error: {}\n", e.what());
            return EXIT_FAILURE;
        }
    }

    if (result.contains("show-primitives")) {
        PrintEnumerationFunctions(functions);
        return EXIT_SUCCESS;
    }

    auto const rankingName = result["ranking"].as<std::string>();
    if (rankingName != "mdl" && rankingName != "objective") {
        fmt::print(stderr, "error: unknown --ranking '{}' (expected mdl or objective)\n", rankingName);
        return EXIT_FAILURE;
    }

    Operon::Range trainingRange;
    Operon::Range testRange;
    std::unique_ptr<Operon::Dataset> dataset;
    std::string targetName;

    dataset = std::make_unique<Operon::Dataset>(result["dataset"].as<std::string>(), /*hasHeader=*/true);
    if (result.contains("target")) { targetName = result["target"].as<std::string>(); }
    if (result.contains("train"))  { trainingRange = Operon::ParseRange(result["train"].as<std::string>()); }
    if (result.contains("test"))   { testRange = Operon::ParseRange(result["test"].as<std::string>()); }

    try {
        auto const target = Operon::ResolveTarget(*dataset, targetName);
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
        problem.SetLinearScalingEnabled(result["linear-scaling"].as<bool>());
        problem.SetLinearScalingOmitsNonFinite(result["skip-nonfinite"].as<bool>());
        // Every underlying built-in the enabled recipes need for interpretation (e.g. Cube/TenExp both
        // need Pow, Inv needs Div - see UnderlyingPrimitives), plus the baseline arithmetic ops the
        // always-on Expression/Term weighted-sum shape needs (Add/Mul).
        problem.ConfigurePrimitiveSet(Operon::PrimitiveSet::Arithmetic | Operon::UnderlyingPrimitives(functions));

        Operon::Grammar grammar;
        grammar.SetVariables(problem.GetInputs());
        grammar.Configure(functions);

        Operon::EnumerationConfig config;
        config.MaxComplexity = result["max-complexity"].as<std::size_t>();
        config.TopK = result["top-k"].as<std::size_t>();
        if (config.TopK == 0) {
            fmt::print(stderr, "error: --top-k must be at least 1\n");
            return EXIT_FAILURE;
        }
        config.Ranking = rankingName == "mdl" ? Operon::EnumerationRanking::MinimumDescriptionLength : Operon::EnumerationRanking::Objective;
        config.EvaluationBufferSize = problem.TrainingRange().Size();

        Operon::ScalarDispatch dtable;
        Operon::LevenbergMarquardtOptimizer<decltype(dtable), Operon::OptimizerType::Eigen> optimizer{ &dtable, &problem };
        // Enumeration always needs to fit coefficients (unlike GP, where
        // --iterations 0 sensibly means "no local search on top of the
        // evolved structure") - default to a reasonable non-zero iteration
        // count if the user didn't override it, since an unfit weighted-sum
        // Expression is not a meaningful result here.
        auto const iterations = result["iterations"].as<std::size_t>();
        optimizer.SetIterations(iterations == 0 ? 50 : iterations);

        // optimizer only drives CoefficientOptimizer's internal fit - ranking is via the scorer built
        // below (either MDL or the same --objective ErrorMetric GP/NSGP expose).
        std::unique_ptr<Operon::EvaluatorBase> objectiveEvaluator; // kept alive for the Objective scorer closure
        Operon::EnumerationScorer scorer;
        if (config.Ranking == Operon::EnumerationRanking::MinimumDescriptionLength) {
            auto const lik = result["mdl-likelihood"].as<std::string>();
            if (lik == "gaussian") {
                scorer = Operon::MakeMdlScorer<Operon::ScalarDispatch, Operon::GaussianLikelihood<Operon::Scalar>>(&problem, &dtable);
            } else if (lik == "poisson") {
                scorer = Operon::MakeMdlScorer<Operon::ScalarDispatch, Operon::PoissonLikelihood<Operon::Scalar>>(&problem, &dtable);
            } else {
                fmt::print(stderr, "error: unknown --mdl-likelihood '{}' (expected gaussian or poisson)\n", lik);
                return EXIT_FAILURE;
            }
        } else {
            objectiveEvaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable);
            scorer = Operon::MakeObjectiveScorer(objectiveEvaluator.get());
        }

        auto seed = result["seed"].as<Operon::RandomGenerator::result_type>();
        if (seed == 0) { seed = std::random_device{}(); }
        Operon::RandomGenerator rng(seed);

        Operon::GrammarEnumerationAlgorithm algo(config, std::move(grammar), &optimizer, std::move(scorer), rng);

        auto const threads = result["threads"].as<std::size_t>();
        algo.Run(rng, [&]() -> bool {
            auto best = algo.BestTrees();
            if (!best.empty()) {
                fmt::print("best score so far: {:.6g}\n", best.front().Score);
            }
            return false;
        }, threads);

        auto best = algo.BestTrees();
        if (best.empty()) {
            fmt::print(stderr, "no expression found within max-complexity={}\n", config.MaxComplexity);
            return EXIT_FAILURE;
        }
        for (auto const& r : best) {
            if (config.Ranking == Operon::EnumerationRanking::MinimumDescriptionLength) {
                auto const nllBits = r.NegativeLogLikelihood / std::log(2.0);
                fmt::print("mdl_bits={:.6g}\tnll_bits={:.6g}\tparameter_bits={:.6g}\tstructure_bits={:.6g}\t{:infix:roundtrip}\n",
                    r.Score, nllBits, r.ParameterCodeBits, r.StructureCodeBits, Operon::Fmt::WithNames{r.Tree, *problem.GetDataset()});
            } else {
                fmt::print("fitness={:.6g}\t{:infix:roundtrip}\n", r.Score, Operon::Fmt::WithNames{r.Tree, *problem.GetDataset()});
            }
        }
    } catch (std::exception& e) {
        fmt::print(stderr, "error: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return EXIT_SUCCESS;
}
