// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "reporter.hpp"
#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fmt/core.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/taskflow.hpp>
#include <thread>
#include <unordered_map>

#include "operon/algorithms/gp.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/version.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/hash/zobrist.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/optimizer/optimizer.hpp"

#include "cli_error.hpp"
#include "jit_setup.hpp"
#include "operator_factory.hpp"
#include "probes_config.hpp"
#include "shape_constraints_config.hpp"
#include "util.hpp"

namespace {
auto MakeCoeffAndMutation(bool symbolic) -> std::tuple<std::unique_ptr<Operon::CoefficientInitializerBase>,
    std::unique_ptr<Operon::MutatorBase>, std::unique_ptr<Operon::MutatorBase>>
{
    if (symbolic) {
        using Dist = std::uniform_int_distribution<int>;
        auto ci = std::make_unique<Operon::CoefficientInitializer<Dist>>();
        int constexpr range { 5 };
        dynamic_cast<Operon::CoefficientInitializer<Dist>*>(ci.get())->ParameterizeDistribution(-range, +range);
        auto op = std::make_unique<Operon::OnePointMutation<Dist>>();
        dynamic_cast<Operon::OnePointMutation<Dist>*>(op.get())->ParameterizeDistribution(-range, +range);
        auto multi = std::make_unique<Operon::MultiPointMutation<Dist>>();
        dynamic_cast<Operon::MultiPointMutation<Dist>*>(multi.get())->ParameterizeDistribution(-range, +range);
        return { std::move(ci), std::move(op), std::move(multi) };
    }
    using Dist = std::normal_distribution<Operon::Scalar>;
    auto ci = std::make_unique<Operon::CoefficientInitializer<Dist>>();
    dynamic_cast<Operon::NormalCoefficientInitializer*>(ci.get())->ParameterizeDistribution(
        Operon::Scalar { 0 }, Operon::Scalar { 1 });
    auto op = std::make_unique<Operon::OnePointMutation<Dist>>();
    dynamic_cast<Operon::OnePointMutation<Dist>*>(op.get())->ParameterizeDistribution(
        Operon::Scalar { 0 }, Operon::Scalar { 1 });
    auto multi = std::make_unique<Operon::MultiPointMutation<Dist>>();
    dynamic_cast<Operon::MultiPointMutation<Dist>*>(multi.get())
        ->ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
    return { std::move(ci), std::move(op), std::move(multi) };
}
} // anonymous namespace

auto main(int argc, char** argv) -> int // NOLINT(bugprone-exception-escape)
{
    auto opts = Operon::InitOptions("operon_gp", "Genetic programming symbolic regression");
    opts.add_options()("report-json", "Write a versioned lossless machine-readable report to this path",
        cxxopts::value<std::string>());
    auto result = Operon::ParseOptions(std::move(opts), argc, argv);

    // parse and set default values
    Operon::GeneticAlgorithmConfig config {};
    config.Generations = Operon::GetLimit(result, "generations");
    config.PopulationSize = result["population-size"].as<size_t>();
    config.PoolSize = result["pool-size"].as<size_t>();
    config.Evaluations = Operon::GetLimit(result, "evaluations");
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

    Operon::ShapeBoundOptions shapeBoundOptions {};

    // Apply overrides from parsed options
    dataset = std::make_unique<Operon::Dataset>(result["dataset"].as<std::string>(), /*hasHeader=*/true);
    if (result.contains("seed")) {
        config.Seed = result["seed"].as<Operon::RandomGenerator::result_type>();
    }
    if (result.contains("train")) {
        trainingRange = Operon::ParseRange(result["train"].as<std::string>());
    }
    if (result.contains("test")) {
        testRange = Operon::ParseRange(result["test"].as<std::string>());
    }
    if (result.contains("target")) {
        targetName = result["target"].as<std::string>();
    }
    if (result.contains("enable-symbols")) {
        primitiveSetConfig |= Operon::ParsePrimitiveSetConfig(result["enable-symbols"].as<std::string>());
    }
    if (result.contains("disable-symbols")) {
        primitiveSetConfig &= ~Operon::ParsePrimitiveSetConfig(result["disable-symbols"].as<std::string>());
    }
    if (result.contains("threads")) {
        threads = static_cast<decltype(threads)>(result["threads"].as<size_t>());
    }
    if (result.contains("show-primitives")) {
        showPrimitiveSet = true;
    }

    try {
        shapeBoundOptions = Operon::ParseShapeBoundOptions(result["shape-bound-mode"].as<std::string>());
        // validated even without --transposition-cache, so a typo is not silently ignored
        auto const cacheOptions = Operon::ZobristOptions { .MaxAge = result["cache-max-age"].as<size_t>(),
            .Visibility = Operon::ParseCacheVisibility(result["cache-visibility"].as<std::string>()) };
        if (showPrimitiveSet) {
            Operon::PrintPrimitives(primitiveSetConfig);
            return EXIT_SUCCESS;
        }

        // set the target
        auto const target = Operon::ResolveTarget(*dataset, targetName);
        auto const rows { dataset->Rows<std::size_t>() };

        Operon::SetupRanges(result, *dataset, trainingRange, testRange);

        // validate training range
        if (trainingRange.Start() >= rows || trainingRange.End() > rows) {
            fmt::print(stderr, "error: the training range {}:{} exceeds the available data range ({} rows)\n",
                trainingRange.Start(), trainingRange.End(), dataset->Rows());
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
        problem.ConfigurePrimitiveSet(primitiveSetConfig);

        auto [creator, creatorMaxLength, creatorMinDepth, creatorMaxDepth]
            = ParseCreator(result["creator"].as<std::string>(), problem.GetPrimitiveSet(), problem.GetInputs(),
                maxLength, result["creator-mindepth"].as<std::size_t>(), result["creator-maxdepth"].as<std::size_t>());

        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        Operon::UniformTreeInitializer treeInitializer(creator.get());
        treeInitializer.ParameterizeDistribution(amin + 1, creatorMaxLength);
        treeInitializer.SetMinDepth(creatorMinDepth);
        treeInitializer.SetMaxDepth(creatorMaxDepth); // NOLINT

        auto [coeffInitializer, onePoint, multiPoint] = MakeCoeffAndMutation(symbolic);

        Operon::SubtreeCrossover crossover { crossoverInternalProbability, maxDepth, maxLength };
        Operon::MultiMutation mutator {};

        Operon::ChangeVariableMutation changeVar { problem.GetInputs() };
        Operon::ChangeFunctionMutation changeFunc { problem.GetPrimitiveSet() };
        Operon::ReplaceSubtreeMutation replaceSubtree { creator.get(), coeffInitializer.get(), maxDepth, maxLength };
        Operon::InsertSubtreeMutation insertSubtree { creator.get(), coeffInitializer.get(), maxDepth, maxLength };
        Operon::RemoveChildMutation removeChild { problem.GetPrimitiveSet() };
        Operon::RemoveSubtreeMutation removeSubtree { creator.get(), coeffInitializer.get(), maxDepth };
        Operon::ShuffleSubtreesMutation shuffleSubtrees;
        Operon::DiscretePointMutation discretePoint;
        for (auto v : Operon::Math::Constants) {
            discretePoint.Add(static_cast<Operon::Scalar>(v), 1);
        }

        std::unordered_map<std::string, Operon::MutatorBase*> const availableMutators {
            { "onepoint", onePoint.get() },
            { "multipoint", multiPoint.get() },
            { "changevar", &changeVar },
            { "changefunc", &changeFunc },
            { "replacesubtree", &replaceSubtree },
            { "insertsubtree", &insertSubtree },
            { "removechild", &removeChild },
            { "removesubtree", &removeSubtree },
            { "discretepoint", &discretePoint },
            { "shuffle", &shuffleSubtrees },
        };
        Operon::ParseMutators(result["mutators"].as<std::string>(), availableMutators, mutator);

        Operon::ScalarDispatch dtable;
        auto const jitMode = result["jit"].as<std::string>(); // "all", "jac", or ""
        if (jitMode == "all" && result["skip-nonfinite"].as<bool>()) {
            throw std::invalid_argument("--skip-nonfinite is not supported with --jit=all");
        }

        // Optional shape-constraint enforcement (Kronberger et al. 2021, arXiv: 2103.15624). The spec is resolved
        // first because the evaluator factories construct the typed ShapeConstrainedValueEvaluator over the concrete
        // evaluator before it is type-erased: the single evaluator object owns the complete constrained objective
        // for the run -- its single prediction evaluation, the fitted transform, the affine/interval shape
        // measurement and the enforcement (hard-reject to WorstValue(), or a penalty added to the error) happen
        // inside one operator() invocation.
        auto loadedShapeConstraints = Operon::LoadShapeConstraints(result.contains("shape-constraints-config")
                ? result["shape-constraints-config"].as<std::string>()
                : std::string {});
        if (!loadedShapeConstraints) {
            return Operon::Cli::Report(loadedShapeConstraints.error());
        }
        auto shapeConstraints = std::move(*loadedShapeConstraints);
        if (!shapeConstraints && result.count("shape-enforcement") != 0) {
            throw std::invalid_argument("--shape-enforcement requires --shape-constraints-config");
        }
        // --shape-constraints-config domain bounds are raw feature-space values,
        // resolved into the constraint evaluators here (before --standardize runs
        // below). Combining the two would silently certify constraints against
        // the wrong coordinate system, so reject it rather than mis-certify.
        if (shapeConstraints && result["standardize"].as<bool>()) {
            throw std::invalid_argument("--shape-constraints-config domains are raw feature-space bounds and are not "
                                        "currently transformed for --standardize; use one or the other");
        }

        std::optional<Operon::ShapeConstraintSpec> shapeSpec;
        if (shapeConstraints) {
            auto const shapeEnforcement = result.count("shape-enforcement") != 0
                ? Operon::ParseShapeEnforcement(result["shape-enforcement"].as<std::string>())
                : Operon::ShapeConstraintEnforcement::HardReject;
            auto const unknownViolation = result["shape-unknown-violation"].as<double>();
            auto const penaltyWeight = result["shape-penalty-weight"].as<double>();
            auto const worstValue = result["shape-worst-value"].as<double>();
            if (!(std::isfinite(unknownViolation) && unknownViolation >= 0.0)) {
                throw std::invalid_argument(fmt::format(
                    "--shape-unknown-violation must be a finite, non-negative value (got {})", unknownViolation));
            }
            if (!(std::isfinite(penaltyWeight) && penaltyWeight >= 0.0)) {
                throw std::invalid_argument(
                    fmt::format("--shape-penalty-weight must be a finite, non-negative value (got {})", penaltyWeight));
            }
            if (!std::isfinite(worstValue)) {
                throw std::invalid_argument(fmt::format("--shape-worst-value must be finite (got {})", worstValue));
            }

            Operon::ShapeConstraintPolicy const policy {
                .Enforcement = shapeEnforcement,
                .UnknownViolation = static_cast<Operon::Scalar>(unknownViolation),
                .PenaltyWeight = static_cast<Operon::Scalar>(penaltyWeight),
            };
            if (auto error = Operon::ValidatePolicy(policy, /*isNsga2=*/false)) {
                throw std::invalid_argument(*error);
            }
            shapeSpec = Operon::ShapeConstraintSpec {
                .Constraints = std::move(*shapeConstraints),
                .Policy = policy,
                .WorstValue = worstValue,
                .Bound = shapeBoundOptions,
            };
        }

        std::unique_ptr<Operon::Zobrist> zobrist;
        std::unique_ptr<Operon::EvaluatorBase> evaluator;
        std::unique_ptr<Operon::EvaluatorBase> jacEvalStorage;
        std::unique_ptr<Operon::OptimizerBase> optimizer;
        std::function<void()> jitReport = []() {};

        if (jitMode.empty()) {
            if (result["transposition-cache"].as<bool>()) {
                Operon::RandomGenerator cacheRng(config.Seed);
                zobrist = std::make_unique<Operon::Zobrist>(
                    cacheRng, static_cast<int>(maxLength), problem.GetInputs(), cacheOptions);
                config.Cache = zobrist.get();
            }
            evaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable,
                result["skip-nonfinite"].as<bool>(), result["nonfinite-penalty-weight"].as<double>(), shapeSpec);
            optimizer
                = std::make_unique<Operon::LevenbergMarquardtOptimizer<decltype(dtable), Operon::OptimizerType::Eigen>>(
                    &dtable, &problem);
        } else {
            auto jobj = Operon::CLI::MakeJitObjects(jitMode, problem, dtable, result["objective"].as<std::string>(),
                result["jit-max-length"].as<int>(), result["jit-min-visits"].as<std::size_t>(),
                static_cast<int>(maxLength), config.Seed, cacheOptions, shapeSpec);
            if (jobj.Error) {
                return EXIT_FAILURE;
            }
            evaluator = std::move(jobj.Evaluator);
            jacEvalStorage = std::move(jobj.OptimizerJacEval);
            optimizer = std::move(jobj.Optimizer);
            zobrist = std::move(jobj.Zobrist);
            jitReport = std::move(jobj.Report);
            if (result["transposition-cache"].as<bool>()) {
                config.Cache = zobrist.get();
            }
            // "jac" mode: factory leaves evaluator null; create interpreter evaluator here.
            if (!evaluator) {
                evaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable,
                    result["skip-nonfinite"].as<bool>(), result["nonfinite-penalty-weight"].as<double>(), shapeSpec);
            }
            // unknown mode: factory returned null optimizer; fall back to defaults.
            if (!optimizer) {
                optimizer = std::make_unique<
                    Operon::LevenbergMarquardtOptimizer<decltype(dtable), Operon::OptimizerType::Eigen>>(
                    &dtable, &problem);
            }
        }
        optimizer->SetIterations(config.Iterations);

        // The constrained evaluator (when shape constraints are configured) is `evaluator` itself: it is the
        // outermost evaluator the GP loop and the Reporter see, and it is still an `Operon::Evaluator<T>` for the
        // Reporter's typed R2/MSE stats dynamic_cast below. Its shape-constraint state is reached through the
        // non-evaluator detail::ShapeConstraintEnforcer base.
        auto const* shapeEvaluator = dynamic_cast<Operon::detail::ShapeConstraintEnforcer const*>(evaluator.get());
        if (shapeSpec && shapeEvaluator == nullptr) {
            throw std::logic_error("shape constraints are configured but the evaluator is not shape-constrained");
        }

        Operon::CoefficientOptimizer const cOpt { optimizer.get() };

        EXPECT(problem.TrainingRange().Size() > 0);

        // Single-objective comparison on objective 0.
        Operon::ComparisonCallback const comp = Operon::SingleObjectiveComparison {};

        auto femaleSelector
            = Operon::ParseSelector(result["female-selector"].as<std::string>(), Operon::ComparisonCallback { comp });
        auto maleSelector
            = Operon::ParseSelector(result["male-selector"].as<std::string>(), Operon::ComparisonCallback { comp });

        auto generator = Operon::ParseGenerator(result["offspring-generator"].as<std::string>(), *evaluator, crossover,
            mutator, *femaleSelector, *maleSelector, &cOpt);
        // Default 1: preserves GP's historical single-elite behavior (previously
        // a hardcoded offspring[0] overwrite in gp.cpp, now handled uniformly by
        // ReinserterBase - see reinserter.hpp).
        auto const eliteCount = result.count("elitism") ? result["elitism"].as<size_t>() : size_t { 1 };
        auto reinserter = Operon::ParseReinserter(
            result["reinserter"].as<std::string>(), Operon::ComparisonCallback { comp }, eliteCount);

        Operon::RandomGenerator random(config.Seed);
        if (result["shuffle"].as<bool>()) {
            problem.GetDataset()->Shuffle(random);
        }
        if (result["standardize"].as<bool>()) {
            problem.StandardizeData(problem.TrainingRange());
        }

        tf::Executor executor(threads);
        Operon::GeneticProgrammingAlgorithm gp { config, &problem, &treeInitializer, coeffInitializer.get(),
            generator.get(), reinserter.get() };

        auto const fingerprint = Operon::MakeRunFingerprint(gp, shapeSpec ? &shapeSpec->Constraints : nullptr);
        auto const warmStart = Operon::ResumeFromCheckpoint(gp, random, result, fingerprint);

        std::unique_ptr<Operon::Evaluator<decltype(dtable)>> reporterEvalStorage;
        Operon::Evaluator<decltype(dtable)> const* ptr = nullptr;
        if (jitMode == "all") {
            reporterEvalStorage
                = std::make_unique<Operon::Evaluator<decltype(dtable)>>(&problem, &dtable, Operon::MSE {});
            ptr = reporterEvalStorage.get();
        } else {
            ptr = dynamic_cast<Operon::Evaluator<decltype(dtable)> const*>(evaluator.get());
        }
        Operon::Reporter<Operon::Evaluator<decltype(dtable)>> reporter(ptr);
        if (warmStart && result.contains("probes-config")) {
            fmt::print(stderr,
                "warning: --probes-config sinks/traces truncate on start; resuming via --resume discards prior "
                "instrumentation history at any reused output path\n");
        }
        auto loadedProbes = Operon::LoadProbeConfig(
            result.contains("probes-config") ? result["probes-config"].as<std::string>() : std::string {});
        if (!loadedProbes) {
            return Operon::Cli::Report(loadedProbes.error());
        }
        auto probes = std::move(*loadedProbes);
        gp.Run(
            executor, random,
            [&]() -> bool {
                reporter(executor, gp);
                if (probes) {
                    (*probes)(gp);
                }
                Operon::MaybeSaveCheckpoint(gp, random, result, fingerprint);
                return false;
            },
            warmStart);
        if (probes) {
            probes->Finish();
        }
        Operon::MaybeSaveCheckpoint(gp, random, result, fingerprint, /*force=*/true);
        jitReport();
        auto best = reporter.GetBest();
        if (shapeEvaluator != nullptr) {
            // Hard-reject with no certifiably-feasible individual in the final
            // population still returns a "best" (tied at WorstValue()/highest
            // violation) -- flag that here so a caller relying on the printed
            // model can't mistake a failed-to-find-feasible run for an ordinary
            // successful one. The stored genotype is measured under its canonical
            // transform, which is the transform the exported model applies.
            bool const feasible = shapeEvaluator->Certifier().MeasureCanonical(best.Genotype).Feasible;
            fmt::print(stderr, "shape-constraints: final model is {}\n",
                feasible ? "feasible" : "INFEASIBLE (not certified over the domain box)");
        }
        auto const model
            = fmt::format("{:infix:roundtrip}", Operon::Fmt::TreeFormatArgs { best.Genotype, *problem.GetDataset() });
        fmt::print("{}\n", model);
        if (result.contains("report-json")) {
            reporter.SetSymbolicModel(model);
            Operon::WriteMachineReport(reporter.GetMachineReport(), result["report-json"].as<std::string>());
        }
    } catch (std::exception& e) {
        fmt::print(stderr, "error: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return 0;
}
