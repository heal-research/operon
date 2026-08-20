// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <chrono>
#include <cmath>
#include <cstdlib>
#include <fmt/core.h>
#include <fmt/ranges.h>
#include <limits>
#include <memory>
#include <stdexcept>
#include <unordered_map>
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
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/solvers/sgd.hpp"

#include "jit_setup.hpp"
#include "operator_factory.hpp"
#include "pareto_front.hpp"
#include "probes_config.hpp"
#include "reporter.hpp"
#include "shape_constraints_config.hpp"
#include "util.hpp"

namespace {
template<typename EvaluatorType>
auto FrontSelect(EvaluatorType const& eval, Operon::Span<Operon::Individual const> pop) -> Operon::Individual
{
    Operon::RandomGenerator rng(0);
    std::vector<Operon::Scalar> buf(eval.GetProblem()->TrainingRange().Size());
    auto span = Operon::Span<Operon::Scalar>{buf.data(), buf.size()};

    Operon::Individual const* best{nullptr};
    auto bestVal = std::numeric_limits<Operon::Scalar>::max();
    for (auto const& ind : pop) {
        if (ind.Rank != 0) { continue; }
        auto r = eval(rng, ind, span);
        if (!best || r[0] < bestVal) { bestVal = r[0]; best = &ind; }
    }
    return best ? *best : pop.front();
}
} // namespace

auto main(int argc, char** argv) -> int
{
    auto opts = Operon::InitOptions("operon_gp", "Genetic programming symbolic regression");
    opts.add_options()("sorter", "Non-dominated sorter: rs (RankIntersect) or ms (Merge)", cxxopts::value<std::string>()->default_value("rs"));
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

    // parse remaining config options
    Operon::Range trainingRange;
    Operon::Range testRange;
    std::unique_ptr<Operon::Dataset> dataset;
    std::string targetName;
    bool showPrimitiveSet = false;
    auto threads = std::thread::hardware_concurrency();
    auto primitiveSetConfig = Operon::PrimitiveSet::Arithmetic;

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
        auto const target = Operon::ResolveTarget(*dataset, targetName);
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
        problem.SetLinearScalingEnabled(result["linear-scaling"].as<bool>());
        problem.SetLinearScalingOmitsNonFinite(result["skip-nonfinite"].as<bool>());
        problem.ConfigurePrimitiveSet(primitiveSetConfig);

        auto [creator, creatorMaxLength, creatorMinDepth, creatorMaxDepth] = ParseCreator(
            result["creator"].as<std::string>(), problem.GetPrimitiveSet(), problem.GetInputs(),
            maxLength, result["creator-mindepth"].as<std::size_t>(), result["creator-maxdepth"].as<std::size_t>());

        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        Operon::UniformTreeInitializer treeInitializer(creator.get());
        treeInitializer.ParameterizeDistribution(amin + 1, creatorMaxLength);
        treeInitializer.SetMinDepth(creatorMinDepth);
        treeInitializer.SetMaxDepth(creatorMaxDepth); // NOLINT

        std::unique_ptr<Operon::CoefficientInitializerBase> coeffInitializer;
        std::unique_ptr<Operon::MutatorBase> onePoint;
        std::unique_ptr<Operon::MutatorBase> multiPoint;
        if (symbolic) {
            using Dist = std::uniform_int_distribution<int>;
            coeffInitializer = std::make_unique<Operon::CoefficientInitializer<Dist>>();
            int constexpr range { 5 };
            dynamic_cast<Operon::CoefficientInitializer<Dist>*>(coeffInitializer.get())->ParameterizeDistribution(-range, +range);
            onePoint = std::make_unique<Operon::OnePointMutation<Dist>>();
            dynamic_cast<Operon::OnePointMutation<Dist>*>(onePoint.get())->ParameterizeDistribution(-range, +range);
            multiPoint = std::make_unique<Operon::MultiPointMutation<Dist>>();
            dynamic_cast<Operon::MultiPointMutation<Dist>*>(multiPoint.get())->ParameterizeDistribution(-range, +range);
        } else {
            using Dist = std::normal_distribution<Operon::Scalar>;
            coeffInitializer = std::make_unique<Operon::CoefficientInitializer<Dist>>();
            dynamic_cast<Operon::NormalCoefficientInitializer*>(coeffInitializer.get())->ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
            onePoint = std::make_unique<Operon::OnePointMutation<Dist>>();
            dynamic_cast<Operon::OnePointMutation<Dist>*>(onePoint.get())->ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
            multiPoint = std::make_unique<Operon::MultiPointMutation<Dist>>();
            dynamic_cast<Operon::MultiPointMutation<Dist>*>(multiPoint.get())->ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
        }

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

        std::unordered_map<std::string, Operon::MutatorBase*> const availableMutators{
            {"onepoint", onePoint.get()},
            {"multipoint", multiPoint.get()},
            {"changevar", &changeVar},
            {"changefunc", &changeFunc},
            {"replacesubtree", &replaceSubtree},
            {"insertsubtree", &insertSubtree},
            {"removechild", &removeChild},
            {"removesubtree", &removeSubtree},
            {"discretepoint", &discretePoint},
            {"shuffle", &shuffleSubtrees},
        };
        Operon::ParseMutators(result["mutators"].as<std::string>(), availableMutators, mutator);

        Operon::ScalarDispatch dtable;
        // DynamicPrimitives::Saxpy<Operon::Scalar, Operon::Backend::BatchSize<Operon::Scalar>> f{};
        // dtable.RegisterCallable(12345UL, f, f);

        auto const jitMode = result["jit"].as<std::string>(); // "all", "jac", or ""
        if (jitMode == "all" && result["skip-nonfinite"].as<bool>()) {
            throw std::invalid_argument("--skip-nonfinite is not supported with --jit=all");
        }

        std::unique_ptr<Operon::Zobrist>       zobrist;
        std::unique_ptr<Operon::EvaluatorBase> errorEvaluator;
        std::unique_ptr<Operon::EvaluatorBase> jacEvalStorage;
        std::unique_ptr<Operon::OptimizerBase> optimizer;
        std::function<void()>                  jitReport = [](){};

        if (jitMode.empty()) {
            if (result["transposition-cache"].as<bool>()) {
                Operon::RandomGenerator cacheRng(config.Seed);
                zobrist = std::make_unique<Operon::Zobrist>(cacheRng, static_cast<int>(maxLength), problem.GetInputs(), result["cache-max-age"].as<size_t>());
                config.Cache = zobrist.get();
            }
            errorEvaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable,
                result["skip-nonfinite"].as<bool>(), result["nonfinite-penalty-weight"].as<double>());
            optimizer = std::make_unique<Operon::LevenbergMarquardtOptimizer<decltype(dtable), Operon::OptimizerType::Eigen>>(&dtable, &problem);
        } else {
            auto jobj = Operon::CLI::MakeJitObjects(
                jitMode, problem, dtable,
                result["objective"].as<std::string>(),
                result["jit-max-length"].as<int>(),
                result["jit-min-visits"].as<std::size_t>(),
                static_cast<int>(maxLength), config.Seed,
                result["cache-max-age"].as<std::size_t>());
            if (jobj.Error) { return EXIT_FAILURE; }
            errorEvaluator = std::move(jobj.Evaluator);
            jacEvalStorage = std::move(jobj.OptimizerJacEval);
            optimizer      = std::move(jobj.Optimizer);
            zobrist        = std::move(jobj.Zobrist);
            jitReport      = std::move(jobj.Report);
            if (result["transposition-cache"].as<bool>()) { config.Cache = zobrist.get(); }
            // "jac" mode: factory leaves evaluator null; create interpreter evaluator here.
            if (!errorEvaluator) {
                errorEvaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable,
                    result["skip-nonfinite"].as<bool>(), result["nonfinite-penalty-weight"].as<double>());
            }
            // unknown mode: factory returned null optimizer; fall back to defaults.
            if (!optimizer) {
                optimizer = std::make_unique<Operon::LevenbergMarquardtOptimizer<decltype(dtable), Operon::OptimizerType::Eigen>>(&dtable, &problem);
            }
        }
        errorEvaluator->SetBudget(config.Evaluations);
        optimizer->SetIterations(config.Iterations);

        Operon::TreePropertyEvaluator lengthEvaluator(&problem, [](Operon::Tree const& tree) {
            return static_cast<Operon::Scalar>(tree.Length());
        }, static_cast<Operon::Scalar>(maxLength));
        // Operon::EntropyEvaluator entropyEvaluator(&problem);

        auto shapeConstraints = Operon::LoadShapeConstraints(
            result.contains("shape-constraints-config") ? result["shape-constraints-config"].as<std::string>() : std::string{});
        if (!shapeConstraints && result.count("shape-enforcement") != 0) {
            throw std::invalid_argument("--shape-enforcement requires --shape-constraints-config");
        }

        Operon::ShapeConstraintEnforcement shapeEnforcement{Operon::ShapeConstraintEnforcement::None};
        std::unique_ptr<Operon::ShapeViolationEvaluator> shapePenaltyStorage;
        std::unique_ptr<Operon::ShapeViolationEvaluator> shapeExtraObjectiveStorage;
        std::unique_ptr<Operon::MultiEvaluator> penalizedErrorStorage;
        Operon::EvaluatorBase const* errorObjective = errorEvaluator.get();
        if (shapeConstraints) {
            shapeEnforcement = result.count("shape-enforcement") != 0
                ? Operon::ParseShapeEnforcement(result["shape-enforcement"].as<std::string>())
                : Operon::ShapeConstraintEnforcement::HardReject;
            auto const unknownViolation = result["shape-unknown-violation"].as<double>();
            auto const penaltyWeight = result["shape-penalty-weight"].as<double>();
            auto const worstValue = result["shape-worst-value"].as<double>();
            if (!(std::isfinite(unknownViolation) && unknownViolation >= 0.0)) {
                throw std::invalid_argument(fmt::format("--shape-unknown-violation must be a finite, non-negative value (got {})", unknownViolation));
            }
            if (!(std::isfinite(penaltyWeight) && penaltyWeight >= 0.0)) {
                throw std::invalid_argument(fmt::format("--shape-penalty-weight must be a finite, non-negative value (got {})", penaltyWeight));
            }
            if (!std::isfinite(worstValue)) {
                throw std::invalid_argument(fmt::format("--shape-worst-value must be finite (got {})", worstValue));
            }

            Operon::ShapeConstraintPolicy const policy{
                .Enforcement = shapeEnforcement,
                .UnknownViolation = static_cast<Operon::Scalar>(unknownViolation),
                .PenaltyWeight = static_cast<Operon::Scalar>(penaltyWeight),
            };
            if (auto error = Operon::ValidatePolicy(policy, /*isNsga2=*/true)) { throw std::invalid_argument(*error); }
            auto const boundMode = Operon::ParseShapeBoundMode(result["shape-bound-mode"].as<std::string>());

            if (Operon::HasFlag(shapeEnforcement, Operon::ShapeConstraintEnforcement::Penalty)) {
                shapePenaltyStorage = std::make_unique<Operon::ShapeViolationEvaluator>(
                    &problem, &dtable, *shapeConstraints, static_cast<Operon::Scalar>(penaltyWeight), static_cast<Operon::Scalar>(unknownViolation));
                shapePenaltyStorage->SetBoundMode(boundMode);
                penalizedErrorStorage = std::make_unique<Operon::MultiEvaluator>(&problem);
                penalizedErrorStorage->Add(errorEvaluator.get());
                penalizedErrorStorage->Add(shapePenaltyStorage.get());
                penalizedErrorStorage->SetAggregateType(Operon::MultiEvaluator::AggregateType::Sum);
                errorObjective = penalizedErrorStorage.get();
            }
            if (Operon::HasFlag(shapeEnforcement, Operon::ShapeConstraintEnforcement::ExtraObjective)) {
                shapeExtraObjectiveStorage = std::make_unique<Operon::ShapeViolationEvaluator>(
                    &problem, &dtable, *shapeConstraints, Operon::Scalar{1}, static_cast<Operon::Scalar>(unknownViolation));
                shapeExtraObjectiveStorage->SetBoundMode(boundMode);
            }
        }

        Operon::MultiEvaluator evaluator(&problem);
        evaluator.SetBudget(config.Evaluations);
        evaluator.Add(errorObjective);
        evaluator.Add(&lengthEvaluator);
        if (shapeExtraObjectiveStorage) { evaluator.Add(shapeExtraObjectiveStorage.get()); }
        // evaluator.Add(&entropyEvaluator);

        // Optional shape-constraint wrapper (see operon_gp.cpp for the same
        // pattern/rationale) — wraps the whole MultiEvaluator, so an
        // infeasible individual gets WorstValue() on every objective.
        std::unique_ptr<Operon::ShapeConstrainedEvaluator> shapeConstrainedStorage;
        if (shapeConstraints && Operon::HasFlag(shapeEnforcement, Operon::ShapeConstraintEnforcement::HardReject)) {
            shapeConstrainedStorage = std::make_unique<Operon::ShapeConstrainedEvaluator>(&evaluator, &dtable, *shapeConstraints);
            shapeConstrainedStorage->SetWorstValue(result["shape-worst-value"].as<double>());
            shapeConstrainedStorage->SetBoundMode(Operon::ParseShapeBoundMode(result["shape-bound-mode"].as<std::string>()));
        }
        Operon::EvaluatorBase* activeEvaluator = shapeConstrainedStorage ? static_cast<Operon::EvaluatorBase*>(shapeConstrainedStorage.get()) : static_cast<Operon::EvaluatorBase*>(&evaluator);

        EXPECT(problem.TrainingRange().Size() > 0);

        // Deliberately NOT swapped for Operon::FeasibilityFirstComparison
        // when shape constraints are active (unlike operon_gp.cpp): the
        // gate above already rejects an infeasible individual on every
        // NSGA2 objective at once, and a feasibility-first *crowded*
        // comparator (accounting for Rank/Distance too) is a separate
        // design this CLI doesn't build yet -- not an oversight.
        Operon::CrowdedComparison comp;

        auto femaleSelector = Operon::ParseSelector(result["female-selector"].as<std::string>(), comp);
        auto maleSelector = Operon::ParseSelector(result["male-selector"].as<std::string>(), comp);
        Operon::CoefficientOptimizer cOpt { optimizer.get() };

        auto generator = Operon::ParseGenerator(result["offspring-generator"].as<std::string>(), *activeEvaluator, crossover, mutator, *femaleSelector, *maleSelector, &cOpt);
        // Default 0: NSGA2 had no elitism before this option existed, so an
        // unspecified --elitism preserves that. Opt in explicitly to enable it.
        auto const eliteCount = result.count("elitism") ? result["elitism"].as<size_t>() : size_t{0};
        auto reinserter = Operon::ParseReinserter(result["reinserter"].as<std::string>(), comp, eliteCount);

        Operon::RandomGenerator random(config.Seed);
        if (result["shuffle"].as<bool>()) {
            problem.GetDataset()->Shuffle(random);
        }
        if (result["standardize"].as<bool>()) {
            problem.StandardizeData(problem.TrainingRange());
        }
        tf::Executor executor(threads);
        // Reuse the same executor for shape-constraint Prepare() rather than
        // each evaluator owning a private one -- see
        // ShapeConstrainedEvaluator::SetExecutor's doc comment.
        if (shapeConstrainedStorage) { shapeConstrainedStorage->SetExecutor(executor); }
        if (shapePenaltyStorage) { shapePenaltyStorage->SetExecutor(executor); }
        if (shapeExtraObjectiveStorage) { shapeExtraObjectiveStorage->SetExecutor(executor); }
        auto const sorterName = result["sorter"].as<std::string>();
        Operon::RankIntersectSorter rsSorter;
        Operon::MergeSorter msSorter;
        Operon::NondominatedSorterBase* sorterPtr = nullptr;
        if (sorterName == "rs") { sorterPtr = &rsSorter; }
        else if (sorterName == "ms") { sorterPtr = &msSorter; }
        else { throw std::runtime_error(fmt::format("unknown sorter: {}", sorterName)); }
        Operon::NSGA2 gp { config, &problem, &treeInitializer, coeffInitializer.get(), generator.get(), reinserter.get(), sorterPtr };

        Operon::ModelSelectorFn modelSelector;
        auto const modelSelection = result["model-selection"].as<std::string>();
        if (modelSelection != "obj0") {
            using DTable = decltype(dtable);
            if (modelSelection == "mdl") {
                auto const& lik = result["mdl-likelihood"].as<std::string>();
                if (lik == "gaussian") {
                    auto eval = std::make_shared<Operon::MinimumDescriptionLengthEvaluator<DTable, Operon::GaussianLikelihood<Operon::Scalar>>>(&problem, &dtable);
                    modelSelector = [eval](auto pop) { return FrontSelect(*eval, pop); };
                } else if (lik == "poisson") {
                    auto eval = std::make_shared<Operon::MinimumDescriptionLengthEvaluator<DTable, Operon::PoissonLikelihood<Operon::Scalar>>>(&problem, &dtable);
                    modelSelector = [eval](auto pop) { return FrontSelect(*eval, pop); };
                } else {
                    throw std::runtime_error(fmt::format("unknown mdl-likelihood: {}", lik));
                }
            } else if (modelSelection == "bic") {
                auto eval = std::make_shared<Operon::BayesianInformationCriterionEvaluator<DTable>>(&problem, &dtable);
                modelSelector = [eval](auto pop) { return FrontSelect(*eval, pop); };
            } else if (modelSelection == "aic") {
                auto eval = std::make_shared<Operon::AkaikeInformationCriterionEvaluator<DTable>>(&problem, &dtable);
                modelSelector = [eval](auto pop) { return FrontSelect(*eval, pop); };
            } else {
                throw std::runtime_error(fmt::format("unknown model-selection criterion: {}", modelSelection));
            }
        }

        std::unique_ptr<Operon::Evaluator<decltype(dtable)>> reporterEvalStorage;
        Operon::Evaluator<decltype(dtable)> const* ptr = nullptr;
        if (jitMode == "all") {
            reporterEvalStorage = std::make_unique<Operon::Evaluator<decltype(dtable)>>(&problem, &dtable, Operon::MSE{});
            ptr = reporterEvalStorage.get();
        } else {
            ptr = dynamic_cast<Operon::Evaluator<decltype(dtable)> const*>(errorEvaluator.get());
        }
        Operon::Reporter<Operon::Evaluator<decltype(dtable)>> reporter(ptr, std::move(modelSelector), activeEvaluator);
        auto const warmStart = Operon::ResumeFromCheckpoint(gp, random, result);
        if (warmStart && result.contains("probes-config")) {
            fmt::print(stderr, "warning: --probes-config sinks/traces truncate on start; resuming via --resume discards prior instrumentation history at any reused output path\n");
        }
        auto probes = Operon::LoadProbeConfig(result.contains("probes-config") ? result["probes-config"].as<std::string>() : std::string{});
        gp.Run(executor, random, [&]() -> bool {
            reporter(executor, gp);
            if (probes) { (*probes)(gp); }
            Operon::MaybeSaveCheckpoint(gp, random, result);
            return false;
        }, warmStart);
        if (probes) { probes->Finish(); }
        Operon::MaybeSaveCheckpoint(gp, random, result, /*force=*/true);
        jitReport();
        auto best = reporter.GetBest();
        if (shapeConstrainedStorage || shapePenaltyStorage || shapeExtraObjectiveStorage) {
            // Same rationale as operon_gp.cpp: hard-reject (or a penalty/extra-
            // objective run that never drove violation to zero) can still
            // return a "best" individual that isn't actually certified feasible
            // over the domain box -- flag that explicitly rather than leaving a
            // caller to assume printed-successfully means feasible.
            bool const feasible = shapeConstrainedStorage ? shapeConstrainedStorage->Feasible(best.Genotype)
                : shapeExtraObjectiveStorage    ? shapeExtraObjectiveStorage->Measure(best.Genotype).Feasible
                                                 : shapePenaltyStorage->Measure(best.Genotype).Feasible;
            fmt::print(stderr, "shape-constraints: final model is {}\n", feasible ? "feasible" : "INFEASIBLE (not certified over the domain box)");
        }
        fmt::print("{}\n", Operon::InfixFormatter::Format(best.Genotype, *problem.GetDataset(), std::numeric_limits<Operon::Scalar>::max_digits10));
        if (result.contains("pareto-front")) {
            Operon::WriteParetoFront(result["pareto-front"].as<std::string>(), gp.Individuals(), dtable, problem);
        }
    } catch (std::exception& e) {
        fmt::print(stderr, "error: {}\n", e.what());
        return EXIT_FAILURE;
    }

    return 0;
}
