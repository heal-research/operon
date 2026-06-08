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
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/solvers/sgd.hpp"

#include "operator_factory.hpp"
#include "pareto_front.hpp"
#include "reporter.hpp"
#include "util.hpp"

namespace {
template<typename Lik, typename DTable>
auto MdlFrontSelect(DTable const& dtable, Operon::Problem const& problem,
                    Operon::Span<Operon::Individual const> pop) -> Operon::Individual
{
    Operon::MinimumDescriptionLengthEvaluator<DTable, Lik> mdlEval(&problem, &dtable);
    Operon::RandomGenerator rng(0);
    std::vector<Operon::Scalar> buf(problem.TrainingRange().Size());
    auto span = Operon::Span<Operon::Scalar>{buf.data(), buf.size()};

    Operon::Individual const* best{nullptr};
    auto bestMdl = std::numeric_limits<Operon::Scalar>::max();
    for (auto const& ind : pop) {
        if (ind.Rank != 0) { continue; }
        auto r = mdlEval(rng, ind, span);
        if (!best || r[0] < bestMdl) { bestMdl = r[0]; best = &ind; }
    }
    return best ? *best : pop.front();
}

template<typename Lik, typename DTable>
auto PenalizedLikelihoodFrontSelect(DTable const& dtable, Operon::Problem const& problem,
                                    Operon::Span<Operon::Individual const> pop,
                                    auto penaltyFn) -> Operon::Individual
{
    Operon::RandomGenerator rng(0);
    auto const nObs = static_cast<double>(problem.TrainingRange().Size());
    std::vector<Operon::Scalar> buf(nObs);

    Operon::Individual const* best{nullptr};
    auto bestScore = std::numeric_limits<double>::max();
    for (auto const& ind : pop) {
        if (ind.Rank != 0) { continue; }

        auto const& tree = ind.Genotype;
        auto const params = tree.GetCoefficients();
        auto const p = static_cast<double>(params.size());
        auto span = Operon::Span<Operon::Scalar>{buf.data(), buf.size()};

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{&dtable, problem.GetDataset(), &ind.Genotype};
        interpreter.Evaluate(params, problem.TrainingRange(), span);

        auto estimated = span;
        auto target = problem.TargetValues(problem.TrainingRange());

        Operon::Scalar profiledSigma{};
        std::span<Operon::Scalar const> sigmaSpan{};
        if constexpr (Lik::UsesSigma) {
            auto ssr = 0.0;
            for (auto i = 0; i < static_cast<std::ptrdiff_t>(nObs); ++i) {
                auto const e = static_cast<double>(estimated[i]) - static_cast<double>(target[i]);
                ssr += e * e;
            }
            profiledSigma = std::max(static_cast<Operon::Scalar>(std::sqrt(ssr / nObs)),
                                     std::numeric_limits<Operon::Scalar>::epsilon());
            sigmaSpan = std::span<Operon::Scalar const>{&profiledSigma, 1};
        }

        auto nll = static_cast<double>(Lik::ComputeLikelihood(estimated, target, sigmaSpan));
        if (!std::isfinite(nll)) { continue; }

        auto score = nll + penaltyFn(p, nObs);
        if (!best || score < bestScore) { bestScore = score; best = &ind; }
    }
    return best ? *best : pop.front();
}
} // namespace

auto main(int argc, char** argv) -> int
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
        auto const rows { dataset->Rows<std::size_t>() };
        if (result.count("train") == 0) {
            trainingRange = Operon::Range { 0, 2 * rows / 3 }; // by default use 66% of the data as training
        }
        if (result.count("test") == 0) {
            // if no test range is specified, we try to infer a reasonable range based on the trainingRange
            if (trainingRange.Start() > 0) {
                testRange = Operon::Range { 0, trainingRange.Start() };
            } else if (trainingRange.End() < rows) {
                testRange = Operon::Range { trainingRange.End(), dataset->Rows<std::size_t>() };
            } else {
                testRange = Operon::Range { 0, 1 };
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
        creator = ParseCreator(result["creator"].as<std::string>(), problem.GetPrimitiveSet(), problem.GetInputs(), maxLength);

        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        Operon::UniformTreeInitializer treeInitializer(creator.get());

        auto const initialMinDepth = result["creator-mindepth"].as<std::size_t>();
        auto const initialMaxDepth = result["creator-maxdepth"].as<std::size_t>();
        auto const initialMaxLength = result["creator-maxlength"].as<std::size_t>();
        treeInitializer.ParameterizeDistribution(amin + 1, initialMaxLength);
        treeInitializer.SetMinDepth(initialMinDepth);
        treeInitializer.SetMaxDepth(initialMaxDepth); // NOLINT

        std::unique_ptr<Operon::CoefficientInitializerBase> coeffInitializer;
        std::unique_ptr<Operon::MutatorBase> onePoint;
        if (symbolic) {
            using Dist = std::uniform_int_distribution<int>;
            coeffInitializer = std::make_unique<Operon::CoefficientInitializer<Dist>>();
            int constexpr range { 5 };
            dynamic_cast<Operon::CoefficientInitializer<Dist>*>(coeffInitializer.get())->ParameterizeDistribution(-range, +range);
            onePoint = std::make_unique<Operon::OnePointMutation<Dist>>();
            dynamic_cast<Operon::OnePointMutation<Dist>*>(onePoint.get())->ParameterizeDistribution(-range, +range);
        } else {
            using Dist = std::normal_distribution<Operon::Scalar>;
            coeffInitializer = std::make_unique<Operon::CoefficientInitializer<Dist>>();
            dynamic_cast<Operon::NormalCoefficientInitializer*>(coeffInitializer.get())->ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
            onePoint = std::make_unique<Operon::OnePointMutation<Dist>>();
            dynamic_cast<Operon::OnePointMutation<Dist>*>(onePoint.get())->ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
        }

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
        // DynamicPrimitives::Saxpy<Operon::Scalar, Operon::Backend::BatchSize<Operon::Scalar>> f{};
        // dtable.RegisterCallable(12345UL, f, f);

        auto scale = result["linear-scaling"].as<bool>();
        auto errorEvaluator = Operon::ParseEvaluator(result["objective"].as<std::string>(), problem, dtable, scale);
        errorEvaluator->SetBudget(config.Evaluations);

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
        Operon::CoefficientOptimizer cOpt { optimizer.get() };

        auto generator = Operon::ParseGenerator(result["offspring-generator"].as<std::string>(), evaluator, crossover, mutator, *femaleSelector, *maleSelector, &cOpt);
        auto reinserter = Operon::ParseReinserter(result["reinserter"].as<std::string>(), comp);

        std::unique_ptr<Operon::Zobrist> cache;
        if (result["transposition-cache"].as<bool>()) {
            Operon::RandomGenerator cacheRng(config.Seed);
            cache = std::make_unique<Operon::Zobrist>(cacheRng, static_cast<int>(maxLength));
            config.Cache = cache.get();
        }

        Operon::RandomGenerator random(config.Seed);
        if (result["shuffle"].as<bool>()) {
            problem.GetDataset()->Shuffle(random);
        }
        if (result["standardize"].as<bool>()) {
            problem.StandardizeData(problem.TrainingRange());
        }
        tf::Executor executor(threads);
        auto const sorterName = result["sorter"].as<std::string>();
        Operon::RankIntersectSorter rsSorter;
        Operon::MergeSorter msSorter;
        Operon::NondominatedSorterBase* sorterPtr = &rsSorter;
        if (sorterName == "ms") { sorterPtr = &msSorter; }
        Operon::NSGA2 gp { config, &problem, &treeInitializer, coeffInitializer.get(), generator.get(), reinserter.get(), sorterPtr };

        Operon::ModelSelectorFn modelSelector;
        auto const modelSelection = result["model-selection"].as<std::string>();
        if (modelSelection != "obj0") {
            auto const& lik = result["mdl-likelihood"].as<std::string>();
            if (lik == "gaussian") {
                if (modelSelection == "mdl") {
                    modelSelector = [&](auto pop) { return MdlFrontSelect<Operon::GaussianLikelihood<Operon::Scalar>>(dtable, problem, pop); };
                } else if (modelSelection == "bic") {
                    modelSelector = [&](auto pop) { return PenalizedLikelihoodFrontSelect<Operon::GaussianLikelihood<Operon::Scalar>>(dtable, problem, pop, [](auto p, auto n) { return p * std::log(n); }); };
                } else if (modelSelection == "aic") {
                    modelSelector = [&](auto pop) { return PenalizedLikelihoodFrontSelect<Operon::GaussianLikelihood<Operon::Scalar>>(dtable, problem, pop, [](auto, auto) { return 2.0; }); };
                }
            } else if (lik == "poisson") {
                if (modelSelection == "mdl") {
                    modelSelector = [&](auto pop) { return MdlFrontSelect<Operon::PoissonLikelihood<Operon::Scalar>>(dtable, problem, pop); };
                } else if (modelSelection == "bic") {
                    modelSelector = [&](auto pop) { return PenalizedLikelihoodFrontSelect<Operon::PoissonLikelihood<Operon::Scalar>>(dtable, problem, pop, [](auto p, auto n) { return p * std::log(n); }); };
                } else if (modelSelection == "aic") {
                    modelSelector = [&](auto pop) { return PenalizedLikelihoodFrontSelect<Operon::PoissonLikelihood<Operon::Scalar>>(dtable, problem, pop, [](auto, auto) { return 2.0; }); };
                }
            } else {
                throw std::runtime_error(fmt::format("unknown mdl-likelihood: {}", lik));
            }
            if (!modelSelector) {
                throw std::runtime_error(fmt::format("unknown model-selection criterion: {}", modelSelection));
            }
        }

        auto const* ptr = dynamic_cast<Operon::Evaluator<decltype(dtable)> const*>(errorEvaluator.get());
        Operon::Reporter<Operon::Evaluator<decltype(dtable)>> reporter(ptr, std::move(modelSelector));
        gp.Run(executor, random, [&]() { reporter(executor, gp); });
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
