// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include <taskflow/taskflow.hpp>

#include "reporter.hpp"
#include "operon/algorithms/config.hpp"
#include "operon/algorithms/ga_base.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/optimizer/optimizer.hpp"

namespace Operon::Test {
namespace {

    constexpr std::size_t FunctionEvaluationsPerSearch = 3;
    constexpr std::size_t JacobianEvaluationsPerSearch = 2;
    using DTable = DispatchTable<Scalar>;

    class FixedWorkOptimizer final : public OptimizerBase {
    public:
        explicit FixedWorkOptimizer(gsl::not_null<Problem const*> problem) : OptimizerBase(problem) {}

        [[nodiscard]] auto Optimize(RandomGenerator&, Tree const& tree) const -> FitOutcome override
        {
            FitResult result;
            result.InitialParameters = tree.GetCoefficients();
            result.FinalParameters = result.InitialParameters;
            result.InitialCost = 1;
            result.FinalCost = 0;
            result.FunctionEvaluations = FunctionEvaluationsPerSearch;
            result.JacobianEvaluations = JacobianEvaluationsPerSearch;
            return result;
        }
    };

    auto MakeDataset() -> Dataset
    {
        std::vector<std::vector<Scalar>> values(2, std::vector<Scalar>(8));
        std::iota(values[0].begin(), values[0].end(), Scalar { 1 });
        for (std::size_t i = 0; i < values[1].size(); ++i) {
            values[1][i] = Scalar { 2 } * values[0][i];
        }
        return Dataset({ "X1", "Y" }, values);
    }

    auto ReadFile(std::filesystem::path const& path) -> std::string
    {
        std::ifstream input { path, std::ios::binary };
        return { std::istreambuf_iterator<char> { input }, std::istreambuf_iterator<char> {} };
    }

} // namespace

TEST_CASE("Reporter machine report includes combined GP evaluation diagnostics", "[cli][reporter][stats]")
{
    Dataset dataset { MakeDataset() };
    Problem problem { &dataset };
    problem.SetTrainingRange({ 0, 8 });
    problem.SetTestRange({ 0, 8 });
    problem.SetTarget("Y");

    DTable dtable;
    Evaluator<DTable> evaluator { &problem, &dtable, MSE {} };
    FixedWorkOptimizer optimizer { &problem };
    CoefficientOptimizer localSearch { &optimizer };

    PrimitiveSet pset;
    pset.SetConfig(PrimitiveSet::Arithmetic);
    std::vector<Hash> variables { dataset.GetVariable("X1")->Hash };
    BalancedTreeCreator creator { &pset, variables, 0, 6 };
    UniformTreeInitializer treeInitializer { &creator };
    treeInitializer.ParameterizeDistribution(std::size_t { 1 }, std::size_t { 4 });
    UniformCoefficientInitializer coefficientInitializer;
    SubtreeCrossover crossover { 1.0, 4, 12 };
    OnePointMutation<std::normal_distribution<Scalar>> onePoint;
    onePoint.ParameterizeDistribution(Scalar { 0 }, Scalar { 1 });
    MultiMutation mutation;
    mutation.Add(&onePoint, 1.0);
    TournamentSelector femaleSelector { SingleObjectiveComparison { 0 } };
    TournamentSelector maleSelector { SingleObjectiveComparison { 0 } };
    KeepBestReinserter reinserter { SingleObjectiveComparison { 0 } };
    BasicOffspringGenerator generator { &evaluator, &crossover, &mutation, &femaleSelector, &maleSelector,
        &localSearch };

    GeneticAlgorithmConfig config {};
    config.Generations = 1;
    config.PopulationSize = 4;
    config.PoolSize = 4;
    config.Seed = 17;
    config.CrossoverProbability = 1;
    config.MutationProbability = 1;
    config.LocalSearchProbability = 1;
    config.LamarckianProbability = 1;
    GeneticProgrammingAlgorithm gp { config, &problem, &treeInitializer, &coefficientInitializer, &generator,
        &reinserter };

    RandomGenerator rng { config.Seed };
    gp.Run(rng, nullptr, /*threads=*/1);

    auto const stats = gp.DetailedStats();
    REQUIRE(stats.Calls > 0);
    CHECK(stats.ResidualOptimizer == FunctionEvaluationsPerSearch * stats.Calls);
    CHECK(stats.JacobianOptimizer == JacobianEvaluationsPerSearch * stats.Calls);
    CHECK(stats.ResidualTotal() == stats.ResidualScoring + stats.ResidualOptimizer);
    CHECK(stats.JacobianTotal() == stats.JacobianScoring + stats.JacobianOptimizer);
    CHECK(stats.Consumed() == stats.ResidualTotal() + stats.JacobianTotal());

    Reporter<Evaluator<DTable>> reporter { &evaluator };
    tf::Executor executor { 1 };
    tf::Taskflow taskflow;
    taskflow.emplace([&] { reporter(executor, gp); });
    executor.run(taskflow).get();
    reporter.SetSymbolicModel("X1");
    auto const& report = reporter.GetMachineReport();
    CHECK(report.seed == config.Seed);
    CHECK(report.symbolic_model == "X1");
    CHECK(report.tree_node_count == reporter.GetBest().Genotype.Length());
    CHECK(report.metrics.evaluator_calls == stats.Calls);
    CHECK(report.metrics.result_evaluations == stats.ResidualTotal());
    CHECK(report.metrics.jacobian_evaluations == stats.JacobianTotal());
    CHECK(report.metrics.best_fitness == reporter.GetBest()[0]);
    CHECK(report.metrics.average_fitness >= report.metrics.best_fitness);
    CHECK(report.metrics.average_tree_node_count > 0);

    auto const path = std::filesystem::temp_directory_path() / "operon-reporter-regression.json";
    std::filesystem::remove(path);
    WriteMachineReport(report, path);
    auto const json = ReadFile(path);
    std::filesystem::remove(path);

    CHECK(json.find("\"report_kind\":\"operon_gp\"") != std::string::npos);
    CHECK(json.find("\"schema_version\":1") != std::string::npos);
    CHECK(json.find("\"symbolic_model\":\"X1\"") != std::string::npos);
    CHECK(json.find("\"tree_node_count\":" + std::to_string(report.tree_node_count)) != std::string::npos);
    CHECK(json.find("\"seed\":" + std::to_string(config.Seed)) != std::string::npos);
    CHECK(json.find("\"metrics\":{") != std::string::npos);
    CHECK(json.find("\"r2_train\":") != std::string::npos);
    CHECK(json.find("\"r2_test\":") != std::string::npos);
    CHECK(json.find("\"mae_train\":") != std::string::npos);
    CHECK(json.find("\"mae_test\":") != std::string::npos);
    CHECK(json.find("\"nmse_train\":") != std::string::npos);
    CHECK(json.find("\"nmse_test\":") != std::string::npos);
    CHECK(json.find("\"mse_train\":") != std::string::npos);
    CHECK(json.find("\"mse_test\":") != std::string::npos);
    CHECK(json.find("\"best_fitness\":") != std::string::npos);
    CHECK(json.find("\"average_fitness\":") != std::string::npos);
    CHECK(json.find("\"average_tree_node_count\":") != std::string::npos);
    CHECK(json.find("\"elapsed_seconds\":") != std::string::npos);
    CHECK(json.find("\"optimizer_seconds\":") != std::string::npos);
    CHECK(json.find("\"evaluator_calls\":" + std::to_string(stats.Calls)) != std::string::npos);
    CHECK(json.find("\"result_evaluations\":" + std::to_string(stats.ResidualTotal())) != std::string::npos);
    CHECK(json.find("\"jacobian_evaluations\":" + std::to_string(stats.JacobianTotal())) != std::string::npos);
}

} // namespace Operon::Test
