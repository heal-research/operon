// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <cstddef>
#include <limits>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>

#include "../operon_test.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/population_encoding.hpp"
#if defined(OPERON_HAVE_HIP)
#include "operon/optimizer/hip_context.hpp"
#endif

#if defined(OPERON_HAVE_HIP)
namespace Operon::Test {
namespace {

struct OptimizationWorkload {
    Dataset Dataset;
    std::vector<Hash> Variables;
    std::vector<Individual> Population;
    PopulationOptimization::EncodedPopulation Encoded;
    std::vector<Scalar> Columns;
    std::vector<Scalar> Target;
    std::size_t Rows{};

    OptimizationWorkload(std::size_t populationSize, std::size_t rows)
        : Dataset([rows] {
              RandomGenerator rng{9876};
              return Util::RandomDataset(rng, static_cast<int>(rows), 4);
          }())
        , Rows(rows)
    {
        Variables = Dataset.VariableHashes();
        auto const targetHash = Dataset.GetVariable("Y")->Hash;
        std::erase(Variables, targetHash);
        for (auto const variable : Variables) {
            auto const values = Dataset.GetValues(variable);
            Columns.insert(Columns.end(), values.begin(), values.end());
        }
        Target.assign(Dataset.GetValues(targetHash).begin(), Dataset.GetValues(targetHash).end());

        PrimitiveSet primitives{PrimitiveSet::Arithmetic | NodeType::Constant};
        primitives.SetFrequency(static_cast<Hash>(BuiltinOp::Div), 0);
        primitives.SetFrequency(static_cast<Hash>(BuiltinOp::Mul), 0);
        primitives.SetMinMaxArity(static_cast<Hash>(BuiltinOp::Add), 2, 2);
        primitives.SetMinMaxArity(static_cast<Hash>(BuiltinOp::Sub), 2, 2);
        BalancedTreeCreator creator{&primitives, Variables, /*bias=*/0.25, 100};
        RandomGenerator rng{1234};
        std::uniform_int_distribution<std::size_t> lengthDistribution{1, 100};
        Population.reserve(populationSize);
        while (Population.size() < populationSize) {
            auto tree = creator(rng, lengthDistribution(rng), 0, 32);
            for (auto& node : tree.Nodes()) {
                if (node.IsConstant()) {
                    node.Value = Scalar{0.1};
                    node.Optimize = true;
                } else if (node.IsVariable()) {
                    node.Value = Scalar{1};
                    node.Optimize = false;
                }
            }
            tree.UpdateNodes();
            if (tree.CoefficientsCount() == 0 || tree.CoefficientsCount() > 16) { continue; }
            Individual individual{1};
            individual.Genotype = std::move(tree);
            Population.push_back(std::move(individual));
        }

        std::vector<std::size_t> selected(Population.size());
        std::iota(selected.begin(), selected.end(), 0);
        auto encoded = PopulationOptimization::EncodePopulation(Population, selected);
        REQUIRE(encoded);
        Encoded = std::move(*encoded);
    }
};

void BenchmarkCpu(OptimizationWorkload& workload, uint32_t iterations)
{
    Problem problem{&workload.Dataset};
    problem.SetTarget("Y");
    problem.SetInputs(workload.Variables);
    problem.SetTrainingRange({0, workload.Rows});
    problem.SetLinearScalingEnabled(false);
    ScalarDispatch dtable;
    LevenbergMarquardtOptimizer<ScalarDispatch> optimizer{&dtable, &problem};
    optimizer.SetIterations(iterations);

    auto const workers = std::max(std::size_t{1}, static_cast<std::size_t>(std::thread::hardware_concurrency()) / 2);
    tf::Executor executor(workers);
    ankerl::nanobench::Bench benchmark;
    benchmark.title("Population LM throughput (trees/s)").batch(static_cast<double>(workload.Population.size())).epochs(3).epochIterations(1);
    benchmark.run(fmt::format("CPU parallel; workers={}; trees={}; rows={}; iterations={}", workers, workload.Population.size(), workload.Rows, iterations), [&] {
        tf::Taskflow taskflow;
        taskflow.for_each_index(std::size_t{0}, workload.Population.size(), std::size_t{1}, [&](std::size_t index) {
            RandomGenerator rng{static_cast<uint64_t>(index)};
            auto const outcome = optimizer.Optimize(rng, workload.Population[index].Genotype);
            ankerl::nanobench::doNotOptimizeAway(Diagnostics(outcome).FinalCost);
        });
        executor.run(taskflow).wait();
    });
}

void BenchmarkHip(OptimizationWorkload const& workload, uint32_t iterations)
{
    PopulationOptimization::Hip::Context context;
    context.Upload(workload.Encoded, workload.Variables);
    ankerl::nanobench::Bench benchmark;
    benchmark.title("Population LM throughput (trees/s)").batch(static_cast<double>(workload.Population.size())).epochs(3).epochIterations(1);
    benchmark.run(fmt::format("HIP batched; trees={}; rows={}; iterations={}", workload.Population.size(), workload.Rows, iterations), [&] {
        auto const result = context.OptimizeGaussian(workload.Columns, workload.Variables.size(), workload.Rows, workload.Target, {}, iterations);
        ankerl::nanobench::doNotOptimizeAway(result.FinalCosts);
    });
}

} // namespace

TEST_CASE("Accelerator population LM throughput", "[performance][accelerator][optimizer]")
{
    constexpr auto Iterations = uint32_t{8};
    for (auto const populationSize : {1'000UL, 4'000UL, 16'000UL}) {
        for (auto const rows : {100UL, 1'000UL}) {
            DYNAMIC_SECTION("trees=" << populationSize << "; rows=" << rows) {
                auto workload = OptimizationWorkload{populationSize, rows};
                BenchmarkCpu(workload, Iterations);
                BenchmarkHip(workload, Iterations);
            }
        }
    }
}

} // namespace Operon::Test
#endif
