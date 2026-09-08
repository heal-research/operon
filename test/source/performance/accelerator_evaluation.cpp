// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_floating_point.hpp>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <numeric>
#include <random>
#include <thread>
#include <utility>
#include <vector>

#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/taskflow.hpp>

#include "../operon_test.hpp"
#include "operon/core/pset.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/optimizer/population_encoding.hpp"
#if defined(OPERON_HAVE_HIP)
#include "operon/optimizer/hip_context.hpp"
#endif
#if defined(OPERON_HAVE_SYCL)
#include "operon/optimizer/sycl_context.hpp"
#endif

namespace Operon::Test {
namespace {

#if defined(OPERON_HAVE_HIP) || defined(OPERON_HAVE_SYCL)
struct EvaluationWorkload {
    Dataset Dataset;
    std::vector<Hash> Variables;
    std::vector<Scalar> Columns;
    std::vector<Individual> Population;
    PopulationOptimization::EncodedPopulation Encoded;
    std::vector<Scalar> Expected;
    char const* PrimitiveSetLabel{};
    std::size_t Rows{};
    std::size_t TotalNodes{};

    explicit EvaluationWorkload(std::size_t rows, bool includeTranscendentals)
        : Dataset([rows] {
              RandomGenerator rng{1234};
              return Util::RandomDataset(rng, static_cast<int>(rows), 4);
          }())
        , PrimitiveSetLabel(includeTranscendentals ? "arithmetic + sin/cos" : "arithmetic")
        , Rows(rows)
    {
        Variables = Dataset.VariableHashes();
        std::erase(Variables, Dataset.GetVariable("Y")->Hash);
        for (auto const variable : Variables) {
            auto const values = Dataset.GetValues(variable);
            Columns.insert(Columns.end(), values.begin(), values.end());
        }

        PrimitiveSet primitives{includeTranscendentals ? PrimitiveSet::Arithmetic | BuiltinOp::Sin | BuiltinOp::Cos : PrimitiveSet::Arithmetic};
        for (auto const op : {BuiltinOp::Add, BuiltinOp::Sub, BuiltinOp::Mul, BuiltinOp::Div}) {
            primitives.SetMinMaxArity(static_cast<Hash>(op), 2, 2);
        }
        primitives.SetFrequency(static_cast<Hash>(BuiltinOp::Div), 0);
        primitives.SetFrequency(static_cast<Hash>(BuiltinOp::Mul), 0);
        if (includeTranscendentals) {
            for (auto const op : {BuiltinOp::Sin, BuiltinOp::Cos}) {
                primitives.SetMinMaxArity(static_cast<Hash>(op), 1, 1);
            }
        }

        BalancedTreeCreator creator{&primitives, Variables, /*bias=*/0.25, 100};
        RandomGenerator rng{1234};
        std::uniform_int_distribution<std::size_t> lengthDistribution{1, 100};
        constexpr auto PopulationSize = std::size_t{1'000};
        Population.reserve(PopulationSize);
        ScalarDispatch dtable;
        while (Population.size() < PopulationSize) {
            auto tree = creator(rng, lengthDistribution(rng), 0, 32);
            auto firstVariable = true;
            for (auto& node : tree.Nodes()) {
                if (node.IsVariable()) {
                    node.Value = Scalar{0.1};
                    node.Optimize = std::exchange(firstVariable, false);
                } else if (node.IsConstant()) {
                    node.Optimize = false;
                }
            }
            auto expected = Interpreter<Scalar, ScalarDispatch>{&dtable, &Dataset, &tree}.Evaluate(tree.GetCoefficients(), Range{0, rows});
            if (!std::ranges::all_of(expected, [](Scalar value) { return std::isfinite(value); })) { continue; }
            TotalNodes += tree.Length();
            Expected.insert(Expected.end(), expected.begin(), expected.end());
            Individual individual{1};
            individual.Genotype = std::move(tree);
            Population.push_back(std::move(individual));
        }
        REQUIRE((!includeTranscendentals || std::ranges::any_of(Population, [](Individual const& individual) {
            return std::ranges::any_of(individual.Genotype.Nodes(), [](Node const& node) {
                return node.HashValue == static_cast<Hash>(BuiltinOp::Sin) || node.HashValue == static_cast<Hash>(BuiltinOp::Cos);
            });
        })));

        std::vector<std::size_t> selected(Population.size());
        std::iota(selected.begin(), selected.end(), 0);
        auto encoded = PopulationOptimization::EncodePopulation(Population, selected);
        REQUIRE(encoded);
        Encoded = std::move(*encoded);
        REQUIRE(Encoded.Trees.size() == Population.size());
        REQUIRE(Encoded.Nodes.size() == TotalNodes);
    }
};

auto BenchmarkCpu(EvaluationWorkload const& workload) -> void
{
    ScalarDispatch dtable;
    std::vector<Scalar> output(workload.Population.size() * workload.Rows);
    auto const batch = static_cast<double>(workload.TotalNodes * workload.Rows);
    ankerl::nanobench::Bench benchmark;
    benchmark.title("Population evaluation throughput (node-evaluations/s)").batch(batch).epochs(10).epochIterations(10);
    benchmark.run(fmt::format("CPU sequential; primitives={}; trees={}; rows={}", workload.PrimitiveSetLabel, workload.Population.size(), workload.Rows), [&] {
        for (std::size_t treeIndex = 0; treeIndex < workload.Population.size(); ++treeIndex) {
            auto const& tree = workload.Population[treeIndex].Genotype;
            auto const result = Interpreter<Scalar, ScalarDispatch>{&dtable, &workload.Dataset, &tree}.Evaluate(tree.GetCoefficients(), Range{0, workload.Rows});
            std::copy(result.begin(), result.end(), output.begin() + static_cast<std::ptrdiff_t>(treeIndex * workload.Rows));
        }
        ankerl::nanobench::doNotOptimizeAway(output);
    });
}

auto BenchmarkCpuParallel(EvaluationWorkload const& workload) -> void
{
    auto const workers = std::max(std::size_t{1}, static_cast<std::size_t>(std::thread::hardware_concurrency()) / 2);
    ScalarDispatch dtable;
    tf::Executor executor(workers);
    std::vector<Scalar> output(workload.Population.size() * workload.Rows);
    auto const batch = static_cast<double>(workload.TotalNodes * workload.Rows);
    ankerl::nanobench::Bench benchmark;
    benchmark.title("Population evaluation throughput (node-evaluations/s)").batch(batch).epochs(10).epochIterations(10);
    benchmark.run(fmt::format("CPU parallel; workers={}; primitives={}; trees={}; rows={}", workers, workload.PrimitiveSetLabel, workload.Population.size(), workload.Rows), [&] {
        tf::Taskflow taskflow;
        taskflow.for_each_index(std::size_t{0}, workload.Population.size(), std::size_t{1}, [&](std::size_t treeIndex) {
            auto const& tree = workload.Population[treeIndex].Genotype;
            auto const result = Interpreter<Scalar, ScalarDispatch>{&dtable, &workload.Dataset, &tree}.Evaluate(tree.GetCoefficients(), Range{0, workload.Rows});
            std::copy(result.begin(), result.end(), output.begin() + static_cast<std::ptrdiff_t>(treeIndex * workload.Rows));
        });
        executor.run(taskflow).wait();
        ankerl::nanobench::doNotOptimizeAway(output);
    });
}

template<typename Context>
auto VerifyAndBenchmark(char const* backend, EvaluationWorkload const& workload) -> void
{
    Context context;
    context.Upload(workload.Encoded, workload.Variables);
    auto const output = context.Evaluate(workload.Columns, workload.Variables.size(), workload.Rows, workload.Encoded.Coefficients);
    REQUIRE(output.size() == workload.Population.size() * workload.Rows);
    for (std::size_t index = 0; index < workload.Expected.size(); ++index) {
        CHECK_THAT(output[index], Catch::Matchers::WithinAbs(workload.Expected[index], 1e-5F));
    }

    auto const batch = static_cast<double>(workload.TotalNodes * workload.Rows);
    ankerl::nanobench::Bench steady;
    steady.title("Accelerator evaluation throughput (node-evaluations/s)").batch(batch).epochs(10).epochIterations(10);
    steady.run(fmt::format("{} steady; primitives={}; trees={}; rows={}", backend, workload.PrimitiveSetLabel, workload.Population.size(), workload.Rows), [&] {
        auto result = context.Evaluate(workload.Columns, workload.Variables.size(), workload.Rows, workload.Encoded.Coefficients);
        ankerl::nanobench::doNotOptimizeAway(result);
    });

    ankerl::nanobench::Bench upload;
    Context uploadContext;
    upload.title("Accelerator upload and evaluation throughput (node-evaluations/s)").batch(batch).epochs(10).epochIterations(10);
    upload.run(fmt::format("{} upload + evaluate; primitives={}; trees={}; rows={}", backend, workload.PrimitiveSetLabel, workload.Population.size(), workload.Rows), [&] {
        uploadContext.Upload(workload.Encoded, workload.Variables);
        auto result = uploadContext.Evaluate(workload.Columns, workload.Variables.size(), workload.Rows, workload.Encoded.Coefficients);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
}

#if defined(OPERON_HAVE_HIP)
auto BenchmarkHipResident(EvaluationWorkload const& workload) -> void
{
    PopulationOptimization::Hip::Context context;
    context.Upload(workload.Encoded, workload.Variables);
    auto const batch = static_cast<double>(workload.TotalNodes * workload.Rows);
    ankerl::nanobench::Bench benchmark;
    benchmark.title("Accelerator resident evaluation throughput (node-evaluations/s)").batch(batch).epochs(10).epochIterations(10);
    benchmark.run(fmt::format("HIP resident; primitives={}; trees={}; rows={}", workload.PrimitiveSetLabel, workload.Population.size(), workload.Rows), [&] {
        auto result = context.EvaluateResident(workload.Columns, workload.Variables.size(), workload.Rows);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
}
#endif
#if defined(OPERON_HAVE_HIP)
auto BenchmarkHipGaussianCosts(EvaluationWorkload const& workload) -> void
{
    PopulationOptimization::Hip::Context context;
    context.Upload(workload.Encoded, workload.Variables);
    auto const target = std::span{workload.Dataset.GetValues(workload.Dataset.GetVariable("Y")->Hash)}.subspan(0, workload.Rows);
    ankerl::nanobench::Bench benchmark;
    auto const batch = static_cast<double>(workload.TotalNodes * workload.Rows);
    benchmark.title("HIP fused Gaussian cost throughput (node-evaluations/s)").batch(batch).epochs(10).epochIterations(10);
    benchmark.run(fmt::format("HIP fused cost; primitives={}; trees={}; rows={}", workload.PrimitiveSetLabel, workload.Population.size(), workload.Rows), [&] {
        auto result = context.GaussianCosts(workload.Columns, workload.Variables.size(), workload.Rows, target);
        ankerl::nanobench::doNotOptimizeAway(result);
    });
}
#endif


#endif
} // namespace

#if defined(OPERON_HAVE_HIP) || defined(OPERON_HAVE_SYCL)
TEST_CASE("Accelerator population evaluation throughput", "[performance][accelerator]")
{
    for (auto const includeTranscendentals : {false, true}) {
        DYNAMIC_SECTION("primitives=" << (includeTranscendentals ? "arithmetic + sin/cos" : "arithmetic")) {
            for (auto const rows : {4'096UL, 65'536UL}) {
                DYNAMIC_SECTION("rows=" << rows) {
                    auto const workload = EvaluationWorkload{rows, includeTranscendentals};
                    BenchmarkCpuParallel(workload);
                    BenchmarkCpu(workload);
#if defined(OPERON_HAVE_HIP)
                    SECTION("HIP") { VerifyAndBenchmark<PopulationOptimization::Hip::Context>("HIP", workload); }
                    BenchmarkHipResident(workload);
                    BenchmarkHipGaussianCosts(workload);
#endif
#if defined(OPERON_HAVE_SYCL)
                    SECTION("SYCL") { VerifyAndBenchmark<PopulationOptimization::Sycl::Context>("SYCL", workload); }
#endif
                }
            }
        }
    }
}
#endif

} // namespace Operon::Test
