// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <catch2/catch_test_macros.hpp>

#include <algorithm>
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/algorithm/reduce.hpp>
#include <taskflow/taskflow.hpp>
#include <thread>
#include <utility>
#include <vstat/vstat.hpp>

#include "../operon_test.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/error_metrics/error_metrics.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"

namespace Operon::Test {
namespace {
auto TotalNodes(const Operon::Vector<Tree>& trees)
{
    return vstat::univariate::accumulate<Operon::Scalar>(trees.begin(), trees.end(), [](auto const& t) -> auto { return t.Length(); }).sum;
}
} // namespace

namespace nb = ankerl::nanobench;

namespace {
    template<typename T, typename DTable>
    void Evaluate(tf::Executor& executor, DTable const& dt, Operon::Vector<Tree> const& trees, Dataset const& ds, Range range)
    {
        tf::Taskflow taskflow;
        Operon::Vector<Operon::Vector<T>> results(executor.num_workers());
        for (auto& res : results) {
            res.resize(range.Size());
        }
        taskflow.for_each_index(size_t{0}, trees.size(), size_t{1}, [&](auto i) -> auto {
            auto& res = results[executor.this_worker_id()];
            auto coeff = trees[i].GetCoefficients();
            Operon::Interpreter<T, DTable>{&dt, &ds, &trees[i]}.Evaluate({coeff}, range, {res});
        });
        executor.run(taskflow).wait();
    }
} // namespace

TEST_CASE("Evaluation performance", "[performance]") // NOLINT(readability-function-cognitive-complexity)
{
    constexpr size_t n = 1000;
    constexpr size_t maxLength = 64;
    constexpr size_t maxDepth = 1000;
    constexpr size_t nrow = 12800;
    constexpr size_t ncol = 10;

    constexpr size_t minEpochIterations = 5;
    Operon::RandomGenerator rd(1234);
    auto ds = Util::RandomDataset(rd, nrow, ncol);
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable("Y").value().Hash);

    Range range = {0, nrow};

    PrimitiveSet pset;

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto creator = BalancedTreeCreator{&pset, inputs, /* bias= */ 0.0, maxLength};

    Operon::Vector<Tree> trees(n);

    using DTable = DispatchTable<Operon::Scalar, Operon::Seq<Dispatch::DefaultBatchSize<Operon::Scalar>>>;
    DTable dtable;

    auto test = [&](tf::Executor& executor, nb::Bench& b, PrimitiveSetConfig cfg, const std::string& name) -> void {
        pset.SetConfig(cfg);
        for (auto t : {NodeType::Add, NodeType::Sub, NodeType::Div, NodeType::Mul}) {
            pset.SetMinMaxArity(Node(t).HashValue, 2, 2);
        }
        std::ranges::generate(trees, [&]() -> Tree { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

        auto totalOps = TotalNodes(trees) * range.Size();
        b.batch(static_cast<double>(totalOps));
        b.run(name, [&]() -> void { Evaluate<Operon::Scalar>(executor, dtable, trees, ds, range); });
    };

    nb::Bench b;

    auto const maxConcurrency = std::thread::hardware_concurrency();

    SECTION("arithmetic") {
        b.title("arithmetic").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + exp") {
        b.title("arithmetic + exp").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | NodeType::Exp, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + log") {
        b.title("arithmetic + log").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | NodeType::Log, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + sin") {
        nb::Bench b2;
        b2.title("arithmetic + sin").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b2, PrimitiveSet::Arithmetic | NodeType::Sin, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + cos") {
        b.title("arithmetic + cos").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cos, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + tan") {
        b.title("arithmetic + tan").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | NodeType::Tan, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + sqrt") {
        b.title("arithmetic + sqrt").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sqrt, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + cbrt") {
        b.title("arithmetic + cbrt").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cbrt, fmt::format("N = {}", i));
        }
    }
}

TEST_CASE("Evaluator performance", "[performance]")
{
    const size_t n = 1000;
    const size_t maxLength = 100;
    const size_t maxDepth = 1000;

    constexpr size_t nrow = 10000;
    constexpr size_t ncol = 10;

    Operon::RandomGenerator rd(1234);
    auto ds = Util::RandomDataset(rd, nrow, ncol);

    auto variables = ds.GetVariables();
    auto target = variables.back().Name;
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target).value().Hash);
    Range range = {0, ds.Rows<std::size_t>()};

    Operon::Problem problem{&ds};
    problem.SetTrainingRange(range);
    problem.SetTestRange(range);
    problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);
    problem.SetTarget(target);

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    auto creator = BalancedTreeCreator{&problem.GetPrimitiveSet(), inputs, /* bias= */ 0.0, maxLength};

    Operon::Vector<Tree> trees(n);
    std::ranges::generate(trees, [&]() -> Tree { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

    Operon::Vector<Individual> individuals(n);
    for (size_t i = 0; i < individuals.size(); ++i) {
        individuals[i].Genotype = trees[i];
    }

    nb::Bench b;
    b.title("Evaluator performance").relative(true).performanceCounters(true).minEpochIterations(10);

    auto totalNodes = TotalNodes(trees);

    auto test = [&](std::string const& name, EvaluatorBase&& evaluator) -> void { // NOLINT(cppcoreguidelines-rvalue-reference-param-not-moved)
        evaluator.SetBudget(std::numeric_limits<size_t>::max());
        tf::Executor executor(std::thread::hardware_concurrency());
        tf::Taskflow taskflow;

        Operon::Vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());
        double sum{0};
        taskflow.transform_reduce(individuals.begin(), individuals.end(), sum, std::plus<>{}, [&](Operon::Individual& ind) -> Operon::Scalar {
            auto id = executor.this_worker_id();
            if (slots[id].size() < range.Size()) {
                slots[id].resize(range.Size());
            }
            return evaluator(rd, ind, slots[id]).front();
        });

        b.batch(static_cast<double>(totalNodes * range.Size())).epochs(10).epochIterations(100).run(name, [&]() -> double {
            sum = 0;
            executor.run(taskflow).wait();
            return sum;
        });
    };

    using DTable = ScalarDispatch;
    DTable dtable;

    test("c2", Operon::Evaluator<DTable>(&problem, &dtable, Operon::C2{}, /*linearScaling=*/false));
    test("r2", Operon::Evaluator<DTable>(&problem, &dtable, Operon::R2{}, /*linearScaling=*/false));
    test("nmse", Operon::Evaluator<DTable>(&problem, &dtable, Operon::NMSE{}, /*linearScaling=*/false));
    test("mae", Operon::Evaluator<DTable>(&problem, &dtable, Operon::MAE{}, /*linearScaling=*/false));
    test("mse", Operon::Evaluator<DTable>(&problem, &dtable, Operon::MSE{}, /*linearScaling=*/false));
}

// Accumulate path vs buf path — MSE only
//
// For MSE (scaling=false) the evaluator feeds interpreter batches directly into
// a vstat accumulator ("accumulate path"), never writing a full result span.
// The "buf path" reference leg manually calls Interpreter::Evaluate into a
// pre-allocated span and then calls MeanSquaredError on it — exactly what the
// old code did, same metric, same trees, same N.
//
// This isolates path overhead from metric-computation differences.
TEST_CASE("Accumulate vs buf path performance", "[performance]")
{
    constexpr size_t nTrees    = 500;
    constexpr size_t maxLength = 50;
    constexpr size_t maxDepth  = 1000;
    constexpr size_t ncol      = 10;

    using DTable = ScalarDispatch;
    DTable dtable;

    Operon::RandomGenerator rd(1234);

    auto runBench = [&](size_t nrow) {
        auto ds = Util::RandomDataset(rd, nrow, ncol);
        auto variables = ds.GetVariables();
        auto target    = variables.back().Name;
        auto inputs    = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable(target).value().Hash);
        Range const range{0, ds.Rows<std::size_t>()};

        Operon::Problem problem{&ds};
        problem.SetTrainingRange(range);
        problem.SetTestRange(range);
        problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);
        problem.SetTarget(target);

        std::uniform_int_distribution<size_t> sizeDist(1, maxLength);
        BalancedTreeCreator creator{&problem.GetPrimitiveSet(), inputs, 0.0, maxLength};

        Operon::Vector<Tree> trees(nTrees);
        std::ranges::generate(trees, [&]{ return creator(rd, sizeDist(rd), 0, maxDepth); });
        Operon::Vector<Individual> inds(nTrees);
        for (size_t i = 0; i < nTrees; ++i) { inds[i].Genotype = trees[i]; }

        auto const totalNodes = TotalNodes(trees);
        tf::Executor executor(std::thread::hardware_concurrency());
        auto const nWorkers = executor.num_workers();

        nb::Bench b;
        b.title(fmt::format("MSE: accumulate vs buf  N={}", nrow))
         .relative(true).performanceCounters(true).minEpochIterations(5);

        // accumulate path: Evaluator routes MSE+!scaling through Interpreter::Accumulate
        {
            Operon::Evaluator<DTable> ev{&problem, &dtable, Operon::MSE{}, false};
            ev.SetBudget(std::numeric_limits<size_t>::max());
            tf::Taskflow tf;
            Operon::Vector<Operon::Vector<Operon::Scalar>> slots(nWorkers);
            double sum{0};
            tf.transform_reduce(inds.begin(), inds.end(), sum, std::plus<>{},
                [&](Individual& ind) -> Operon::Scalar {
                    auto id = executor.this_worker_id();
                    if (slots[id].size() < range.Size()) { slots[id].resize(range.Size()); }
                    return ev(rd, ind, slots[id]).front();
                });
            b.batch(static_cast<double>(totalNodes * range.Size()))
             .epochs(5).epochIterations(50)
             .run("mse (accumulate)", [&]() -> double {
                 sum = 0; executor.run(tf).wait(); return sum;
             });
        }

        // buf path: Interpreter::Evaluate → pre-allocated span → MeanSquaredError
        // Same metric, same trees, same span — only the code path differs.
        {
            auto targetValues = problem.TargetValues(range);
            tf::Taskflow tf;
            Operon::Vector<Operon::Vector<Operon::Scalar>> slots(nWorkers);
            double sum{0};
            tf.transform_reduce(inds.begin(), inds.end(), sum, std::plus<>{},
                [&](Individual& ind) -> Operon::Scalar {
                    auto id = executor.this_worker_id();
                    if (slots[id].size() < range.Size()) { slots[id].resize(range.Size()); }
                    auto coeff = ind.Genotype.GetCoefficients();
                    Operon::Interpreter<Operon::Scalar, DTable>{&dtable, &ds, &ind.Genotype}
                        .Evaluate(coeff, range, slots[id]);
                    return static_cast<Operon::Scalar>(
                        MeanSquaredError<Operon::Scalar>(slots[id], targetValues));
                });
            b.batch(static_cast<double>(totalNodes * range.Size()))
             .epochs(5).epochIterations(50)
             .run("mse (buf)",        [&]() -> double {
                 sum = 0; executor.run(tf).wait(); return sum;
             });
        }
    };

    SECTION("N = 1 000")   { runBench(1'000); }
    SECTION("N = 5 000")   { runBench(5'000); }
    SECTION("N = 20 000")  { runBench(20'000); }
    SECTION("N = 100 000") { runBench(100'000); }
}

TEST_CASE("Parallel interpreter", "[performance]")
{
    const size_t n = 1000;
    const size_t maxLength = 100;
    const size_t maxDepth = 1000;

    constexpr size_t nrow = 10000;
    constexpr size_t ncol = 10;

    Operon::RandomGenerator rd(1234);
    auto ds = Util::RandomDataset(rd, nrow, ncol);

    auto variables = ds.GetVariables();
    auto target = variables.back().Name;
    auto inputs = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target).value().Hash);
    Range range = {0, ds.Rows<std::size_t>()};

    std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
    Operon::PrimitiveSet pset;
    pset.SetConfig(Operon::PrimitiveSet::Arithmetic);
    auto creator = BalancedTreeCreator{&pset, inputs, /* bias= */ 0.0, maxLength};

    Operon::Vector<Tree> trees(n);
    std::ranges::generate(trees, [&]() -> Tree { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

    nb::Bench b;
    b.relative(true).epochs(10).minEpochIterations(100).performanceCounters(true);
    Operon::Vector<size_t> threads(std::thread::hardware_concurrency());
    std::iota(threads.begin(), threads.end(), 1);
    Operon::Vector<Operon::Scalar> result(trees.size() * range.Size());
    for (auto t : threads) {
        b.batch(static_cast<double>(TotalNodes(trees) * range.Size())).run(fmt::format("{} thread(s)", t), [&]() -> void { Operon::EvaluateTrees(trees, &ds, range, result, t); });
    }
}

} // namespace Operon::Test
