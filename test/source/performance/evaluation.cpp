// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

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
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/optimizer/optimizer.hpp"
#ifdef HAVE_ASMJIT
#include "operon/hash/zobrist.hpp"
#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#endif

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
        for (auto t : {BuiltinOp::Add, BuiltinOp::Sub, BuiltinOp::Div, BuiltinOp::Mul}) {
            pset.SetMinMaxArity(static_cast<Operon::Hash>(t), 2, 2);
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
            test(executor, b, PrimitiveSet::Arithmetic | BuiltinOp::Exp, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + log") {
        b.title("arithmetic + log").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | BuiltinOp::Log, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + sin") {
        nb::Bench b2;
        b2.title("arithmetic + sin").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b2, PrimitiveSet::Arithmetic | BuiltinOp::Sin, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + cos") {
        b.title("arithmetic + cos").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | BuiltinOp::Cos, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + tan") {
        b.title("arithmetic + tan").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | BuiltinOp::Tan, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + sqrt") {
        b.title("arithmetic + sqrt").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | BuiltinOp::Sqrt, fmt::format("N = {}", i));
        }
    }

    SECTION("arithmetic + cbrt") {
        b.title("arithmetic + cbrt").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
        for (size_t i = 1; i <= maxConcurrency; ++i) {
            tf::Executor executor(i);
            test(executor, b, PrimitiveSet::Arithmetic | BuiltinOp::Cbrt, fmt::format("N = {}", i));
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
    problem.SetLinearScalingEnabled(false);

    test("c2", Operon::Evaluator<DTable>(&problem, &dtable, Operon::C2{}));
    test("r2", Operon::Evaluator<DTable>(&problem, &dtable, Operon::R2{}));
    test("nmse", Operon::Evaluator<DTable>(&problem, &dtable, Operon::NMSE{}));
    test("mae", Operon::Evaluator<DTable>(&problem, &dtable, Operon::MAE{}));
    test("mse", Operon::Evaluator<DTable>(&problem, &dtable, Operon::MSE{}));
}

TEST_CASE("skipNonFinite_ evaluator performance", "[performance]")
{
    // Stage 4 benchmark checkpoint: compares default vs skipNonFinite_ mode
    // end-to-end (tree interpretation + metric), on (a) all-finite data and
    // (b) data with a fraction of X1 set to NaN, swept across dataset sizes.
    const size_t n = 100;
    const size_t maxLength = 30;
    const size_t maxDepth = 100;
    constexpr size_t ncol = 5;

    Operon::RandomGenerator rd(4321);

    auto makeDataset = [&](size_t nrow, double nonFiniteFraction) -> Operon::Dataset {
        std::uniform_real_distribution<Operon::Scalar> dist(-1.F, +1.F);
        std::vector<std::string> names(ncol);
        for (auto i = 0UL; i < ncol - 1; ++i) { names[i] = fmt::format("X{}", i + 1); }
        names.back() = "Y";
        std::vector<std::vector<Operon::Scalar>> data(ncol, std::vector<Operon::Scalar>(nrow));
        for (auto& col : data) {
            for (auto& v : col) { v = dist(rd); }
        }
        auto nBad = static_cast<size_t>(nonFiniteFraction * static_cast<double>(nrow));
        for (auto i = 0UL; i < nBad; ++i) {
            data[0][i] = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        }
        return Operon::Dataset(names, data);
    };

    nb::Bench b;
    b.relative(true).performanceCounters(true).epochs(10).epochIterations(100);

    for (auto nrow : { 1000UL, 20000UL, 500000UL }) {
        for (auto [label, nonFiniteFraction] : { std::pair{"all-finite", 0.0}, std::pair{"10pct-nonfinite", 0.10} }) {
            auto ds = makeDataset(nrow, nonFiniteFraction);
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

            auto totalNodes = TotalNodes(trees);

            b.context("nrow", std::to_string(nrow));
            b.context("data", label);

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

                b.batch(static_cast<double>(totalNodes * range.Size())).run(name, [&]() -> double {
                    sum = 0;
                    executor.run(taskflow).wait();
                    return sum;
                });
            };

            using DTable = ScalarDispatch;
            DTable dtable;
            problem.SetLinearScalingEnabled(false);

            test("mse (default)", Operon::Evaluator<DTable>(&problem, &dtable, Operon::MSE{}));
            test("mse (skipNonFinite)", Operon::Evaluator<DTable>(&problem, &dtable, Operon::MSE{}, /*skipNonFinite=*/true, /*nonFinitePenaltyWeight=*/1.0));
            test("nmse (default)", Operon::Evaluator<DTable>(&problem, &dtable, Operon::NMSE{}));
            test("nmse (skipNonFinite)", Operon::Evaluator<DTable>(&problem, &dtable, Operon::NMSE{}, /*skipNonFinite=*/true, /*nonFinitePenaltyWeight=*/1.0));
        }
    }
    auto constexpr csvTemplate = R"DELIM("name";"nrow";"data";"batch";"elapsed";"instructions";"branches";"branch misses"
{{#result}}"{{name}}";"{{context(nrow)}}";"{{context(data)}}";{{batch}};{{median(elapsed)}};{{median(instructions)}};{{median(branchinstructions)}};{{median(branchmisses)}}
{{/result}})DELIM";
    b.render(csvTemplate, std::cout);
}

namespace {
    struct PerfIsolatedFixture {
        Operon::Dataset ds;
        Operon::Vector<Individual> individuals;
        Range range;
        std::string target;
        std::vector<Operon::Hash> inputs;
    };

    auto MakePerfIsolatedFixture(size_t nrow, double nonFiniteFraction) -> PerfIsolatedFixture
    {
        constexpr size_t ncol = 5;
        constexpr size_t n = 100;
        constexpr size_t maxLength = 30;
        constexpr size_t maxDepth = 100;

        Operon::RandomGenerator rd(4321);
        std::uniform_real_distribution<Operon::Scalar> dist(-1.F, +1.F);
        std::vector<std::string> names(ncol);
        for (auto i = 0UL; i < ncol - 1; ++i) { names[i] = fmt::format("X{}", i + 1); }
        names.back() = "Y";
        std::vector<std::vector<Operon::Scalar>> data(ncol, std::vector<Operon::Scalar>(nrow));
        for (auto& col : data) {
            for (auto& v : col) { v = dist(rd); }
        }
        auto nBad = static_cast<size_t>(nonFiniteFraction * static_cast<double>(nrow));
        for (auto i = 0UL; i < nBad; ++i) {
            data[0][i] = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        }
        Operon::Dataset ds(names, data);
        auto variables = ds.GetVariables();
        auto target = variables.back().Name;
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable(target).value().Hash);
        Range range = {0, ds.Rows<std::size_t>()};

        Operon::PrimitiveSet pset;
        pset.SetConfig(Operon::PrimitiveSet::Arithmetic);
        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator{&pset, inputs, /* bias= */ 0.0, maxLength};

        Operon::Vector<Individual> individuals(n);
        for (auto& ind : individuals) {
            ind.Genotype = creator(rd, sizeDistribution(rd), 0, maxDepth);
        }
        return PerfIsolatedFixture{std::move(ds), std::move(individuals), range, target, inputs};
    }

    template<typename DTable>
    void RunPerfIsolated(EvaluatorBase const& evaluator, Operon::Vector<Individual>& individuals, Range range, int reps)
    {
        Operon::RandomGenerator rd(1);
        std::vector<Operon::Scalar> buf(range.Size());
        double sum = 0;
        for (int r = 0; r < reps; ++r) {
            for (auto& ind : individuals) {
                sum += evaluator(rd, ind, buf).front();
            }
        }
        // prevent the loop from being optimized away
        volatile double sink = sum;
        (void)sink;
    }

    template<typename DTable>
    void RunPerfIsolatedCase(Operon::ErrorMetric metric, bool skipNonFinite)
    {
        auto fx = MakePerfIsolatedFixture(500000, 0.10);
        Operon::Problem problem{&fx.ds};
        problem.SetTrainingRange(fx.range);
        problem.SetTestRange(fx.range);
        problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);
        problem.SetTarget(fx.target);
        problem.SetLinearScalingEnabled(false);
        DTable dtable;
        Operon::Evaluator<DTable> ev(&problem, &dtable, metric, skipNonFinite, /*nonFinitePenaltyWeight=*/1.0);
        RunPerfIsolated<DTable>(ev, fx.individuals, fx.range, 20);
    }
} // namespace

TEST_CASE("perf-isolated: mse default", "[performance][perf-isolated]")
{
    RunPerfIsolatedCase<ScalarDispatch>(Operon::MSE{}, false);
}

TEST_CASE("perf-isolated: mse skipNonFinite", "[performance][perf-isolated]")
{
    RunPerfIsolatedCase<ScalarDispatch>(Operon::MSE{}, true);
}

TEST_CASE("perf-isolated: nmse default", "[performance][perf-isolated]")
{
    RunPerfIsolatedCase<ScalarDispatch>(Operon::NMSE{}, false);
}

TEST_CASE("perf-isolated: nmse skipNonFinite", "[performance][perf-isolated]")
{
    RunPerfIsolatedCase<ScalarDispatch>(Operon::NMSE{}, true);
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

#ifdef HAVE_ASMJIT
TEST_CASE("JIT evaluator performance", "[performance][jit]")
{
    constexpr size_t n         = 1000;
    constexpr size_t maxLength = 64;
    constexpr size_t maxDepth  = 1000;
    constexpr size_t nrow      = 10000;
    constexpr size_t ncol      = 10;
    // Use physical core count (SMT doubles the reported value; AVX/FMA units
    // are shared per physical core so virtual threads add contention, not throughput).
    auto const nPhysical = std::max(std::size_t{1}, static_cast<std::size_t>(std::thread::hardware_concurrency()) / 2);

    Operon::RandomGenerator rd(1234);
    auto ds = Util::RandomDataset(rd, nrow, ncol);

    auto variables = ds.GetVariables();
    auto target    = variables.back().Name;
    auto inputs    = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target).value().Hash);
    Range range = {0, ds.Rows<std::size_t>()};

    Problem problem{&ds};
    problem.SetTrainingRange(range);
    problem.SetTestRange(range);
    problem.SetTarget(target);

    Operon::JIT::JitZobrist zobrist(rd, static_cast<int>(maxLength), inputs);

    auto bench_pset = [&](PrimitiveSetConfig cfg, std::string const& title) {
        problem.GetPrimitiveSet().SetConfig(cfg);
        for (auto t : {BuiltinOp::Add, BuiltinOp::Sub, BuiltinOp::Div, BuiltinOp::Mul}) {
            problem.GetPrimitiveSet().SetMinMaxArity(static_cast<Operon::Hash>(t), 2, 2);
        }

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator{&problem.GetPrimitiveSet(), inputs, 0.0, maxLength};

        Vector<Tree> trees(n);
        std::ranges::generate(trees, [&]() -> Tree { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

        Vector<Individual> individuals(n);
        for (size_t i = 0; i < n; ++i) individuals[i].Genotype = trees[i];

        auto const totalNodes = TotalNodes(trees);

        ScalarDispatch dtable;
        problem.SetLinearScalingEnabled(false);
        Evaluator<ScalarDispatch> interp(&problem, &dtable, MSE{});
        JIT::JitEvaluator          jit   (&problem, &zobrist, MSE{});
        interp.SetBudget(std::numeric_limits<size_t>::max());
        jit   .SetBudget(std::numeric_limits<size_t>::max());

        // --- cold-cache run: first pass includes compilation ---
        auto const nThreads = nPhysical;
        for (auto nT : {std::size_t{1}, nThreads}) {
            nb::Bench bc;
            bc.title(fmt::format("{} / cold cache ({} thread{})", title, nT, nT == 1 ? "" : "s"))
              .performanceCounters(true).epochs(1).epochIterations(1);
            tf::Executor executor(nT);
            tf::Taskflow taskflow;
            Vector<Vector<Scalar>> slots(executor.num_workers());
            for (auto& s : slots) { s.resize(range.Size()); }
            double sum{0};
            taskflow.transform_reduce(individuals.begin(), individuals.end(), sum, std::plus<>{},
                [&](Individual& ind) -> Scalar {
                    auto& buf = slots[executor.this_worker_id()];
                    return jit(rd, ind, {buf}).front();
                });
            bc.batch(static_cast<double>(totalNodes * range.Size()))
              .run(fmt::format("jit (cold, {} thread{})", nT, nT == 1 ? "" : "s"),
                   [&]() -> double {
                       jit.ClearCache();
                       sum = 0;
                       executor.run(taskflow).wait();
                       return sum;
                   });
        }

        // Pre-warm JIT cache for the remaining comparisons.
        {
            Vector<Scalar> buf(range.Size());
            for (auto& ind : individuals) { jit(rd, ind, {buf}); }
        }

        // --- single-threaded warm comparison ---
        {
            nb::Bench bw;
            bw.title(title + " / warm cache (1 thread)")
              .relative(true).performanceCounters(true)
              .epochs(5).minEpochIterations(20);

            auto run1 = [&](std::string const& name, EvaluatorBase& ev) {
                tf::Executor executor(1);
                tf::Taskflow taskflow;
                Vector<Scalar> buf(range.Size());
                double sum{0};
                taskflow.transform_reduce(individuals.begin(), individuals.end(), sum, std::plus<>{},
                    [&](Individual& ind) -> Scalar { return ev(rd, ind, {buf}).front(); });
                bw.batch(static_cast<double>(totalNodes * range.Size()))
                  .run(name, [&]() -> double {
                      sum = 0; executor.run(taskflow).wait(); return sum;
                  });
            };

            run1("interpreter", interp);
            run1("jit", jit);
        }

        // --- parallel warm comparison ---
        {
            auto const nThreads = nPhysical;
            nb::Bench bp;
            bp.title(fmt::format("{} / warm cache ({} threads)", title, nThreads))
              .relative(true).performanceCounters(true)
              .epochs(5).minEpochIterations(10);

            auto runN = [&](std::string const& name, EvaluatorBase& ev) {
                tf::Executor executor(nThreads);
                tf::Taskflow taskflow;
                Vector<Vector<Scalar>> slots(executor.num_workers());
                for (auto& s : slots) s.resize(range.Size());
                double sum{0};
                taskflow.transform_reduce(individuals.begin(), individuals.end(), sum, std::plus<>{},
                    [&](Individual& ind) -> Scalar {
                        auto& buf = slots[executor.this_worker_id()];
                        return ev(rd, ind, {buf}).front();
                    });
                bp.batch(static_cast<double>(totalNodes * range.Size()))
                  .run(name, [&]() -> double {
                      sum = 0; executor.run(taskflow).wait(); return sum;
                  });
            };

            runN("interpreter", interp);
            runN("jit", jit);
        }

        fmt::print("  JIT cache: {} entries, {} hits\n", jit.CacheSize(), jit.CacheHits());
    };

    SECTION("arithmetic")       { bench_pset(PrimitiveSet::Arithmetic, "arithmetic"); }
    SECTION("arithmetic + sin") { bench_pset(PrimitiveSet::Arithmetic | BuiltinOp::Sin, "arithmetic+sin"); }
    SECTION("arithmetic + exp") { bench_pset(PrimitiveSet::Arithmetic | BuiltinOp::Exp, "arithmetic+exp"); }
}

TEST_CASE("JIT LM optimizer performance", "[performance][jit][optimizer]")
{
    // Compare LevenbergMarquardtOptimizer (interpreter residuals)
    // vs JitLevenbergMarquardtOptimizer (JIT residuals, interpreter JacRev).
    // Uses a population of trees with constants so the optimizer has work to do.
    constexpr size_t n         = 200;
    constexpr size_t maxLength = 30;
    constexpr size_t maxDepth  = 1000;
    constexpr size_t nrow      = 1000;
    constexpr size_t ncol      = 10;
    constexpr size_t lmIters   = 50;

    Operon::RandomGenerator rd(42);
    auto ds = Util::RandomDataset(rd, nrow, ncol);

    auto variables = ds.GetVariables();
    auto target    = variables.back().Name;
    auto inputs    = ds.VariableHashes();
    std::erase(inputs, ds.GetVariable(target).value().Hash);
    Range range = {0, ds.Rows<std::size_t>()};

    Problem problem{&ds};
    problem.SetTrainingRange(range);
    problem.SetTestRange(range);
    problem.SetTarget(target);
    problem.SetLinearScalingEnabled(false);

    Operon::JIT::JitZobrist zobrist(rd, static_cast<int>(maxLength), inputs);

    auto bench_pset = [&](PrimitiveSetConfig cfg, std::string const& title) {
        problem.GetPrimitiveSet().SetConfig(cfg | NodeType::Constant);
        for (auto t : {BuiltinOp::Add, BuiltinOp::Sub, BuiltinOp::Div, BuiltinOp::Mul}) {
            problem.GetPrimitiveSet().SetMinMaxArity(static_cast<Operon::Hash>(t), 2, 2);
        }

        std::uniform_int_distribution<size_t> sizeDistribution(5, maxLength);
        auto creator = BalancedTreeCreator{&problem.GetPrimitiveSet(), inputs, 0.3, maxLength};

        Vector<Tree> trees(n);
        std::ranges::generate(trees, [&]() -> Tree { return creator(rd, sizeDistribution(rd), 0, maxDepth); });
        // skip trees with no constants — nothing for LM to do
        trees.erase(std::remove_if(trees.begin(), trees.end(),
            [](Tree const& t) { return t.CoefficientsCount() == 0; }), trees.end());
        if (trees.empty()) { return; }

        ScalarDispatch dtable;
        JIT::JitEvaluator jitEval{&problem, &zobrist, MSE{}};

        // Pre-warm JIT cache.
        {
            RandomGenerator rng2{1};
            Vector<Individual> inds(trees.size());
            for (size_t i = 0; i < trees.size(); ++i) inds[i].Genotype = trees[i];
            Vector<Scalar> buf(range.Size());
            for (auto& ind : inds) { jitEval(rng2, ind, {buf}); }
        }

        LevenbergMarquardtOptimizer<ScalarDispatch> lmOpt{&dtable, &problem};
        lmOpt.SetIterations(lmIters);

        JitLevenbergMarquardtOptimizer<ScalarDispatch> jitLmOpt{&dtable, &problem, &jitEval};
        jitLmOpt.SetIterations(lmIters);

        nb::Bench b;
        b.title(title)
         .relative(true)
         .performanceCounters(true)
         .epochs(3)
         .minEpochIterations(5);

        b.batch(static_cast<double>(trees.size()))
         .run("lm (interpreter)", [&]() {
             for (auto& tree : trees) {
                 auto summary = lmOpt.Optimize(rd, tree);
                 nb::doNotOptimizeAway(Operon::Diagnostics(summary).FinalCost);
             }
         });

        b.batch(static_cast<double>(trees.size()))
         .run("lm (jit residuals)", [&]() {
             for (auto& tree : trees) {
                 auto summary = jitLmOpt.Optimize(rd, tree);
                 nb::doNotOptimizeAway(Operon::Diagnostics(summary).FinalCost);
             }
         });

        fmt::print("  JIT cache: {} entries, {} hits\n", jitEval.CacheSize(), jitEval.CacheHits());
    };

    SECTION("arithmetic") { bench_pset(PrimitiveSet::Arithmetic, "arithmetic"); }
    SECTION("arithmetic + sin") { bench_pset(PrimitiveSet::Arithmetic | BuiltinOp::Sin, "arithmetic+sin"); }
}
#endif // HAVE_ASMJIT

} // namespace Operon::Test
