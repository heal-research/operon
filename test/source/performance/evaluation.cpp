// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <doctest/doctest.h>
#include <thread>
#include <taskflow/taskflow.hpp>
#include <taskflow/algorithm/for_each.hpp>   // for taskflow.for_each_index
#include <utility>
#include <vstat/vstat.hpp>

#if TF_MINOR_VERSION > 2
#include <taskflow/algorithm/reduce.hpp>
#endif

#include "../operon_test.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/nsga2.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"
#include "operon/interpreter/interpreter.hpp"

namespace Operon::Test {
    auto TotalNodes(const std::vector<Tree>& trees) {
        return vstat::univariate::accumulate<Operon::Scalar>(trees.begin(), trees.end(), [](auto const& t) { return t.Length(); }).sum;
    }

    namespace nb = ankerl::nanobench;

    template<typename T, std::align_val_t A = std::align_val_t{__STDCPP_DEFAULT_NEW_ALIGNMENT__}>
    inline auto MakeUnique(std::size_t n, std::optional<typename std::remove_extent_t<T>> init = std::nullopt)
    {
        using E = typename std::remove_extent_t<T>;
        using Ptr = std::unique_ptr<T, std::add_pointer_t<void(E*)>>;
        auto ptr = Ptr(static_cast<E*>(::operator new[](n * sizeof(E), A)), [](E* ptr){ ::operator delete[](ptr, A); });
        if (init) { std::fill_n(ptr.get(), n, init.value()); }
        return ptr;
    }

    template <typename T, typename DTable>
    void Evaluate(tf::Executor& executor, DTable const& dt, std::vector<Tree> const& trees, Dataset const& ds, Range range)
    {
        tf::Taskflow taskflow;
        std::vector<std::vector<T>> results(executor.num_workers());
        for (auto& res: results) { res.resize(range.Size()); }
        taskflow.for_each_index(size_t{0}, trees.size(), size_t{1}, [&](auto i) {
            auto& res = results[executor.this_worker_id()];
            auto coeff = trees[i].GetCoefficients();
            Operon::Interpreter<T, DTable>{dt, ds, trees[i]}.Evaluate({coeff}, range, {res});
        });
        executor.run(taskflow).wait();
    }


    // used by some Langdon & Banzhaf papers as benchmark for measuring GPops/s
    TEST_CASE("Evaluation performance")
    {
        constexpr size_t n = 1000;
        constexpr size_t maxLength = 100;
        constexpr size_t maxDepth = 1000;
        constexpr size_t nrow = 10000;
        constexpr size_t ncol = 10;

        constexpr size_t minEpochIterations = 5;
        Operon::RandomGenerator rd(1234);
        auto ds = Util::RandomDataset(rd, nrow, ncol);
        fmt::print("dataset rows: {}, cols: {}\n", nrow, ncol);
        auto target = "Y";
        auto variables = ds.GetVariables();
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable(target)->Hash);

        Range range = { 0, nrow };

        PrimitiveSet pset;

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { pset, inputs };

        std::vector<Tree> trees(n);

        using DTable = DispatchTable<Operon::Scalar, Operon::Seq<std::size_t, Dispatch::DefaultBatchSize<Operon::Scalar>>>;
        DTable dtable;

        auto test = [&](tf::Executor& executor, nb::Bench& b, PrimitiveSetConfig cfg, const std::string& name) {
            pset.SetConfig(cfg);
            for (auto t : { NodeType::Add, NodeType::Sub, NodeType::Div, NodeType::Mul }) {
                pset.SetMinMaxArity(Node(t).HashValue, 2, 2);
            }
            std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

            auto totalOps = TotalNodes(trees) * range.Size();
            b.batch(totalOps);
            b.run(name, [&]() { Evaluate<Operon::Scalar>(executor, dtable, trees, ds, range); });
        };

        nb::Bench b;
        // b.output(nullptr);

        auto const max_concurrency = std::thread::hardware_concurrency();

        SUBCASE("arithmetic") {
            // single-thread
            b.title("arithmetic").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= max_concurrency; ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + exp") {
            b.title("arithmetic + exp").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= max_concurrency; ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Exp, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + log") {
            b.title("arithmetic + log").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= max_concurrency; ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Log, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + sin") {
            nb::Bench b;
            b.title("arithmetic + sin").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= max_concurrency; ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sin, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + cos") {
            b.title("arithmetic + cos").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= max_concurrency; ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cos, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + tan") {
            b.title("arithmetic + tan").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= max_concurrency; ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Tan, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + sqrt") {
            b.title("arithmetic + sqrt").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= max_concurrency; ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Sqrt, fmt::format("N = {}", i));
            }
        }

        SUBCASE("arithmetic + cbrt") {
            b.title("arithmetic + cbrt").relative(true).performanceCounters(true).minEpochIterations(minEpochIterations);
            for (size_t i = 1; i <= max_concurrency; ++i) {
                tf::Executor executor(i);
                test(executor, b, PrimitiveSet::Arithmetic | NodeType::Cbrt, fmt::format("N = {}", i));
            }
        }

        // b.render(nb::templates::csv(), std::cout);
    }

    TEST_CASE("Evaluator performance")
    {
        const size_t n         = 1000;
        const size_t maxLength = 100;
        const size_t maxDepth  = 1000;

        constexpr size_t nrow = 10000;
        constexpr size_t ncol = 10;

        Operon::RandomGenerator rd(1234);
        auto ds = Util::RandomDataset(rd, nrow, ncol);

        auto variables = ds.GetVariables();
        auto target = variables.back().Name;
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable(target)->Hash);
        Range range = { 0, ds.Rows<std::size_t>() };

        Operon::Problem problem(ds, range, range);
        problem.GetPrimitiveSet().SetConfig(Operon::PrimitiveSet::Arithmetic);

        std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
        auto creator = BalancedTreeCreator { problem.GetPrimitiveSet(), inputs };

        std::vector<Tree> trees(n);
        std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

        std::vector<Individual> individuals(n);
        for (size_t i = 0; i < individuals.size(); ++i) {
            individuals[i].Genotype = trees[i];
        }

        nb::Bench b;
        b.title("Evaluator performance").relative(true).performanceCounters(true).minEpochIterations(10);

        auto totalNodes = TotalNodes(trees);

        Operon::Vector<Operon::Scalar> buf(range.Size());

        auto test = [&](std::string const& name, EvaluatorBase&& evaluator) {
            evaluator.SetBudget(std::numeric_limits<size_t>::max());
            tf::Executor executor(std::thread::hardware_concurrency());
            tf::Taskflow taskflow;

            std::vector<Operon::Vector<Operon::Scalar>> slots(executor.num_workers());
            double sum{0};
            taskflow.transform_reduce(individuals.begin(), individuals.end(), sum, std::plus<>{}, [&](Operon::Individual& ind) {
                auto id = executor.this_worker_id();
                if (slots[id].size() < range.Size()) { slots[id].resize(range.Size()); }
                return evaluator(rd, ind, slots[id]).front();
            });

            b.batch(totalNodes * range.Size()).epochs(10).epochIterations(100).run(name, [&]() {
                sum = 0;
                executor.run(taskflow).wait();
                return sum;
            });
        };


        using DTable = DefaultDispatch;
        DTable dtable;

        test("c2",        Operon::Evaluator(problem, dtable, Operon::C2{}, /*linearScaling=*/false));
        test("c2 + ls",   Operon::Evaluator(problem, dtable, Operon::C2{}, /*linearScaling=*/true));
        test("r2",        Operon::Evaluator(problem, dtable, Operon::R2{}, /*linearScaling=*/false));
        test("r2 + ls",   Operon::Evaluator(problem, dtable, Operon::R2{}, /*linearScaling=*/true));
        test("nmse",      Operon::Evaluator(problem, dtable, Operon::NMSE{}, /*linearScaling=*/false));
        test("nmse + ls", Operon::Evaluator(problem, dtable, Operon::NMSE{}, /*linearScaling=*/true));
        test("mae",       Operon::Evaluator(problem, dtable, Operon::MAE{}, /*linearScaling=*/false));
        test("mae + ls",  Operon::Evaluator(problem, dtable, Operon::MAE{}, /*linearScaling=*/true));
        test("mse",       Operon::Evaluator(problem, dtable, Operon::MSE{}, /*linearScaling=*/false));
        test("mse + ls",  Operon::Evaluator(problem, dtable, Operon::MSE{}, /*linearScaling=*/true));
    }

    TEST_CASE("Parallel interpreter")
    {
       const size_t n         = 1000;
       const size_t maxLength = 100;
       const size_t maxDepth  = 1000;

       constexpr size_t nrow = 10000;
       constexpr size_t ncol = 10;

       Operon::RandomGenerator rd(1234);
       auto ds = Util::RandomDataset(rd, nrow, ncol);

       auto variables = ds.GetVariables();
       auto target = variables.back().Name;
       auto inputs = ds.VariableHashes();
       std::erase(inputs, ds.GetVariable(target)->Hash);
       Range range = { 0, ds.Rows<std::size_t>() };

       std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);
       Operon::PrimitiveSet pset;
       pset.SetConfig(Operon::PrimitiveSet::Arithmetic);
       auto creator = BalancedTreeCreator { pset, inputs };

       std::vector<Tree> trees(n);
       std::generate(trees.begin(), trees.end(), [&]() { return creator(rd, sizeDistribution(rd), 0, maxDepth); });

       nb::Bench b;
       b.relative(true).epochs(10).minEpochIterations(100).performanceCounters(true);
       std::vector<size_t> threads(std::thread::hardware_concurrency());
       std::iota(threads.begin(), threads.end(), 1);
       std::vector<Operon::Scalar> result(trees.size() * range.Size());
       for (auto t : threads) {
           b.batch(TotalNodes(trees) * range.Size()).run(fmt::format("{} thread(s)", t), [&]() { return Operon::EvaluateTrees(trees, ds, range, result, t); });
       }
    }

    TEST_CASE("NSGA2")
    {
        auto ds = Dataset("./data/Friedman-I.csv", /*hasHeader=*/true);

        std::vector<Variable> inputs;
        const auto *targetName = "Y";
        auto variables = ds.GetVariables();
        std::copy_if(variables.begin(),
                variables.end(),
                std::back_inserter(inputs),
                [&](auto const& var) { return var.Name != targetName; });
        auto result = ds.GetVariable(targetName);
        ENSURE(result);
        auto target = result.value();
        auto const nrow{ds.Rows<std::size_t>()};

        Range trainingRange(0, nrow / 2);
        Range testRange(nrow / 2, nrow);

        Operon::Problem problem(ds, trainingRange, testRange);
        problem.GetPrimitiveSet().SetConfig(PrimitiveSet::Arithmetic);
        BalancedTreeCreator creator(problem.GetPrimitiveSet(), problem.GetInputs(), 0.0);

        const size_t maxDepth = 1000;
        const size_t maxLength = 50;
        auto [amin, amax] = problem.GetPrimitiveSet().FunctionArityLimits();
        UniformTreeInitializer initializer(creator);
        initializer.ParameterizeDistribution(amin + 1, maxLength);
        initializer.SetMinDepth(1);
        initializer.SetMaxDepth(maxDepth);

        UniformCoefficientInitializer coeffInit;
        initializer.ParameterizeDistribution(0UL, 1UL);

        const double crossoverInternalProbability = 0.9;

        auto crossover = SubtreeCrossover { crossoverInternalProbability, maxDepth, maxLength };
        auto mutator = MultiMutation {};
        auto onePoint = OnePointMutation<std::uniform_real_distribution<Operon::Scalar>> {};
        onePoint.ParameterizeDistribution(Operon::Scalar{-2}, Operon::Scalar{+2});
        auto changeVar = ChangeVariableMutation { problem.GetInputs() };
        auto changeFunc = ChangeFunctionMutation { problem.GetPrimitiveSet() };
        auto replaceSubtree = ReplaceSubtreeMutation { creator, coeffInit, maxDepth, maxLength };
        auto insertSubtree = InsertSubtreeMutation { creator, coeffInit, maxDepth, maxLength };
        auto removeSubtree = RemoveSubtreeMutation { problem.GetPrimitiveSet() };
        mutator.Add(onePoint, 1.0);
        mutator.Add(changeVar, 1.0);
        mutator.Add(changeFunc, 1.0);
        mutator.Add(removeSubtree, 1.0);
        mutator.Add(insertSubtree, 1.0);
        mutator.Add(removeSubtree, 1.0);

        using DTable = DefaultDispatch;
        DTable dtable;

        Evaluator c2eval(problem, dtable, Operon::C2{}, /*linearScaling=*/false);
        LengthEvaluator lenEval(problem);

        MultiEvaluator evaluator(problem);
        evaluator.Add(c2eval);
        evaluator.Add(lenEval);

        CrowdedComparison comp;
        TournamentSelector selector(comp);
        KeepBestReinserter reinserter(comp);

        BasicOffspringGenerator generator(evaluator, crossover, mutator, selector, selector);
        Operon::RandomGenerator random(1234);

        RankIntersectSorter sorter;

        GeneticAlgorithmConfig config{};
        config.Generations = 100;
        config.PopulationSize = 1000;
        config.PoolSize = 1000;
        config.Evaluations = 1000000;
        config.Iterations = 0;
        config.CrossoverProbability = 1.0;
        config.MutationProbability = 0.25;
        config.TimeLimit = ~size_t{0};
        config.Seed = random();

        Operon::NSGA2 gp { problem, config, initializer, coeffInit, generator, reinserter, sorter };
        //tf::Executor executor;

        auto report = [&]() {
            fmt::print("generation {}\n", gp.Generation());
        };

        gp.Run(random, report, 10);
    }

    TEST_CASE("math cost model")
    {
        const size_t n         = 1000;
        const size_t maxLength = 3;
        const size_t maxDepth  = 2;

        constexpr size_t nrow = 10000;
        constexpr size_t ncol = 10;

        Operon::RandomGenerator rd(1234);
        auto ds = Util::RandomDataset(rd, nrow, ncol);

        auto variables = ds.GetVariables();
        auto target = variables.back().Name;
        auto inputs = ds.VariableHashes();
        std::erase(inputs, ds.GetVariable(target)->Hash);
        Range range = { 0, ds.Rows<std::size_t>() };
        using DTable = DefaultDispatch;
        DTable dtable;

        //Operon::PrimitiveSet base(Operon::PrimitiveSet::Arithmetic);
        auto primitives = NodeType::Constant;
        nb::Bench b;
        for (auto i = 0UL; i < NodeTypes::Count-3; ++i) {
            Operon::PrimitiveSet pset(primitives | static_cast<NodeType>(1U << i));
            Operon::Node node(static_cast<NodeType>(1U << i));
            auto creator = BalancedTreeCreator { pset, inputs };
            std::vector<Tree> trees(n);
            std::uniform_int_distribution<size_t> sizeDistribution(1, maxLength);

            b.batch(range.Size()).epochs(10).epochIterations(1000).run(node.Name(), [&]() {
                auto tree = creator(rd, sizeDistribution(rd), 0, maxDepth);
                return Interpreter<Operon::Scalar, DTable>::Evaluate(tree, ds, range);
            });
        }
    }
} // namespace Operon::Test
