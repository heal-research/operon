// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <doctest/doctest.h>
#include <fmt/ranges.h>
#include <taskflow/core/executor.hpp>
#include <taskflow/taskflow.hpp>

#include "operon/algorithms/nsga2.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/operators/non_dominated_sorter.hpp"
#include "operon/formatter/formatter.hpp"


namespace Operon::Test {
    TEST_CASE("poisson regression" ) {
        constexpr auto nrows = 30;
        constexpr auto ncols = 2;

        Operon::RandomGenerator rng{1234};
        using Uniform = std::uniform_real_distribution<Operon::Scalar>;
        using Normal  = std::normal_distribution<Operon::Scalar>;
        using Poisson = std::poisson_distribution<>;

        Eigen::Array<Operon::Scalar, nrows, ncols> data;

        auto x = data.col(0);
        auto y = data.col(1);

        std::generate_n(x.data(), nrows, [&](){ return Uniform(0.1, 5)(rng); });
        Eigen::Array<Operon::Scalar, nrows, 1> lam = 2 * x.square();


        std::transform(lam.begin(), lam.end(), y.data(), [&](auto v) { return Poisson(v)(rng); });

        Operon::Dataset ds{data};
        Operon::Range rg(0, ds.Rows());

        Operon::Problem problem{ds, {0UL, ds.Rows<std::size_t>()}, {0UL, 1UL}};
        problem.ConfigurePrimitiveSet(Operon::PrimitiveSet::Arithmetic);

        // operator parameters
        constexpr auto pc{1.0};
        constexpr auto pm{0.25};

        constexpr auto maxDepth{10UL};
        constexpr auto maxLength{30UL};

        Operon::BalancedTreeCreator creator{problem.GetPrimitiveSet(), problem.GetInputs()};
        auto [minArity, maxArity] = problem.GetPrimitiveSet().FunctionArityLimits();

        Operon::UniformTreeInitializer treeInitializer(creator);
        treeInitializer.ParameterizeDistribution(minArity+1, maxLength);
        treeInitializer.SetMinDepth(1);
        treeInitializer.SetMaxDepth(maxDepth);

        Operon::CoefficientInitializer<Uniform> coeffInitializer;
        coeffInitializer.ParameterizeDistribution(-5.F, +5.F);

        Operon::SubtreeCrossover crossover{pc, maxDepth, maxLength};
        Operon::MultiMutation mutator{};

        Operon::OnePointMutation<Normal> onePoint{};
        onePoint.ParameterizeDistribution(0.F, 1.F);
        Operon::ChangeVariableMutation changeVar{problem.GetInputs()};
        Operon::ChangeFunctionMutation changeFunc { problem.GetPrimitiveSet() };
        Operon::ReplaceSubtreeMutation replaceSubtree { creator, coeffInitializer, maxDepth, maxLength };
        Operon::InsertSubtreeMutation insertSubtree { creator, coeffInitializer, maxDepth, maxLength };
        Operon::RemoveSubtreeMutation removeSubtree { problem.GetPrimitiveSet() };

        mutator.Add(onePoint, 1.0);
        mutator.Add(changeVar, 1.0);
        mutator.Add(changeFunc, 1.0);
        mutator.Add(replaceSubtree, 1.0);
        mutator.Add(insertSubtree, 1.0);
        mutator.Add(removeSubtree, 1.0);

        constexpr auto maxEvaluations{1'000'000};
        constexpr auto maxGenerations{1'000};

        Operon::LengthEvaluator lengthEvaluator{problem, maxLength};

        Operon::DefaultDispatch dt;

        using Likelihood = Operon::PoissonLikelihood<Operon::Scalar, /*LogInput*/ true>;
        Operon::LikelihoodEvaluator<decltype(dt), Likelihood> poissonEvaluator{problem, dt};
        poissonEvaluator.SetBudget(maxEvaluations);

        Operon::MultiEvaluator evaluator{problem};
        evaluator.SetBudget(maxEvaluations);
        evaluator.Add(poissonEvaluator);
        evaluator.Add(lengthEvaluator);

        // Operon::LevenbergMarquardtOptimizer<decltype(dt)> optimizer{dt, problem};
        Operon::SGDOptimizer<decltype(dt), Likelihood> optimizer{dt, problem};
        // Operon::LBFGSOptimizer<decltype(dt), Operon::PoissonLikelihood<>> optimizer{dt, problem};
        optimizer.SetIterations(100);

        Operon::CrowdedComparison cc;
        Operon::TournamentSelector selector{cc};
        Operon::CoefficientOptimizer co{optimizer};

        Operon::BasicOffspringGenerator gen{evaluator, crossover, mutator, selector, selector, &co};
        Operon::RankIntersectSorter rankSorter;
        Operon::KeepBestReinserter reinserter{cc};

        tf::Executor executor;

        Operon::GeneticAlgorithmConfig config{};
        config.Generations = maxGenerations;
        config.Evaluations = maxEvaluations;
        config.PopulationSize = 100;
        config.PoolSize = 100;
        config.CrossoverProbability = pc;
        config.MutationProbability = pm;
        config.Seed = 1234;
        config.TimeLimit = std::numeric_limits<size_t>::max();

        Operon::NSGA2 algorithm{problem, config, treeInitializer, coeffInitializer, gen, reinserter, rankSorter};
        auto report = [&](){
            fmt::print("{} {}\n", algorithm.Generation(), std::size_t{poissonEvaluator.CallCount});
        };

        algorithm.Run(executor, rng, report);
        fmt::print("{}\n", poissonEvaluator.TotalEvaluations());

        // validate results
        Operon::AkaikeInformationCriterionEvaluator<decltype(dt)> aicEvaluator{problem, dt};

        for (auto ind : algorithm.Best()) {
            auto a = poissonEvaluator(rng, ind, {});

            Operon::Interpreter<Operon::Scalar, decltype(dt)> interpreter(dt, ds, ind.Genotype);
            auto est = interpreter.Evaluate(ind.Genotype.GetCoefficients(), problem.TrainingRange());
            auto tgt = problem.TargetValues(problem.TrainingRange());

            auto b = Likelihood::ComputeLikelihood(est, tgt, poissonEvaluator.Sigma());
            auto c = aicEvaluator(rng, ind, {});

            fmt::print("{}: {} {} {}\n", Operon::InfixFormatter::Format(ind.Genotype, ds), a, b, c);
        }
    }
} // namespace Operon::Test