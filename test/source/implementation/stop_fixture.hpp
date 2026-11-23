// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_TEST_STOP_FIXTURE_HPP
#define OPERON_TEST_STOP_FIXTURE_HPP

#include <cstddef>
#include <memory>
#include <numeric>
#include <random>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/creator.hpp"
#include "operon/operators/crossover.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/operators/generator.hpp"
#include "operon/operators/initializer.hpp"
#include "operon/operators/mutation.hpp"
#include "operon/operators/reinserter.hpp"
#include "operon/operators/selector.hpp"

namespace Operon::Test {

using DTable = DispatchTable<Operon::Scalar>;

// A tiny GP setup (10 rows, y = 2x, 12 + 12 individuals) whose stop conditions the tests set one by one.
inline auto MakeDataset() -> Operon::Dataset
{
    std::vector<std::vector<Operon::Scalar>> data(2, std::vector<Operon::Scalar>(10));
    std::iota(data[0].begin(), data[0].end(), Operon::Scalar { 1 });
    for (std::size_t i = 0; i < 10; ++i) {
        data[1][i] = data[0][i] * Operon::Scalar { 2 };
    }
    return Operon::Dataset({ "X1", "Y" }, data);
}

struct StopFixture {
    static constexpr std::size_t PopSize = 12;
    static constexpr std::size_t PoolSize = 12;

    Operon::Dataset Ds { MakeDataset() };
    Operon::Problem Problem { gsl::not_null<Operon::Dataset*>(&Ds) };
    Operon::PrimitiveSet Pset;
    std::vector<Operon::Hash> Vars;
    DTable Dtable;
    Operon::Evaluator<DTable> Evaluator { &Problem, &Dtable, Operon::MSE {} };
    Operon::SubtreeCrossover Crossover { 0.9, /*maxDepth=*/6, /*maxLength=*/20 };
    Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> OnePoint;
    Operon::MultiMutation Mutator;
    Operon::TournamentSelector FemSel { Operon::SingleObjectiveComparison { 0 } };
    Operon::TournamentSelector MaleSel { Operon::SingleObjectiveComparison { 0 } };
    Operon::BasicOffspringGenerator Basic { &Evaluator, &Crossover, &Mutator, &FemSel, &MaleSel };
    Operon::OffspringSelectionGenerator OffspringSelection { &Evaluator, &Crossover, &Mutator, &FemSel, &MaleSel };
    Operon::KeepBestReinserter Reinserter { Operon::SingleObjectiveComparison { 0 } };
    Operon::BalancedTreeCreator Creator;
    Operon::UniformTreeInitializer TreeInit { &Creator };
    Operon::UniformCoefficientInitializer CoeffInit;
    Operon::GeneticAlgorithmConfig Config;

    StopFixture()
        : Pset([] {
            Operon::PrimitiveSet p;
            p.SetConfig(PrimitiveSet::Arithmetic);
            return p;
        }())
        , Vars { Ds.GetVariable("X1")->Hash }
        , Creator(&Pset, Vars, 0.0, 10)
    {
        Problem.SetTrainingRange({ 0, 10 });
        Problem.SetTestRange({ 0, 10 });
        Problem.SetTarget("Y");
        TreeInit.ParameterizeDistribution(std::size_t { 1 }, std::size_t { 10 });
        OnePoint.ParameterizeDistribution(Operon::Scalar { 0 }, Operon::Scalar { 1 });
        Mutator.Add(&OnePoint, 1.0);
        Config.PopulationSize = PopSize;
        Config.PoolSize = PoolSize;
        Config.Generations = 1000;
        Config.LocalSearchProbability = 0.0;
        Config.Evaluations = std::size_t { 1 } << 40U;
    }

    [[nodiscard]] auto MakeGp(Operon::OffspringGeneratorBase const* generator = nullptr)
        -> std::unique_ptr<Operon::GeneticProgrammingAlgorithm>
    {
        return std::make_unique<Operon::GeneticProgrammingAlgorithm>(
            Config, &Problem, &TreeInit, &CoeffInit, generator != nullptr ? generator : &Basic, &Reinserter);
    }
};

} // namespace Operon::Test

#endif
