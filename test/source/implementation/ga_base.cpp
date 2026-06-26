// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <numeric>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
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
namespace {

using DTable = DispatchTable<Operon::Scalar>;

auto MakeDataset() -> Operon::Dataset
{
    std::vector<std::vector<Operon::Scalar>> data(2, std::vector<Operon::Scalar>(10));
    std::iota(data[0].begin(), data[0].end(), Operon::Scalar{1});
    for (std::size_t i = 0; i < 10; ++i) { data[1][i] = data[0][i] * Operon::Scalar{2}; }
    return Operon::Dataset({"X1", "Y"}, data);
}

auto MakePset() -> Operon::PrimitiveSet
{
    Operon::PrimitiveSet p;
    p.SetConfig(PrimitiveSet::Arithmetic);
    return p;
}

auto MakeConfig(std::size_t popSize, std::size_t poolSize) -> Operon::GeneticAlgorithmConfig
{
    Operon::GeneticAlgorithmConfig c;
    c.PopulationSize = popSize;
    c.PoolSize       = poolSize;
    c.Generations    = 1;
    return c;
}

struct GaBaseFixture {
    static constexpr std::size_t PopSize  = 6;
    static constexpr std::size_t PoolSize = 4;

    Operon::Dataset Ds{ MakeDataset() };
    Operon::Problem Problem{ gsl::not_null<Operon::Dataset*>(&Ds) };
    Operon::PrimitiveSet Pset{ MakePset() };
    std::vector<Operon::Hash> Vars{ Ds.GetVariable("X1")->Hash };

    DTable Dtable;
    Operon::Evaluator<DTable>             Evaluator{ &Problem, &Dtable };
    Operon::SubtreeCrossover              Crossover{ 0.9, /*maxDepth=*/6, /*maxLength=*/20 };
    Operon::MultiMutation                 Mutator;
    Operon::TournamentSelector            FemSel{ Operon::SingleObjectiveComparison{0} };
    Operon::TournamentSelector            MaleSel{ Operon::SingleObjectiveComparison{0} };
    Operon::BasicOffspringGenerator       Generator{ &Evaluator, &Crossover, &Mutator, &FemSel, &MaleSel };
    Operon::KeepBestReinserter            Reinserter{ Operon::SingleObjectiveComparison{0} };
    Operon::BalancedTreeCreator           Creator{ &Pset, Vars, 0.0, 10 };
    Operon::UniformTreeInitializer        TreeInit{ &Creator };
    Operon::UniformCoefficientInitializer CoeffInit;
    Operon::GeneticAlgorithmConfig        Config{ MakeConfig(PopSize, PoolSize) };
    Operon::GeneticProgrammingAlgorithm   Gp{ Config, &Problem, &TreeInit, &CoeffInit, &Generator, &Reinserter };

    GaBaseFixture()
    {
        Problem.SetTrainingRange({0, 10});
        Problem.SetTestRange({0, 10});
        Problem.SetTarget("Y");
    }
};

} // namespace

TEST_CASE("RestoreIndividuals maps parents and offspring spans correctly", "[algorithms]")
{
    GaBaseFixture f;
    constexpr auto popSize  = GaBaseFixture::PopSize;
    constexpr auto poolSize = GaBaseFixture::PoolSize;
    constexpr auto total    = popSize + poolSize;

    // Give each individual a distinct fitness so we can verify which span they land in.
    std::vector<Operon::Individual> inds(total);
    for (std::size_t i = 0; i < total; ++i) {
        inds[i].Fitness.resize(1);
        inds[i].Fitness[0] = static_cast<Operon::Scalar>(i);
    }

    f.Gp.RestoreIndividuals(inds);

    auto parents   = f.Gp.Parents();
    auto offspring = f.Gp.Offspring();

    REQUIRE(parents.size()   == popSize);
    REQUIRE(offspring.size() == poolSize);

    for (std::size_t i = 0; i < popSize; ++i) {
        CHECK(parents[i].Fitness[0] == static_cast<Operon::Scalar>(i));
    }
    for (std::size_t i = 0; i < poolSize; ++i) {
        CHECK(offspring[i].Fitness[0] == static_cast<Operon::Scalar>(popSize + i));
    }
}

} // namespace Operon::Test
