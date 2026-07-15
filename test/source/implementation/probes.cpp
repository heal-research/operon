// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/probes/chain.hpp"
#include "operon/algorithms/probes/probe.hpp"
#include "operon/algorithms/probes/registry.hpp"
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

// Trimmed down from ga_base.cpp's GaBaseFixture: just enough wiring to get a
// GeneticAlgorithmBase-derived object whose Parents()/Offspring() can be
// populated via RestoreIndividuals(), which is all ProbeContext reads.
struct ProbeFixture {
    static constexpr std::size_t PopSize  = 6;
    static constexpr std::size_t PoolSize = 4;

    Operon::Dataset Ds{ MakeDataset() };
    Operon::Problem Problem{ gsl::not_null<Operon::Dataset*>(&Ds) };
    Operon::PrimitiveSet Pset{ MakePset() };
    std::vector<Operon::Hash> Vars{ Ds.GetVariable("X1")->Hash };

    DTable Dtable;
    Operon::Evaluator<DTable>             Evaluator{ &Problem, &Dtable };
    Operon::SubtreeCrossover              Crossover{ 0.9, /*maxDepth=*/6, /*maxLength=*/20 };
    Operon::OnePointMutation<std::normal_distribution<Operon::Scalar>> OnePoint;
    Operon::MultiMutation                 Mutator;
    Operon::TournamentSelector            FemSel{ Operon::SingleObjectiveComparison{0} };
    Operon::TournamentSelector            MaleSel{ Operon::SingleObjectiveComparison{0} };
    Operon::BasicOffspringGenerator       Generator{ &Evaluator, &Crossover, &Mutator, &FemSel, &MaleSel };
    Operon::KeepBestReinserter            Reinserter{ Operon::SingleObjectiveComparison{0} };
    Operon::BalancedTreeCreator           Creator{ &Pset, Vars, 0.0, 10 };
    Operon::UniformTreeInitializer        TreeInit{ &Creator };
    Operon::UniformCoefficientInitializer CoeffInit;
    Operon::GeneticAlgorithmConfig        Config{ [] -> Operon::GeneticAlgorithmConfig { Operon::GeneticAlgorithmConfig c; c.PopulationSize = PopSize; c.PoolSize = PoolSize; c.Generations = 1; return c; }() };
    Operon::GeneticProgrammingAlgorithm   Gp{ Config, &Problem, &TreeInit, &CoeffInit, &Generator, &Reinserter };

    ProbeFixture()
    {
        Problem.SetTrainingRange({0, 10});
        Problem.SetTestRange({0, 10});
        Problem.SetTarget("Y");

        std::vector<Operon::Individual> inds(PopSize + PoolSize);
        for (std::size_t i = 0; i < inds.size(); ++i) {
            inds[i].Fitness.resize(1);
            inds[i].Fitness[0] = static_cast<Operon::Scalar>(i);
        }
        Gp.RestoreIndividuals(inds);
    }
};

// Records every generation it was invoked on, and emits a scalar + a vector
// so both ResultValue shapes get exercised through ProbeContext::Emit.
struct RecordingProbe final : Operon::GenerationProbe {
    std::vector<std::size_t> Seen;
    bool Finished{false};

    auto operator()(Operon::ProbeContext& ctx) -> void override
    {
        Seen.push_back(ctx.Generation());
        ctx.Emit("seen_count", static_cast<std::int64_t>(Seen.size()));
        ctx.Emit("readings", std::vector<double>{1.0, 2.0, 3.0});
    }

    auto Finish() -> void override { Finished = true; }
};

// Bumps Generation() on the fixture's Gp directly (no public setter exists;
// GeneticAlgorithmBase::Generation() is a non-const-returning reference for
// mutable callers, same as production code uses via ++Generation() in
// source/algorithms/gp.cpp).
auto AdvanceTo(Operon::GeneticProgrammingAlgorithm& gp, std::size_t generation) -> void
{
    gp.Generation() = generation;
}

} // namespace

TEST_CASE("ProbeContext exposes the wrapped algorithm's read surface", "[probes]")
{
    ProbeFixture f;
    Operon::ResultRecord record;
    Operon::ProbeContext ctx{f.Gp, record};

    CHECK(ctx.Generation() == 0);
    CHECK(ctx.Parents().size() == ProbeFixture::PopSize);
    CHECK(ctx.Offspring().size() == ProbeFixture::PoolSize);
    CHECK(ctx.Problem() == &f.Problem);

    ctx.Emit("x", std::int64_t{42});
    REQUIRE(record.contains("x"));
    CHECK(std::get<std::int64_t>(record.at("x")) == 42);

    // Emit overwrites, it doesn't accumulate.
    ctx.Emit("x", std::int64_t{7});
    CHECK(record.size() == 1);
    CHECK(std::get<std::int64_t>(record.at("x")) == 7);
}

TEST_CASE("ProbeChain runs a probe every generation by default", "[probes]")
{
    ProbeFixture f;
    Operon::ProbeChain chain;
    auto owned = std::make_unique<RecordingProbe>();
    auto* probe = owned.get();
    chain.Add(std::move(owned));

    for (std::size_t g = 0; g < 4; ++g) {
        AdvanceTo(f.Gp, g);
        chain(f.Gp);
    }

    REQUIRE(probe->Seen == std::vector<std::size_t>{0, 1, 2, 3});

    chain.Finish();
    CHECK(probe->Finished);
}

TEST_CASE("ProbeChain respects every/offset scheduling", "[probes]")
{
    ProbeFixture f;
    Operon::ProbeChain chain;
    auto owned = std::make_unique<RecordingProbe>();
    auto* probe = owned.get();
    // every=2, offset=1 -> fires on generations 1, 3, 5, ...
    chain.Add(std::move(owned), /*every=*/2, /*offset=*/1);

    for (std::size_t g = 0; g < 6; ++g) {
        AdvanceTo(f.Gp, g);
        chain(f.Gp);
    }

    CHECK(probe->Seen == std::vector<std::size_t>{1, 3, 5});
}

TEST_CASE("ProbeChain every=0 disables a probe without removing it", "[probes]")
{
    ProbeFixture f;
    Operon::ProbeChain chain;
    auto owned = std::make_unique<RecordingProbe>();
    auto* probe = owned.get();
    chain.Add(std::move(owned), /*every=*/0);

    for (std::size_t g = 0; g < 3; ++g) {
        AdvanceTo(f.Gp, g);
        chain(f.Gp);
    }

    CHECK(probe->Seen.empty());
}

TEST_CASE("JsonlSink writes one line per generation that ran a probe, none otherwise", "[probes]")
{
    ProbeFixture f;
    auto const path = std::filesystem::temp_directory_path() / "operon_probes_test.jsonl";
    std::filesystem::remove(path);

    {
        Operon::ProbeChain chain;
        chain.Add(std::make_unique<RecordingProbe>(), /*every=*/2); // fires on generation 0, 2
        chain.SetSink(std::make_unique<Operon::JsonlSink>(path.string()));

        for (std::size_t g = 0; g < 4; ++g) {
            AdvanceTo(f.Gp, g);
            chain(f.Gp);
        }
        chain.Finish();
    }

    REQUIRE(std::filesystem::exists(path));
    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) { lines.push_back(line); }
    }
    in.close();
    std::filesystem::remove(path);

    // Only generations 0 and 2 ran the probe, so only two lines should exist,
    // each a self-describing JSON object carrying the emitted keys.
    REQUIRE(lines.size() == 2);
    for (auto const& line : lines) {
        CHECK(line.front() == '{');
        CHECK(line.back() == '}');
        CHECK(line.find("\"generation\"") != std::string::npos);
        CHECK(line.find("\"seen_count\"") != std::string::npos);
        CHECK(line.find("\"readings\"") != std::string::npos);
    }
}

TEST_CASE("ProbeRegistry creates probes by registered type name", "[probes]")
{
    Operon::ProbeRegistry registry;
    CHECK_FALSE(registry.Contains("recording"));

    registry.Register("recording", [](Operon::ProbeParams const& /*params*/) -> std::unique_ptr<Operon::GenerationProbe> {
        return std::make_unique<RecordingProbe>();
    });
    CHECK(registry.Contains("recording"));

    auto probe = registry.Create("recording", {});
    REQUIRE(probe != nullptr);

    auto missing = registry.Create("does-not-exist", {});
    CHECK(missing == nullptr);
}

TEST_CASE("ProbeRegistry factory receives its params", "[probes]")
{
    Operon::ProbeRegistry registry;
    registry.Register("check-params", [](Operon::ProbeParams const& params) -> std::unique_ptr<Operon::GenerationProbe> {
        REQUIRE(params.contains("path"));
        CHECK(std::get<std::string>(params.at("path")) == "out.beve");
        return std::make_unique<RecordingProbe>();
    });

    Operon::ProbeParams params;
    params.insert_or_assign("path", Operon::ProbeParamValue{std::string{"out.beve"}});
    auto probe = registry.Create("check-params", params);
    CHECK(probe != nullptr);
}

} // namespace Operon::Test
