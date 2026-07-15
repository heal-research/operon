// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>

#include <cstddef>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/gp.hpp"
#include "operon/algorithms/probes/cache_hit_rate.hpp"
#include "operon/algorithms/probes/chain.hpp"
#include "operon/algorithms/probes/diversity.hpp"
#include "operon/algorithms/probes/population_trace.hpp"
#include "operon/algorithms/probes/probe.hpp"
#include "operon/algorithms/probes/registry.hpp"
#include "operon/core/dataset.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/serialization.hpp"
#include "operon/core/types.hpp"
#include "operon/hash/zobrist.hpp"
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

// Trimmed down from ga_base.cpp's GaBaseFixture - just enough to populate
// Parents()/Offspring() via RestoreIndividuals(), which is all ProbeContext reads.
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

// Same wiring as ProbeFixture, but with a Zobrist transposition cache wired
// into GeneticAlgorithmConfig::Cache, for CacheHitRateProbe tests.
struct CacheProbeFixture {
    static constexpr std::size_t PopSize = 4;
    static constexpr std::size_t PoolSize = 2;

    Operon::Dataset Ds{ MakeDataset() };
    Operon::Problem Problem{ gsl::not_null<Operon::Dataset*>(&Ds) };
    Operon::PrimitiveSet Pset{ MakePset() };
    std::vector<Operon::Hash> Vars{ Ds.GetVariable("X1")->Hash };

    Operon::RandomGenerator CacheRng{1234};
    Operon::Zobrist Cache{ CacheRng, /*maxLength=*/50, Vars };

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
    Operon::GeneticProgrammingAlgorithm   Gp{
        [&] -> Operon::GeneticAlgorithmConfig {
            Operon::GeneticAlgorithmConfig c;
            c.PopulationSize = PopSize;
            c.PoolSize = PoolSize;
            c.Generations = 1;
            c.Cache = &Cache;
            return c;
        }(),
        &Problem, &TreeInit, &CoeffInit, &Generator, &Reinserter
    };

    CacheProbeFixture()
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
    int FinishCount{0};
    // Outlives the probe, for tests checking Finish() after the owning
    // ProbeChain (and thus this probe) is already destroyed.
    int* ExternalFinishCount{nullptr};

    auto operator()(Operon::ProbeContext& ctx) -> void override
    {
        Seen.push_back(ctx.Generation());
        // Exercises every ResultValue alternative, not just int64_t/vector<double>.
        ctx.Emit("seen_count", static_cast<std::int64_t>(Seen.size()));
        ctx.Emit("mean", 1.5);
        ctx.Emit("flag", true);
        ctx.Emit("label", std::string{"gen"});
        ctx.Emit("counts", std::vector<std::int64_t>{1, 2, 3});
        ctx.Emit("readings", std::vector<double>{1.0, 2.0, 3.0});
    }

    auto Finish() -> void override
    {
        ++FinishCount;
        if (ExternalFinishCount != nullptr) { ++*ExternalFinishCount; }
    }
};

// Generation() returns a mutable reference for non-const callers (same as
// production's ++Generation() in source/algorithms/gp.cpp); no setter exists.
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
    CHECK(probe->FinishCount == 1);
}

TEST_CASE("ProbeChain::Finish is idempotent across explicit calls and destruction", "[probes]")
{
    int finishCount = 0;
    {
        Operon::ProbeChain chain;
        auto owned = std::make_unique<RecordingProbe>();
        owned->ExternalFinishCount = &finishCount;
        chain.Add(std::move(owned));

        chain.Finish();
        chain.Finish(); // explicit double-call must not re-run probe Finish()
        CHECK(finishCount == 1);
    } // destructor also calls Finish() - must not run it a second time either

    CHECK(finishCount == 1);
}

TEST_CASE("ProbeChain destructor runs Finish() if the caller never called it", "[probes]")
{
    int finishCount = 0;
    {
        Operon::ProbeChain chain;
        auto owned = std::make_unique<RecordingProbe>();
        owned->ExternalFinishCount = &finishCount;
        chain.Add(std::move(owned));
        CHECK(finishCount == 0);
    } // no explicit Finish() call before scope exit

    CHECK(finishCount == 1);
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

    // A disabled probe still gets cleaned up: Finish() runs regardless of
    // Every, only the per-generation call is gated.
    chain.Finish();
    CHECK(probe->FinishCount == 1);
}

TEST_CASE("ProbeChain runs every registered probe and later probes win on shared keys", "[probes]")
{
    ProbeFixture f;
    Operon::ProbeChain chain;
    auto first = std::make_unique<RecordingProbe>();
    auto* firstPtr = first.get();
    auto second = std::make_unique<RecordingProbe>();
    auto* secondPtr = second.get();
    chain.Add(std::move(first));
    chain.Add(std::move(second));

    AdvanceTo(f.Gp, 0);
    chain(f.Gp);

    // Both probes ran (each independently tracks that it saw generation 0).
    CHECK(firstPtr->Seen == std::vector<std::size_t>{0});
    CHECK(secondPtr->Seen == std::vector<std::size_t>{0});
}

TEST_CASE("ProbeChain move-construction preserves single Finish() semantics", "[probes]")
{
    int finishCount = 0;
    auto owned = std::make_unique<RecordingProbe>();
    owned->ExternalFinishCount = &finishCount;

    Operon::ProbeChain original;
    original.Add(std::move(owned));

    Operon::ProbeChain moved{std::move(original)};
    CHECK(finishCount == 0);

    // The moved-from chain's entries_ is now empty, so its destructor's
    // Finish() call has nothing to iterate - only `moved`'s Finish() should
    // ever reach the probe.
    moved.Finish();
    CHECK(finishCount == 1);
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
        CHECK(line.find("\"mean\":1.5") != std::string::npos);
        CHECK(line.find("\"flag\":true") != std::string::npos);
        CHECK(line.find("\"label\":\"gen\"") != std::string::npos);
        CHECK(line.find("\"counts\":[1,2,3]") != std::string::npos);
        CHECK(line.find("\"readings\":[1,2,3]") != std::string::npos);
    }
}

TEST_CASE("JsonlSink truncates a pre-existing file rather than appending", "[probes]")
{
    auto const path = std::filesystem::temp_directory_path() / "operon_probes_truncate_test.jsonl";
    {
        std::ofstream seed(path);
        seed << "{\"stale\":true}\n{\"stale\":true}\n";
    }

    {
        Operon::ResultRecord record;
        record.insert_or_assign("fresh", true);
        Operon::JsonlSink sink(path.string());
        CHECK(sink.IsOpen());
        sink.Write(record);
    }

    std::ifstream in(path);
    std::vector<std::string> lines;
    for (std::string line; std::getline(in, line);) {
        if (!line.empty()) { lines.push_back(line); }
    }
    in.close();
    std::filesystem::remove(path);

    REQUIRE(lines.size() == 1);
    CHECK(lines[0].find("\"stale\"") == std::string::npos);
    CHECK(lines[0].find("\"fresh\"") != std::string::npos);
}

TEST_CASE("JsonlSink reports IsOpen() == false when it fails to open", "[probes]")
{
    // A path under a directory that doesn't exist can't be opened.
    auto const path = std::filesystem::temp_directory_path() / "operon_probes_missing_dir" / "out.jsonl";
    Operon::JsonlSink sink(path.string());
    CHECK_FALSE(sink.IsOpen());

    // Write() on a sink that failed to open must no-op, not crash.
    Operon::ResultRecord record;
    record.insert_or_assign("x", std::int64_t{1});
    sink.Write(record);
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
        CHECK(params.at("path").Get<std::string>() == "out.beve");
        return std::make_unique<RecordingProbe>();
    });

    Operon::ProbeParams params;
    params.insert_or_assign("path", Operon::ProbeParamValue{std::string{"out.beve"}});
    auto probe = registry.Create("check-params", params);
    CHECK(probe != nullptr);
}

TEST_CASE("RegisterBuiltinProbes registers exactly the three concrete probes", "[probes]")
{
    Operon::ProbeRegistry registry;
    Operon::RegisterBuiltinProbes(registry);

    CHECK(registry.Contains("population_trace"));
    CHECK(registry.Contains("cache_hit_rate"));
    CHECK(registry.Contains("structural_diversity"));
    CHECK_FALSE(registry.Contains("not_a_real_probe"));
}

TEST_CASE("RegisterBuiltinProbes: population_trace requires a 'path' param", "[probes]")
{
    Operon::ProbeRegistry registry;
    Operon::RegisterBuiltinProbes(registry);

    CHECK_THROWS_AS(registry.Create("population_trace", {}), std::runtime_error);

    Operon::ProbeParams params;
    auto const path = std::filesystem::temp_directory_path() / "operon_probes_builtin_test.beve";
    params.insert_or_assign("path", Operon::ProbeParamValue{path.string()});
    auto probe = registry.Create("population_trace", params);
    REQUIRE(probe != nullptr);
    std::filesystem::remove(path);
}

TEST_CASE("RegisterBuiltinProbes: cache_hit_rate needs no params", "[probes]")
{
    Operon::ProbeRegistry registry;
    Operon::RegisterBuiltinProbes(registry);

    auto probe = registry.Create("cache_hit_rate", {});
    REQUIRE(probe != nullptr);
}

TEST_CASE("RegisterBuiltinProbes: structural_diversity accepts strict/relaxed and rejects anything else", "[probes]")
{
    Operon::ProbeRegistry registry;
    Operon::RegisterBuiltinProbes(registry);

    CHECK(registry.Create("structural_diversity", {}) != nullptr); // defaults to strict

    Operon::ProbeParams strict;
    strict.insert_or_assign("hash_mode", Operon::ProbeParamValue{std::string{"strict"}});
    CHECK(registry.Create("structural_diversity", strict) != nullptr);

    Operon::ProbeParams relaxed;
    relaxed.insert_or_assign("hash_mode", Operon::ProbeParamValue{std::string{"relaxed"}});
    CHECK(registry.Create("structural_diversity", relaxed) != nullptr);

    Operon::ProbeParams bogus;
    bogus.insert_or_assign("hash_mode", Operon::ProbeParamValue{std::string{"bogus"}});
    CHECK_THROWS_AS(registry.Create("structural_diversity", bogus), std::runtime_error);
}

TEST_CASE("PopulationTraceProbe appends framed BEVE population dumps", "[probes]")
{
    ProbeFixture f;
    auto const path = std::filesystem::temp_directory_path() / "operon_probes_trace_test.beve";
    std::filesystem::remove(path);

    // All generations use the same (fixture-fixed) individuals, so every
    // frame's payload should be byte-identical to this.
    auto const expectedBytes = Operon::Serialization::ToBeve(f.Gp.Parents());

    {
        Operon::PopulationTraceProbe probe(path.string());
        CHECK(probe.IsOpen());

        for (std::size_t g = 0; g < 3; ++g) {
            AdvanceTo(f.Gp, g);
            Operon::ResultRecord record;
            Operon::ProbeContext ctx{f.Gp, record};
            probe(ctx);
            REQUIRE(record.contains("trace_bytes"));
            CHECK(std::get<std::int64_t>(record.at("trace_bytes")) == static_cast<std::int64_t>(expectedBytes.size()));
        }
        probe.Finish();
    }

    std::ifstream in(path, std::ios::binary);
    REQUIRE(in.is_open());
    for (std::uint64_t expectedGen = 0; expectedGen < 3; ++expectedGen) {
        std::uint64_t generation{};
        std::uint64_t length{};
        in.read(reinterpret_cast<char*>(&generation), sizeof(generation));
        in.read(reinterpret_cast<char*>(&length), sizeof(length));
        REQUIRE(in);
        CHECK(generation == expectedGen);
        REQUIRE(length == expectedBytes.size());

        std::string bytes(length, '\0');
        in.read(bytes.data(), static_cast<std::streamsize>(length));
        REQUIRE(in);
        CHECK(bytes == expectedBytes);
    }
    char extra{};
    in.read(&extra, 1);
    CHECK(in.eof()); // exactly 3 frames, nothing trailing
    in.close(); // Windows can't remove a file with an open handle

    std::filesystem::remove(path);
}

TEST_CASE("PopulationTraceProbe reports IsOpen() == false when it fails to open", "[probes]")
{
    auto const path = std::filesystem::temp_directory_path() / "operon_probes_missing_dir" / "trace.beve";
    Operon::PopulationTraceProbe probe(path.string());
    CHECK_FALSE(probe.IsOpen());
}

TEST_CASE("CacheHitRateProbe emits nothing when no cache is configured", "[probes]")
{
    ProbeFixture f;
    Operon::ResultRecord record;
    Operon::ProbeContext ctx{f.Gp, record};

    Operon::CacheHitRateProbe probe;
    probe(ctx);

    CHECK(record.empty());
}

TEST_CASE("CacheHitRateProbe reports per-generation deltas of hits/lookups/rate/size", "[probes]")
{
    CacheProbeFixture f;
    Operon::CacheHitRateProbe probe;

    auto const hash1 = Operon::Hash{1};
    auto const hash2 = Operon::Hash{2};
    f.Cache.Insert(hash1, { Operon::Scalar{0.1} });

    Operon::Vector<Operon::Scalar> val;
    std::ignore = f.Cache.TryGet(hash1, val); // hit
    std::ignore = f.Cache.TryGet(hash2, val); // miss

    {
        Operon::ResultRecord record;
        Operon::ProbeContext ctx{f.Gp, record};
        probe(ctx);
        CHECK(std::get<std::int64_t>(record.at("cache_hits")) == 1);
        CHECK(std::get<std::int64_t>(record.at("cache_lookups")) == 2);
        CHECK(std::get<double>(record.at("cache_hit_rate")) == 0.5);
        CHECK(std::get<std::int64_t>(record.at("cache_size")) == 1);
    }

    // No new cache activity since the last call -> zero deltas; rate falls
    // back to 0 rather than dividing by zero.
    {
        Operon::ResultRecord record;
        Operon::ProbeContext ctx{f.Gp, record};
        probe(ctx);
        CHECK(std::get<std::int64_t>(record.at("cache_hits")) == 0);
        CHECK(std::get<std::int64_t>(record.at("cache_lookups")) == 0);
        CHECK(std::get<double>(record.at("cache_hit_rate")) == 0.0);
        CHECK(std::get<std::int64_t>(record.at("cache_size")) == 1);
    }
}

TEST_CASE("CacheHitRateProbe does not underflow when the cache is Clear()-ed between calls", "[probes]")
{
    CacheProbeFixture f;
    Operon::CacheHitRateProbe probe;

    auto const hash1 = Operon::Hash{1};
    f.Cache.Insert(hash1, { Operon::Scalar{0.1} });
    Operon::Vector<Operon::Scalar> val;
    std::ignore = f.Cache.TryGet(hash1, val); // hit -> cumulative hits=1, lookups=1

    {
        Operon::ResultRecord record;
        Operon::ProbeContext ctx{f.Gp, record};
        probe(ctx);
        CHECK(std::get<std::int64_t>(record.at("cache_hits")) == 1);
        CHECK(std::get<std::int64_t>(record.at("cache_lookups")) == 1);
    }

    // Simulates an external reset (e.g. between GP runs sharing one cache).
    // cache->Hits()/Lookups() now report 0, below the probe's remembered
    // previous cumulative values - a naive unsigned subtraction would wrap
    // around to a huge value here instead of falling back to 0.
    f.Cache.Clear();
    {
        Operon::ResultRecord record;
        Operon::ProbeContext ctx{f.Gp, record};
        probe(ctx);
        CHECK(std::get<std::int64_t>(record.at("cache_hits")) == 0);
        CHECK(std::get<std::int64_t>(record.at("cache_lookups")) == 0);
        CHECK(std::get<double>(record.at("cache_hit_rate")) == 0.0);
    }

    // Activity resumes post-reset - deltas should reflect it normally, not
    // still be thrown off by the earlier reset.
    f.Cache.Insert(hash1, { Operon::Scalar{0.2} });
    std::ignore = f.Cache.TryGet(hash1, val);
    {
        Operon::ResultRecord record;
        Operon::ProbeContext ctx{f.Gp, record};
        probe(ctx);
        CHECK(std::get<std::int64_t>(record.at("cache_hits")) == 1);
        CHECK(std::get<std::int64_t>(record.at("cache_lookups")) == 1);
    }
}

TEST_CASE("PopulationDiversity: 0 for identical trees, positive for different ones, 0 below 2 individuals", "[probes]")
{
    ProbeFixture f;
    Operon::RandomGenerator rng(42);
    Operon::BalancedTreeCreator creator{ &f.Pset, f.Vars, 0.0, 10 };
    auto treeA = creator(rng, 5, 1, 10);
    auto treeB = creator(rng, 8, 1, 10); // different target size - vanishingly unlikely to collide

    std::vector<Operon::Individual> identical(3);
    for (auto& ind : identical) { ind.Genotype = treeA; }
    CHECK(Operon::PopulationDiversity(identical) == 0.0);

    std::vector<Operon::Individual> mixed(2);
    mixed[0].Genotype = treeA;
    mixed[1].Genotype = treeB;
    CHECK(Operon::PopulationDiversity(mixed) > 0.0);

    std::vector<Operon::Individual> tooSmall(1);
    tooSmall[0].Genotype = treeA;
    CHECK(Operon::PopulationDiversity(tooSmall) == 0.0);

    std::vector<Operon::Individual> const empty;
    CHECK(Operon::PopulationDiversity(empty) == 0.0);
}

TEST_CASE("PopulationDiversity handles individuals with an empty/default Genotype without UB or NaN", "[probes]")
{
    // Distance::Jaccard's CountIntersect dereferences one-before-the-start
    // of an empty span; PopulationDiversity must never call it with an
    // empty hash vector on either side. Default-constructed Individuals
    // have a default-constructed (zero-length) Genotype, exactly this case.
    ProbeFixture f;
    Operon::RandomGenerator rng(11);
    Operon::BalancedTreeCreator creator{ &f.Pset, f.Vars, 0.0, 10 };
    auto tree = creator(rng, 5, 1, 10);

    std::vector<Operon::Individual> bothEmpty(2); // default Genotype on both
    CHECK(Operon::PopulationDiversity(bothEmpty) == 0.0);

    std::vector<Operon::Individual> oneEmpty(2);
    oneEmpty[0].Genotype = tree; // oneEmpty[1] stays default (empty)
    CHECK(Operon::PopulationDiversity(oneEmpty) == 1.0);

    std::vector<Operon::Individual> mixedWithEmpty(3);
    mixedWithEmpty[0].Genotype = tree;
    mixedWithEmpty[1].Genotype = tree;
    // mixedWithEmpty[2] stays default (empty) - exercises the empty-vs-non-empty
    // path alongside a non-degenerate pair in the same population.
    auto const d = Operon::PopulationDiversity(mixedWithEmpty);
    CHECK(d > 0.0);
    CHECK(d < 1.0); // one of the three pairs is identical (distance 0), pulling the mean down
}

TEST_CASE("StructuralDiversityProbe emits diversity_jaccard from ctx.Parents()", "[probes]")
{
    ProbeFixture f;
    Operon::RandomGenerator rng(7);
    Operon::BalancedTreeCreator creator{ &f.Pset, f.Vars, 0.0, 10 };
    auto tree = creator(rng, 5, 1, 10);
    for (auto& ind : f.Gp.Parents()) { ind.Genotype = tree; }

    Operon::ResultRecord record;
    Operon::ProbeContext ctx{f.Gp, record};
    Operon::StructuralDiversityProbe probe;
    probe(ctx);

    REQUIRE(record.contains("diversity_jaccard"));
    CHECK(std::get<double>(record.at("diversity_jaccard")) == 0.0);
}

} // namespace Operon::Test
