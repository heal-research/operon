// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>

#include <cmath>
#include <csignal>
#include <cstdlib>
#include <cxxopts.hpp>
#include <filesystem>
#include <fstream>
#include <limits>
#include <optional>
#include <sstream>
#include <stdexcept>
#include <string>
#include <thread>
#include <tuple>
#include <vector>

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

#include "operon/algorithms/stop_reason.hpp"
#include "operon/core/dataset.hpp"
#include "operon/hash/model_ref.hpp"
#include "operon/hash/sha256.hpp"
#include "operon/parser/infix.hpp"

#include "final_record.hpp"
#include "stop_fixture.hpp"
#include "stop_signal.hpp"

namespace Operon::Test {
namespace {
    using Operon::Cli::FinalModel;
    using Operon::Cli::FinalRecord;
    using Operon::Cli::FinalRun;

    constexpr auto RunId = "aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa";
    constexpr auto AttemptId = "11111111-2222-4333-8444-555555555555";

    auto Mask(auto... reasons) -> Operon::StopMask { return (Operon::MaskOf(reasons) | ...); }

    auto ReadFile(std::filesystem::path const& path) -> std::string
    {
        std::ifstream in(path, std::ios::binary);
        std::ostringstream ss;
        ss << in.rdbuf();
        return ss.str();
    }

    auto ScratchDir(std::string const& name) -> std::filesystem::path
    {
        auto dir = std::filesystem::temp_directory_path() / ("operon-final-record-" + name);
        std::filesystem::remove_all(dir);
        std::filesystem::create_directories(dir);
        return dir;
    }

    auto Contains(std::string const& text, std::string const& part) -> bool
    {
        return text.find(part) != std::string::npos;
    }

    // The options a --final-json run has, parsed the way the CLI parses them.
    auto ParseFinalOptions(std::vector<std::string> args) -> cxxopts::ParseResult
    {
        cxxopts::Options opts("test", "test");
        opts.add_options()("final-json", "", cxxopts::value<std::string>())(
            "run-id", "", cxxopts::value<std::string>())("attempt-id", "", cxxopts::value<std::string>())(
            "resume", "", cxxopts::value<std::string>());
        args.insert(args.begin(), "test");
        std::vector<char*> argv;
        for (auto& a : args) {
            argv.push_back(a.data());
        }
        return opts.parse(static_cast<int>(argv.size()), argv.data());
    }

    auto FullRecord() -> FinalRecord
    {
        FinalRecord r;
        r.RunId = RunId;
        r.AttemptId = AttemptId;
        r.Reasons = Mask(Operon::StopReason::GenerationLimit, Operon::StopReason::EvaluationLimit);
        r.GenerationsCompleted = 4;
        r.Incomplete = Operon::OffspringProgress { .Attempted = 7, .Completed = 5, .PoolSize = 12 };
        r.GenerationLimit = 10;
        r.EvaluationLimit = 100;
        r.ElapsedSeconds = 1.5;
        r.PressureMax = 100;
        r.PressureLast = 3.25;
        r.Stats = Operon::EvaluationStats { .Calls = 40,
            .ResidualScoring = 40,
            .ResidualOptimizer = 12,
            .JacobianScoring = 1,
            .JacobianOptimizer = 8,
            .OptimizerTimeUs = 2'500'000,
            .CacheHits = 10,
            .CacheLookups = 50 };
        r.Model = FinalModel { "obj0", std::string(64, 'b'), "((1.0 * X1) + 0.5)" };
        r.UnsupportedFields = { "segments", "record_count" };
        return r;
    }
} // namespace

TEST_CASE("operon.final document: golden text", "[final-record]")
{
    auto const text = Operon::Cli::SerializeFinalRecord(FullRecord());
    constexpr auto counters
        = R"({"evaluator_calls":40,"residual_scoring":40,"residual_optimizer":12,"jacobian_scoring":1,)"
          R"("jacobian_optimizer":8,"residual_total":52,"jacobian_total":9,"cache_hits":10,"cache_misses":40,)"
          R"("optimizer_seconds":2.5})";
    auto const expected = std::string { R"({"schema":"operon.final","schema_version":"1.2","run_id":")" } + RunId
        + R"(","attempt_id":")" + AttemptId + R"(","exit_status":"ok",)"
        + R"("termination":{"primary":"evaluation_limit","reasons":["evaluation_limit","generation_limit"],)"
        + R"("precedence_version":2},"generations_completed":4,)"
        + R"("incomplete_generation":{"attempted_offspring":7,"completed_offspring":5,"pool_count":12},)"
        + R"("budgets":{"generations":{"limit":10},)"
        + R"("evaluations":{"unit":"residual+jacobian","limit":100,"consumed":61,"overshoot":0},)"
        + R"("time":{"limit_s":null,"elapsed_s":1.5},"pressure":{"max":100,"last":3.25}},)"
        + R"("counters":{"logical":)" + counters + R"(,"since_restart":)" + counters + "}," + R"("final_model":{)"
        + R"("selection_rule":"obj0","model_ref":")" + std::string(64, 'b') + R"j(","infix":"((1.0 * X1) + 0.5)"},)j"
        + R"("front_ref":null,"segments":[],"record_count":0,"instrumentation_errors":0,)"
        + R"("producer":{"adapter":"operon","unsupported_fields":["segments","record_count"],"derived_fields":[]}})"
        + "\n";
    CHECK(text == expected);
}

TEST_CASE("operon.final document: nulls, overshoot and escaping", "[final-record]")
{
    SECTION("no cache, no limits, no model, no incomplete generation")
    {
        FinalRecord r;
        r.RunId = RunId;
        r.AttemptId = AttemptId;
        r.Reasons = Mask(Operon::StopReason::Callback);
        auto const text = Operon::Cli::SerializeFinalRecord(r);
        CHECK(Contains(text, R"("primary":"callback","reasons":["callback"])"));
        CHECK(Contains(text, R"("incomplete_generation":null)"));
        CHECK(Contains(text, R"("generations":{"limit":null})"));
        CHECK(
            Contains(text, R"("evaluations":{"unit":"residual+jacobian","limit":null,"consumed":0,"overshoot":null})"));
        CHECK(Contains(text, R"("time":{"limit_s":null,"elapsed_s":0},"pressure":{"max":null,"last":null})"));
        CHECK(Contains(text, R"("cache_hits":null,"cache_misses":null)"));
        CHECK(Contains(text, R"("final_model":null,"front_ref":null)"));
        CHECK_FALSE(Contains(text, "producer"));
    }

    SECTION("overshoot is consumed - limit")
    {
        FinalRecord r = FullRecord();
        r.EvaluationLimit = 50;
        CHECK(Contains(Operon::Cli::SerializeFinalRecord(r), R"("limit":50,"consumed":61,"overshoot":11})"));
    }

    SECTION("non-finite numbers never reach the document")
    {
        FinalRecord r = FullRecord();
        r.ElapsedSeconds = std::numeric_limits<double>::quiet_NaN();
        r.PressureLast = std::numeric_limits<double>::infinity();
        auto const text = Operon::Cli::SerializeFinalRecord(r);
        CHECK(Contains(text, R"("elapsed_s":null)"));
        CHECK(Contains(text, R"("last":null)"));
        CHECK_FALSE(Contains(text, ":nan"));
        CHECK_FALSE(Contains(text, ":inf"));
    }

    SECTION("strings are escaped")
    {
        FinalRecord r = FullRecord();
        r.Model->Infix = "a\"b\\c\n\t\x01";
        CHECK(Contains(Operon::Cli::SerializeFinalRecord(r), R"("infix":"a\"b\\c\n\t\u0001")"));
    }

    SECTION("an error record names engine_error first, then what else held")
    {
        FinalRecord r = FullRecord();
        r.Error = true;
        r.Reasons |= Operon::MaskOf(Operon::StopReason::EngineError);
        auto const text = Operon::Cli::SerializeFinalRecord(r);
        CHECK(Contains(text, R"("exit_status":"error")"));
        CHECK(Contains(
            text, R"("primary":"engine_error","reasons":["engine_error","evaluation_limit","generation_limit"])"));
    }
}

TEST_CASE("Unlimited limits are reported as null", "[final-record]")
{
    Operon::GeneticAlgorithmConfig config;
    config.Generations = 1000;
    config.Evaluations = std::size_t { 1 } << 62U;
    config.TimeLimit = ~std::size_t { 0 };
    auto record = Operon::Cli::MakeFinalRecord(config);
    CHECK(record.GenerationLimit == std::optional<std::size_t> { 1000 });
    CHECK_FALSE(record.EvaluationLimit.has_value());
    CHECK_FALSE(record.TimeLimitSeconds.has_value());
    CHECK(record.Reasons == 0);
    CHECK(record.Stats == Operon::EvaluationStats {});

    config.Evaluations = (std::size_t { 1 } << 62U) - 1;
    config.TimeLimit = 30;
    record = Operon::Cli::MakeFinalRecord(config);
    CHECK(record.EvaluationLimit == std::optional<std::size_t> { (std::size_t { 1 } << 62U) - 1 });
    CHECK(record.TimeLimitSeconds == std::optional<double> { 30.0 });
}

TEST_CASE("WriteFileAtomically publishes the whole file or nothing", "[final-record]")
{
    auto const dir = ScratchDir("atomic");
    auto const path = dir / "final.json";

    Operon::Cli::WriteFileAtomically(path, "first\n");
    CHECK(ReadFile(path) == "first\n");
    Operon::Cli::WriteFileAtomically(path, std::string(1 << 20, 'x') + "\n");
    CHECK(ReadFile(path).size() == (1U << 20U) + 1);

    // no temporary files are left behind
    std::size_t entries = 0;
    for ([[maybe_unused]] auto const& e : std::filesystem::directory_iterator(dir)) {
        ++entries;
    }
    CHECK(entries == 1);

    CHECK_THROWS_AS(Operon::Cli::WriteFileAtomically(dir / "missing" / "final.json", "x"), std::runtime_error);
    std::filesystem::remove_all(dir);
}

TEST_CASE("FinalRun validates its options", "[final-record]")
{
    SECTION("nothing requested: disabled, and the exit code is untouched")
    {
        auto const result = ParseFinalOptions({});
        FinalRun run { result };
        CHECK_FALSE(run.Enabled());
        CHECK(run.Finish(0) == 0);
        CHECK(run.Finish(3) == 3);
    }

    SECTION("--final-json needs both identifiers")
    {
        auto const path = (std::filesystem::temp_directory_path() / "operon-final-record-never.json").string();
        CHECK_THROWS_AS(FinalRun { ParseFinalOptions({ "--final-json", path }) }, std::invalid_argument);
        CHECK_THROWS_AS(
            FinalRun { ParseFinalOptions({ "--final-json", path, "--run-id", RunId }) }, std::invalid_argument);
        CHECK_THROWS_AS(
            FinalRun { ParseFinalOptions({ "--final-json", path, "--attempt-id", AttemptId }) }, std::invalid_argument);
    }

    SECTION("identifiers are checked against the record schema")
    {
        auto const path = (std::filesystem::temp_directory_path() / "operon-final-record-never.json").string();
        auto const make = [&](std::string const& run, std::string const& attempt) {
            return FinalRun { ParseFinalOptions({ "--final-json", path, "--run-id", run, "--attempt-id", attempt }) };
        };
        CHECK_THROWS_AS(make("abc", AttemptId), std::invalid_argument);
        CHECK_THROWS_AS(make(std::string(64, 'A'), AttemptId), std::invalid_argument); // upper case
        CHECK_THROWS_AS(make(std::string(63, 'a'), AttemptId), std::invalid_argument);
        CHECK_THROWS_AS(make(RunId, "not-a-uuid"), std::invalid_argument);
        CHECK_THROWS_AS(make(RunId, "11111111-2222-4333-8444-55555555555"), std::invalid_argument);
        CHECK_THROWS_AS(make(RunId, "11111111-2222-4333-8444-55555555555G"), std::invalid_argument);
        CHECK_NOTHROW(make(RunId, AttemptId));
    }

    SECTION("--final-json rejects resumed counters without checkpoint accounting")
    {
        auto const path = (std::filesystem::temp_directory_path() / "operon-final-record-never.json").string();
        CHECK_THROWS_AS(FinalRun { ParseFinalOptions({ "--final-json", path, "--run-id", RunId, "--attempt-id",
                            AttemptId, "--resume", "run.beve" }) },
            std::invalid_argument);
    }

    SECTION("identifiers without --final-json are a mistake")
    {
        CHECK_THROWS_AS(FinalRun { ParseFinalOptions({ "--run-id", RunId }) }, std::invalid_argument);
    }
}

TEST_CASE("FinalRun writes the record of a finished run", "[final-record]")
{
    auto const dir = ScratchDir("run");
    auto const path = dir / "final.json";
    auto const result
        = ParseFinalOptions({ "--final-json", path.string(), "--run-id", RunId, "--attempt-id", AttemptId });
    StopFixture fix;
    Operon::RandomGenerator rng { 3 };
    fix.Config.Generations = 3;

    SECTION("generation limit")
    {
        FinalRun run { result };
        REQUIRE(run.Enabled());
        {
            auto gp = fix.MakeGp();
            FinalRun::Watch const watch { run, *gp };
            gp->Run(rng, nullptr, /*threads=*/1);
            auto const tree = Operon::InfixParser::ParseOrThrow("X1 * 2", fix.Ds);
            run.SetModel("obj0", tree, "(X1 * 2.0)");
        }
        CHECK(run.Finish(0) == 0);

        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"ok")"));
        CHECK(Contains(text, R"("primary":"generation_limit","reasons":["generation_limit"])"));
        CHECK(Contains(text, R"("generations_completed":3,"incomplete_generation":null)"));
        CHECK(Contains(text, R"("generations":{"limit":3})"));
        // counters are the evaluator's, read after Run(); no local search, so scoring is all the work there is
        auto const [residual, jacobian, calls] = fix.Evaluator.Stats();
        CHECK(calls == StopFixture::PopSize * 4); // initialization + three generations
        CHECK(Contains(text, "\"evaluator_calls\":" + std::to_string(calls)));
        CHECK(Contains(text, "\"consumed\":" + std::to_string(residual + jacobian)));
        CHECK(Contains(
            text, "\"model_ref\":\"" + Operon::ModelRef(Operon::InfixParser::ParseOrThrow("X1 * 2", fix.Ds)) + "\""));
        CHECK(Contains(text, R"("front_ref":null,"segments":[],"record_count":0)"));
    }

    SECTION("a run that ends in the middle of a pool reports the cut-short generation and the last filled one")
    {
        // initialization + one full generation + half of the second pool
        fix.Config.Generations = 100;
        fix.Config.Evaluations = StopFixture::PopSize + StopFixture::PoolSize + (StopFixture::PoolSize / 2);
        FinalRun run { result };
        {
            auto gp = fix.MakeGp();
            FinalRun::Watch const watch { run, *gp };
            gp->Run(rng, nullptr, /*threads=*/1);
            CHECK(gp->Generation() == 1); // the discarded generation is not counted
        }
        CHECK(run.Finish(0) == 0);
        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"ok")"));
        CHECK(Contains(text, R"("primary":"evaluation_limit","reasons":["evaluation_limit"])"));
        CHECK(Contains(text, R"("generations_completed":1,)"));
        // the second generation is discarded once it has spent more than the 6 evaluations the budget had left
        CHECK(Contains(
            text, R"("incomplete_generation":{"attempted_offspring":7,"completed_offspring":7,"pool_count":12})"));
        CHECK(Contains(text, R"("consumed":31,"overshoot":1)"));
    }

    SECTION("a failing process makes it an error record, and the counters are still there")
    {
        FinalRun run { result };
        {
            auto gp = fix.MakeGp();
            FinalRun::Watch const watch { run, *gp };
            gp->Run(rng, nullptr, /*threads=*/1);
        }
        CHECK(run.Finish(1) == 1);
        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"error")"));
        CHECK(Contains(text, R"("primary":"engine_error")"));
        CHECK(Contains(text, R"("final_model":null)"));
        CHECK(Contains(text, "\"evaluator_calls\":" + std::to_string(std::get<2>(fix.Evaluator.Stats()))));
    }

    SECTION("an exception that unwinds through the watch still leaves the snapshot")
    {
        FinalRun run { result };
        run.SetConfig(fix.Config);
        try {
            auto gp = fix.MakeGp();
            FinalRun::Watch const watch { run, *gp };
            gp->Run(
                rng,
                [] {
                    throw std::runtime_error("boom");
                    return false;
                },
                /*threads=*/1);
        } catch (std::runtime_error const&) {
            run.MarkError();
        }
        CHECK(run.Finish(EXIT_FAILURE) == EXIT_FAILURE);
        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"error")"));
        CHECK(Contains(text, "\"evaluator_calls\":" + std::to_string(std::get<2>(fix.Evaluator.Stats()))));
        CHECK(std::get<2>(fix.Evaluator.Stats()) == StopFixture::PopSize); // initialization was evaluated
    }

    SECTION("a run that never started has zero counters and its configured limits")
    {
        FinalRun run { result };
        run.SetConfig(fix.Config);
        CHECK(run.Finish(EXIT_FAILURE) == EXIT_FAILURE);
        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"error")"));
        CHECK(Contains(text, R"("generations_completed":0,"incomplete_generation":null)"));
        CHECK(Contains(text, R"("generations":{"limit":3})"));
        CHECK(Contains(text, "\"evaluator_calls\":0"));
    }

    SECTION("an abandoned run writes nothing")
    {
        FinalRun run { result };
        run.Abandon();
        CHECK(run.Finish(0) == 0);
        CHECK_FALSE(std::filesystem::exists(path));
    }

    SECTION("an unwritable destination is reported as a failure")
    {
        auto const bad = ParseFinalOptions({ "--final-json", (dir / "missing" / "final.json").string(), "--run-id",
            RunId, "--attempt-id", AttemptId });
        FinalRun run { bad };
        run.SetConfig(fix.Config);
        CHECK(run.Finish(0) == EXIT_FAILURE);
    }

    std::filesystem::remove_all(dir);
}

namespace {
    // The options of a --generation-log run, parsed the way the CLI parses them (defaults included).
    auto ParseStreamOptions(std::vector<std::string> args) -> cxxopts::ParseResult
    {
        cxxopts::Options opts("test", "test");
        opts.add_options()("final-json", "", cxxopts::value<std::string>())(
            "run-id", "", cxxopts::value<std::string>())("attempt-id", "", cxxopts::value<std::string>())(
            "resume", "", cxxopts::value<std::string>())("generation-log", "", cxxopts::value<std::string>())(
            "sync", "", cxxopts::value<std::string>()->default_value("record"))(
            "objective", "", cxxopts::value<std::string>()->default_value("mse"))(
            "offspring-generator", "", cxxopts::value<std::string>()->default_value("basic"))(
            "model-selection", "", cxxopts::value<std::string>()->default_value("mdl"))(
            "segment-generations", "", cxxopts::value<std::size_t>()->default_value("1000"))(
            "segment-seconds", "", cxxopts::value<double>()->default_value("5"));
        args.insert(args.begin(), "test");
        std::vector<char*> argv;
        for (auto& a : args) {
            argv.push_back(a.data());
        }
        return opts.parse(static_cast<int>(argv.size()), argv.data());
    }

    auto StreamArgs(std::filesystem::path const& final, std::filesystem::path const& stream,
        std::vector<std::string> extra = {}) -> std::vector<std::string>
    {
        std::vector<std::string> args { "--final-json", final.string(), "--run-id", RunId, "--attempt-id", AttemptId,
            "--generation-log", stream.string() };
        args.insert(args.end(), extra.begin(), extra.end());
        return args;
    }

    // One-objective parents with distinct fitness values, so that the recorder can capture and write a record.
    auto PrepareParents(Operon::GeneticProgrammingAlgorithm& gp, Operon::Dataset& ds) -> void
    {
        std::size_t i = 0;
        for (auto& ind : gp.Parents()) {
            ind.Genotype = Operon::InfixParser::ParseOrThrow(i % 2 ? "X1" : "X1 * 2", ds);
            ind.Fitness[0] = static_cast<Operon::Scalar>(++i);
        }
    }

    auto WriteRecord(Operon::GenerationRecorder& recorder, Operon::GeneticProgrammingAlgorithm& gp) -> void
    {
        recorder.Capture(gp);
        recorder.Write(gp, gp.Parents()[0], 0.25, 0.5, 1.0);
        ++gp.Generation();
    }

    constexpr auto NoStream = R"("segments":[],"record_count":0,"instrumentation_errors":1,)";
    constexpr auto UnsupportedStream
        = R"("producer":{"adapter":"operon","unsupported_fields":["segments","record_count"],"derived_fields":[]}})";
} // namespace

TEST_CASE("FinalRun declares the stream manifest unsupported when no complete stream exists", "[final-record]")
{
    auto const dir = ScratchDir("stream");
    auto const path = dir / "final.json";
    auto const stream = dir / "stream";
    StopFixture fix;

    SECTION("the recorder could not be started: invalid option")
    {
        auto const result = ParseStreamOptions(StreamArgs(path, stream, { "--sync", "bogus" }));
        FinalRun run { result };
        run.SetConfig(fix.Config);
        CHECK_THROWS_AS(run.StartGenerationLog(result, false, 1), std::invalid_argument);
        CHECK(run.Finish(0) == EXIT_FAILURE);
        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"error")"));
        CHECK(Contains(text, NoStream));
        CHECK(Contains(text, UnsupportedStream));
    }

    SECTION("the recorder could not be started: the directory is a regular file")
    {
        std::ofstream { dir / "occupied" } << "x";
        auto const result = ParseStreamOptions(StreamArgs(path, dir / "occupied"));
        FinalRun run { result };
        run.SetConfig(fix.Config);
        CHECK_THROWS(run.StartGenerationLog(result, false, 1));
        CHECK(run.Finish(0) == EXIT_FAILURE);
        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"error")"));
        CHECK(Contains(text, NoStream));
        CHECK(Contains(text, UnsupportedStream));
    }

    SECTION("sealing failed: the manifest is neither empty-and-complete nor partial")
    {
        auto const result = ParseStreamOptions(StreamArgs(path, stream, { "--segment-generations", "1" }));
        FinalRun run { result };
        run.SetConfig(fix.Config);
        auto gp = fix.MakeGp();
        PrepareParents(*gp, fix.Ds);
        auto* const recorder = run.StartGenerationLog(result, false, 1);
        REQUIRE(recorder != nullptr);
        WriteRecord(*recorder, *gp); // sealed into generation-000000.jsonl
        std::filesystem::create_directory(stream / "generation-000001.jsonl"); // the next seal cannot publish
        WriteRecord(*recorder, *gp);
        CHECK(run.Finish(0) == EXIT_FAILURE);

        auto const sealed = Operon::Sha256FileHex(stream / "generation-000000.jsonl");
        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"error")"));
        CHECK(Contains(text, NoStream)); // not "segments":[{...one sealed segment...}],"record_count":2
        CHECK(Contains(text, UnsupportedStream));
        CHECK_FALSE(Contains(text, sealed));
        CHECK_FALSE(Contains(text, R"("record_count":2)"));
    }

    SECTION("a background seal failure consumed by a later write still leaves the manifest unsupported")
    {
        auto const result = ParseStreamOptions(StreamArgs(path, stream, { "--segment-generations", "1" }));
        FinalRun run { result };
        run.SetConfig(fix.Config);
        auto gp = fix.MakeGp();
        PrepareParents(*gp, fix.Ds);
        auto* const recorder = run.StartGenerationLog(result, false, 1);
        REQUIRE(recorder != nullptr);
        std::filesystem::create_directory(stream / "generation-000000.jsonl"); // the first seal cannot publish
        WriteRecord(*recorder, *gp); // the seal fails in the background
        // The next write waits for the seal worker and consumes (rethrows) its error, so that Close() finds nothing
        // to rethrow and returns normally.
        CHECK_THROWS(WriteRecord(*recorder, *gp));
        CHECK_NOTHROW(recorder->Close());
        CHECK(recorder->Errors() == 1);
        CHECK(recorder->RecordCount() == 1); // the count includes the segment that was never sealed
        CHECK(recorder->Segments().empty());
        CHECK(run.Finish(0) == EXIT_FAILURE);

        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"error")"));
        CHECK(Contains(text, NoStream));
        CHECK(Contains(text, UnsupportedStream));
        CHECK_FALSE(Contains(text, R"("record_count":1)"));
    }

    SECTION("close failed with no sealed segment")
    {
        auto const result = ParseStreamOptions(StreamArgs(path, stream));
        FinalRun run { result };
        run.SetConfig(fix.Config);
        auto gp = fix.MakeGp();
        PrepareParents(*gp, fix.Ds);
        auto* const recorder = run.StartGenerationLog(result, false, 1);
        REQUIRE(recorder != nullptr);
        WriteRecord(*recorder, *gp);
        std::filesystem::create_directory(stream / "generation-000000.jsonl");
        CHECK(run.Finish(0) == EXIT_FAILURE);
        auto const text = ReadFile(path);
        CHECK(Contains(text, NoStream));
        CHECK(Contains(text, UnsupportedStream));
    }

    SECTION("a sealed stream keeps its manifest and declares nothing unsupported")
    {
        auto const result = ParseStreamOptions(StreamArgs(path, stream));
        FinalRun run { result };
        fix.Config.Generations = 1;
        Operon::RandomGenerator rng { 3 };
        auto recording = fix.MakeGp();
        PrepareParents(*recording, fix.Ds);
        auto* const recorder = run.StartGenerationLog(result, false, 1);
        REQUIRE(recorder != nullptr);
        WriteRecord(*recorder, *recording);
        {
            auto gp = fix.MakeGp();
            FinalRun::Watch const watch { run, *gp };
            gp->Run(rng, nullptr, /*threads=*/1);
        }
        CHECK(run.Finish(0) == 0);
        auto const text = ReadFile(path);
        CHECK(Contains(text, R"("exit_status":"ok")"));
        CHECK(Contains(text,
            R"("segments":[{"segment_id":0,"first_seq":0,"last_seq":0,"count":1,"sha256":")"
                + Operon::Sha256FileHex(stream / "generation-000000.jsonl")
                + R"("}],"record_count":1,"instrumentation_errors":0})" + "\n"));
        CHECK_FALSE(Contains(text, "producer"));
    }

    std::filesystem::remove_all(dir);
}

#if !defined(_WIN32)
TEST_CASE("WriteFileAtomically reports a parent directory that cannot be synced", "[final-record]")
{
    namespace fs = std::filesystem;
    auto const dir = ScratchDir("atomic-dirsync");
    auto const path = dir / "final.json";

    // Write and search permission but no read permission: creating the temporary file and renaming succeed, opening
    // the directory to fsync it does not.
    fs::permissions(dir, fs::perms::owner_write | fs::perms::owner_exec, fs::perm_options::replace);
    auto const restore = [&] { fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace); };
    if (auto const fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); fd >= 0) { // NOLINT
        ::close(fd);
        restore();
        fs::remove_all(dir);
        SKIP("directory permissions are not enforced for this user (running as root?)");
    }

    CHECK_THROWS_WITH(Operon::Cli::WriteFileAtomically(path, "published\n"),
        Catch::Matchers::ContainsSubstring("could not open") && Catch::Matchers::ContainsSubstring("published"));
    restore();
    // The error is truthful: the file was renamed into place (not removed), no temporary file is left behind, and
    // the call did not report success.
    CHECK(ReadFile(path) == "published\n");
    std::size_t entries = 0;
    for ([[maybe_unused]] auto const& e : fs::directory_iterator(dir)) {
        ++entries;
    }
    CHECK(entries == 1);
    fs::remove_all(dir);
}

TEST_CASE("FinalRun reports failure when the final record's directory cannot be synced", "[final-record]")
{
    namespace fs = std::filesystem;
    auto const dir = ScratchDir("run-dirsync");
    auto const path = dir / "final.json";
    auto const result
        = ParseFinalOptions({ "--final-json", path.string(), "--run-id", RunId, "--attempt-id", AttemptId });
    StopFixture fix;
    FinalRun run { result };
    run.SetConfig(fix.Config);

    fs::permissions(dir, fs::perms::owner_write | fs::perms::owner_exec, fs::perm_options::replace);
    auto const restore = [&] { fs::permissions(dir, fs::perms::owner_all, fs::perm_options::replace); };
    if (auto const fd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); fd >= 0) { // NOLINT
        ::close(fd);
        restore();
        fs::remove_all(dir);
        SKIP("directory permissions are not enforced for this user (running as root?)");
    }
    CHECK(run.Finish(0) == EXIT_FAILURE);
    restore();
    fs::remove_all(dir);
}
#endif

TEST_CASE("SIGTERM stops the run gracefully and gets its own exit status", "[final-record][signal]")
{
    auto const dir = ScratchDir("signal");
    auto const path = dir / "final.json";
    auto const result
        = ParseFinalOptions({ "--final-json", path.string(), "--run-id", RunId, "--attempt-id", AttemptId });
    StopFixture fix;
    Operon::RandomGenerator rng { 3 };
    fix.Config.Generations = 1'000'000;

    FinalRun run { result };
    std::size_t reports = 0;
    {
        auto gp = fix.MakeGp();
        auto* const raw = gp.get();
        FinalRun::Watch const watch { run, *gp };
        gp->Run(
            rng,
            [&] {
                Operon::Cli::StopSignals::Forward(*raw);
                if (++reports == 3) {
                    std::raise(SIGTERM); // the handler runs synchronously in this thread
                }
                return false;
            },
            /*threads=*/1);
        CHECK(gp->StopReasons() == Mask(Operon::StopReason::Signal));
        CHECK(gp->Generation() == 2); // reports: initialization, generation 1, generation 2 (signal raised)
    }
    CHECK(Operon::Cli::StopSignals::Received() == SIGTERM);
    CHECK(run.Finish(0) == 128 + SIGTERM);

    auto const text = ReadFile(path);
    CHECK(Contains(text, R"("exit_status":"ok")"));
    CHECK(Contains(text, R"("primary":"signal","reasons":["signal"])"));
    CHECK(Contains(text, R"("generations_completed":2,"incomplete_generation":null)"));
    std::filesystem::remove_all(dir);
}

TEST_CASE("A signal that arrives before the run is forwarded once there is a target", "[final-record][signal]")
{
    Operon::Cli::StopSignals::Install();
    std::raise(SIGINT);
    CHECK(Operon::Cli::StopSignals::Received() == SIGINT);
    StopFixture fix;
    auto gp = fix.MakeGp();
    CHECK_FALSE(gp->StopRequested());
    Operon::Cli::StopSignals::Forward(*gp);
    CHECK(gp->StopReasons() == Mask(Operon::StopReason::Signal));
    CHECK(Operon::Cli::StopSignals::SignalExitCode(SIGINT) == 130);
    Operon::Cli::StopSignals::Uninstall();
    CHECK(Operon::Cli::StopSignals::Received() == 0);
}

TEST_CASE("ModelRef identifies a model by every node and coefficient", "[sha256][model-ref]")
{
    auto ds = MakeDataset();
    auto tree = Operon::InfixParser::ParseOrThrow("X1 * 2 + 0.5", ds);
    auto const ref = Operon::ModelRef(tree);
    CHECK(ref.size() == 64);
    CHECK(ref == Operon::ModelRef(Operon::InfixParser::ParseOrThrow("X1 * 2 + 0.5", ds)));
    CHECK(ref == Operon::Sha256Hex(Operon::ModelSerialization(tree)));
    CHECK(Operon::ModelSerialization(tree).starts_with("operon.model.v1\nnodes "));

    SECTION("a different structure, symbol or coefficient changes it")
    {
        CHECK(ref != Operon::ModelRef(Operon::InfixParser::ParseOrThrow("X1 * 2 - 0.5", ds)));
        CHECK(ref != Operon::ModelRef(Operon::InfixParser::ParseOrThrow("X1 * 2", ds)));
        CHECK(ref != Operon::ModelRef(Operon::InfixParser::ParseOrThrow("X1 * 2 + 0.5000001", ds)));
    }

    SECTION("the last bit of a coefficient matters")
    {
        auto changed = tree;
        for (auto& node : changed.Nodes()) {
            if (node.IsConstant()) {
                node.Value = std::nextafter(node.Value, std::numeric_limits<Operon::Scalar>::max());
                break;
            }
        }
        CHECK(ref != Operon::ModelRef(changed));
    }

    SECTION("enabled and optimize flags matter")
    {
        auto changed = tree;
        changed.Nodes().front().Optimize = !changed.Nodes().front().Optimize;
        CHECK(ref != Operon::ModelRef(changed));
    }
}

} // namespace Operon::Test
