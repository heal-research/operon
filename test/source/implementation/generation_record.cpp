// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#include <algorithm>
#include <catch2/catch_test_macros.hpp>
#include <catch2/matchers/catch_matchers_string.hpp>
#include <filesystem>
#include <fstream>
#include <glaze/glaze.hpp>
#include <limits>
#include <numeric>
#include <sstream>
#include <thread>
#if !defined(_WIN32)
#include <sys/resource.h>
#endif
#include <csignal>
#if defined(_WIN32)
#include <process.h>
#else
#include <unistd.h>
#endif

#include "operon/algorithms/generation_record.hpp"
#include "operon/core/constraint.hpp"
#include "operon/hash/model_ref.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/parser/infix.hpp"
#include "stop_fixture.hpp"

namespace Operon::Test {
namespace {
    auto ProcessId() -> int
    {
#if defined(_WIN32)
        return ::_getpid();
#else
        return static_cast<int>(::getpid());
#endif
    }

    struct RecordingFixture : StopFixture {
        std::filesystem::path Directory;
        GenerationRecorderOptions Options;
        std::unique_ptr<GeneticProgrammingAlgorithm> Gp;
        explicit RecordingFixture(std::string const& name)
            : Directory(std::filesystem::temp_directory_path()
                  / ("operon-generation-" + name + "-" + std::to_string(ProcessId())))
        {
            std::filesystem::remove_all(Directory);
            Options.Directory = Directory;
            Options.RunId = std::string(64, 'a');
            Options.AttemptId = "11111111-2222-4333-8444-555555555555";
            Options.ObjectiveNames = { "mse" };
            Gp = MakeGp();
            std::size_t i = 0;
            for (auto& ind : Gp->Parents()) {
                ind.Genotype = InfixParser::ParseOrThrow(i % 2 ? "X1" : "X1 * 2 + 1", Ds);
                ind.Fitness[0] = static_cast<Scalar>(++i);
            }
        }
        ~RecordingFixture() { std::filesystem::remove_all(Directory); }
        auto Write(GenerationRecorder& recorder) -> void
        {
            recorder.Capture(*Gp);
            recorder.Write(*Gp, Gp->Parents()[0], 0.25, 0.5, 1.0);
            ++Gp->Generation();
        }
    };
    auto Contents(std::filesystem::path const& path) -> std::string
    {
        std::ifstream in(path);
        std::ostringstream out;
        out << in.rdbuf();
        return out.str();
    }
    auto Lines(std::filesystem::path const& path) -> std::vector<std::string>
    {
        std::ifstream in(path);
        std::vector<std::string> lines;
        std::string line;
        while (std::getline(in, line)) {
            lines.push_back(line);
        }
        return lines;
    }
}

TEST_CASE("generation recorder seals by count and preserves exact byte hashes", "[generation-record]")
{
    RecordingFixture f("count");
    f.Options.SegmentGenerations = 2;
    GenerationRecorder recorder(f.Options);
    f.Write(recorder);
    CHECK(std::filesystem::exists(f.Directory / "generation-000000.jsonl.part"));
    CHECK_FALSE(std::filesystem::exists(f.Directory / "generation-000000.jsonl"));
    auto const first = Contents(f.Directory / "generation-000000.jsonl.part");
    CHECK_FALSE(glz::validate_json(first));
    CHECK(first.find("\"seq\":0,\"generation\":0,\"event\":\"initialized\"") != std::string::npos);
    f.Write(recorder);
    recorder.Drain();
    CHECK_FALSE(std::filesystem::exists(f.Directory / "generation-000000.jsonl.part"));
    REQUIRE(recorder.Segments().size() == 1);
    auto const& s = recorder.Segments()[0];
    CHECK(s.FirstSeq == 0);
    CHECK(s.LastSeq == 1);
    CHECK(s.Count == 2);
    CHECK(s.Sha256 == Sha256FileHex(f.Directory / "generation-000000.jsonl"));
    f.Write(recorder);
    recorder.Close();
    recorder.Close();
    REQUIRE(recorder.Segments().size() == 2);
    CHECK(recorder.RecordCount() == 3);
    CHECK(recorder.Errors() == 0);
    CHECK(recorder.Segments()[1].FirstSeq == 2);
    CHECK(recorder.Segments()[1].Count == 1);
    CHECK_FALSE(std::filesystem::exists(f.Directory / "generation-000001.jsonl.part"));
}

TEST_CASE("generation recorder seals by elapsed time", "[generation-record]")
{
    RecordingFixture f("time");
    f.Options.SegmentGenerations = 100;
    f.Options.SegmentSeconds = 0.001;
    GenerationRecorder recorder(f.Options);
    f.Write(recorder);
    std::this_thread::sleep_for(std::chrono::milliseconds(3));
    f.Write(recorder);
    CHECK_FALSE(recorder.Segments().empty());
    recorder.Close();
    CHECK(recorder.RecordCount() == 2);
}

TEST_CASE("generation record reductions equal independent population calculations", "[generation-record]")
{
    RecordingFixture f("reductions");
    f.Options.Nsga2 = true;
    auto pop = f.Gp->Parents();
    pop[0].Fitness[0] = std::numeric_limits<Scalar>::quiet_NaN();
    pop[1].Fitness[0] = std::numeric_limits<Scalar>::infinity();
    pop[2].Fitness[0] = -std::numeric_limits<Scalar>::infinity();
    pop[3].Fitness[0] = EvaluatorBase::ErrMax;
    pop[4].Fitness[0] = Scalar { 1 };
    pop[4].ShapeRejected = true;
    pop[5].Fitness[0] = Scalar { 1 }; // same value, genuinely accepted: no value-based shape inference
    for (std::size_t i = 0; i < pop.size(); ++i) {
        pop[i].Rank = i % 3;
    }
    GenerationRecorder recorder(f.Options);
    recorder.Capture(*f.Gp);
    auto const& r = recorder.Record();
    auto const& o = r.Objectives[0];
    CHECK(o.Finite == 7);
    CHECK(o.Rejected == 2);
    CHECK(o.Nan == 1);
    CHECK(o.PosInf == 1);
    CHECK(o.NegInf == 1);
    double sum = 0;
    std::vector<std::size_t> lengths, depths;
    for (auto const& ind : pop) {
        if (std::isfinite(ind[0]) && ind[0] != EvaluatorBase::ErrMax && !ind.ShapeRejected) {
            sum += ind[0];
        }
        lengths.push_back(ind.Genotype.Length());
        depths.push_back(ind.Genotype.Depth());
    }
    CHECK(o.Sum == sum);
    CHECK(o.Min == 1);
    CHECK(o.Max == 12);
    CHECK(r.Length.Sum == std::accumulate(lengths.begin(), lengths.end(), 0.0));
    CHECK(r.Length.Min == *std::min_element(lengths.begin(), lengths.end()));
    CHECK(r.Length.Max == *std::max_element(lengths.begin(), lengths.end()));
    CHECK(r.Depth.Sum == std::accumulate(depths.begin(), depths.end(), 0.0));
    CHECK(r.Depth.Min == *std::min_element(depths.begin(), depths.end()));
    CHECK(r.Depth.Max == *std::max_element(depths.begin(), depths.end()));
    CHECK(r.Ranks == std::vector<std::size_t> { 4, 4, 4 });
    recorder.Write(
        *f.Gp, pop[0], std::numeric_limits<double>::quiet_NaN(), std::numeric_limits<double>::infinity(), 0.5);
    recorder.Close();
    auto const text = Contents(f.Directory / "generation-000000.jsonl");
    CHECK_FALSE(glz::validate_json(text));
    CHECK(text.find("\"finite_count\":7,\"rejected_count\":2,\"nan_count\":1,\"posinf_count\":1,\"neginf_count\":1")
        != std::string::npos);
    CHECK(text.find("\"nmse_train\":null,\"nmse_validation\":null,\"nmse_test\":null,\"metric_status\":\"nan\"")
        != std::string::npos);
    CHECK(text.find("\"objectives\":[null]") != std::string::npos);
    CHECK(text.find(":nan") == std::string::npos);
    CHECK(text.find(":inf") == std::string::npos);
    CHECK(text.find("\"rank_histogram\":[4,4,4]") != std::string::npos);
}

TEST_CASE("generation objective aggregates use consistent native JSON precision", "[generation-record]")
{
    RecordingFixture f("precision");
    auto const value = static_cast<Scalar>(0.00041120723471976817);
    for (auto& ind : f.Gp->Parents()) {
        ind.Fitness[0] = value;
    }
    GenerationRecorder recorder(f.Options);
    f.Write(recorder);
    recorder.Close();
    auto const text = Contents(f.Directory / "generation-000000.jsonl");
    auto const number = [&](std::string const& key) {
        auto const start = text.find("\"" + key + "\":");
        REQUIRE(start != std::string::npos);
        return std::stod(text.substr(start + key.size() + 3));
    };
    CHECK(number("best") == number("mean"));
    CHECK(number("min") == number("mean"));
    CHECK(number("max") == number("mean"));
    CHECK(static_cast<Scalar>(number("mean")) == value);
}

TEST_CASE("generation model sidecar emits only changed lossless objects", "[generation-record]")
{
    RecordingFixture f("models");
    GenerationRecorder recorder(f.Options);
    auto const ref = ModelRef(f.Gp->Parents()[0].Genotype);
    f.Write(recorder);
    f.Write(recorder);
    f.Gp->Parents()[0].Genotype = InfixParser::ParseOrThrow("X1 + 0.100000001", f.Ds);
    f.Write(recorder);
    recorder.Close();
    auto const models = Lines(f.Directory / "models.jsonl.txt");
    REQUIRE(models.size() == 2);
    CHECK(models[0].find(ref) != std::string::npos);
    for (auto const& line : models) {
        CHECK_FALSE(glz::validate_json(line));
    }
    auto const records = Lines(f.Directory / "generation-000000.jsonl");
    REQUIRE(records.size() == 3);
    CHECK(records[0].find("\"model_object_emitted\":true") != std::string::npos);
    CHECK(records[1].find("\"model_object_emitted\":false") != std::string::npos);
    CHECK(records[2].find("\"model_object_emitted\":true") != std::string::npos);
}

TEST_CASE("generation log failures throw instead of producing a successful stream", "[generation-record]")
{
    RecordingFixture f("errors");
    SECTION("regular file in place of the directory")
    {
        std::filesystem::create_directories(f.Directory);
        auto const file = f.Directory / "not-a-directory";
        std::ofstream { file } << "x";
        REQUIRE(std::filesystem::is_regular_file(file));
        f.Options.Directory = file;
        CHECK_THROWS_AS(GenerationRecorder { f.Options }, std::system_error);
    }
    SECTION("invalid segment settings")
    {
        f.Options.SegmentGenerations = 0;
        CHECK_THROWS(GenerationRecorder { f.Options });
    }
    SECTION("segment creation failure after initialization")
    {
        GenerationRecorder recorder(f.Options);
        std::filesystem::create_directory(f.Directory / "generation-000000.jsonl.part");
        recorder.Capture(*f.Gp);
        CHECK_THROWS(recorder.Write(*f.Gp, f.Gp->Parents()[0], 0, 0, 1));
        CHECK(recorder.Errors() == 1);
        CHECK(recorder.RecordCount() == 0);
    }
    SECTION("segment publication failure is surfaced")
    {
        GenerationRecorder recorder(f.Options);
        f.Write(recorder);
        std::filesystem::create_directory(f.Directory / "generation-000000.jsonl");
        CHECK_THROWS(recorder.Close());
        CHECK(recorder.Errors() == 1);
        // A failed close is terminal: it never turns into a later success, and nothing else can be recorded.
        CHECK_THROWS(recorder.Close());
        CHECK_THROWS(recorder.Close());
        CHECK(recorder.Errors() == 1);
        CHECK_THROWS(recorder.Capture(*f.Gp));
        CHECK_THROWS(recorder.Write(*f.Gp, f.Gp->Parents()[0], 0, 0, 1));
    }
    SECTION("asynchronous seal errors are rethrown on close")
    {
        f.Options.SegmentGenerations = 1;
        GenerationRecorder recorder(f.Options);
        std::filesystem::create_directory(f.Directory / "generation-000000.jsonl");
        f.Write(recorder);
        CHECK_THROWS(recorder.Close());
        CHECK(recorder.Errors() == 1);
        CHECK(recorder.RecordCount() == 1);
        CHECK_THROWS(recorder.Close());
        CHECK(recorder.Errors() == 1);
    }
    SECTION("existing stream cannot be overwritten")
    {
        {
            GenerationRecorder recorder(f.Options);
            f.Write(recorder);
            recorder.Close();
        }
        CHECK_THROWS(GenerationRecorder { f.Options });
    }
}

#if !defined(_WIN32)
namespace {
    // Caps the size any file of this process may grow to. Writes that cross the cap are first shortened, then fail
    // with EFBIG, which makes a partial record deterministic. SIGXFSZ would otherwise terminate the process.
    struct FileSizeLimit {
        rlimit Saved {};
        void (*Handler)(int) { nullptr };
        explicit FileSizeLimit(std::uintmax_t bytes)
        {
            REQUIRE(::getrlimit(RLIMIT_FSIZE, &Saved) == 0);
            Handler = std::signal(SIGXFSZ, SIG_IGN);
            auto limit = Saved;
            limit.rlim_cur = static_cast<rlim_t>(bytes);
            REQUIRE(::setrlimit(RLIMIT_FSIZE, &limit) == 0);
        }
        ~FileSizeLimit()
        {
            ::setrlimit(RLIMIT_FSIZE, &Saved);
            std::signal(SIGXFSZ, Handler);
        }
        FileSizeLimit(FileSizeLimit const&) = delete;
        auto operator=(FileSizeLimit const&) -> FileSizeLimit& = delete;
    };
}

TEST_CASE("generation recorder never seals a partially written record", "[generation-record]")
{
    RecordingFixture f("partial-write");
    GenerationRecorder recorder(f.Options);
    f.Write(recorder);
    auto const part = f.Directory / "generation-000000.jsonl.part";
    auto const complete = std::filesystem::file_size(part);
    auto const first = Contents(part);
    auto const models = f.Directory / "models.jsonl.txt";
    auto const originalModels = Contents(models);
    f.Gp->Parents()[0].Genotype = InfixParser::ParseOrThrow("X1", f.Ds);
    recorder.Capture(*f.Gp);
    {
        // The second record is about as long as the first, so half a record fits and the rest fails.
        FileSizeLimit const limit(complete + complete / 2);
        CHECK_THROWS_AS(recorder.Write(*f.Gp, f.Gp->Parents()[0], 0.25, 0.5, 1.0), std::system_error);
    }
    CHECK(recorder.Errors() == 1);
    CHECK(recorder.RecordCount() == 1);
    // The partial bytes were removed again: the open segment still holds exactly the one hashed record.
    CHECK(std::filesystem::file_size(part) == complete);
    CHECK(Contents(models) == originalModels);
    CHECK_THROWS_WITH(
        recorder.Write(*f.Gp, f.Gp->Parents()[0], 0.25, 0.5, 1.0), "generation recorder write after failure");
    CHECK_NOTHROW(recorder.Close());
    CHECK(recorder.Errors() == 1);
    REQUIRE(recorder.Segments().size() == 1);
    auto const& s = recorder.Segments()[0];
    CHECK(s.Count == 1);
    CHECK(s.FirstSeq == 0);
    CHECK(s.LastSeq == 0);
    auto const sealed = f.Directory / "generation-000000.jsonl";
    CHECK_FALSE(std::filesystem::exists(part));
    CHECK(Contents(sealed) == first);
    CHECK(s.Sha256 == Sha256FileHex(sealed));
}

TEST_CASE("generation recorder removes an empty segment after a failed first write", "[generation-record]")
{
    RecordingFixture f("empty-failed-write");
    GenerationRecorder recorder(f.Options);
    recorder.Capture(*f.Gp);
    {
        FileSizeLimit const limit(0);
        CHECK_THROWS_AS(recorder.Write(*f.Gp, f.Gp->Parents()[0], 0.25, 0.5, 1.0), std::system_error);
    }
    CHECK_NOTHROW(recorder.Close());
    CHECK(recorder.Segments().empty());
    CHECK_FALSE(std::filesystem::exists(f.Directory / "generation-000000.jsonl.part"));
    CHECK_FALSE(std::filesystem::exists(f.Directory / "generation-000000.jsonl"));
    CHECK(Contents(f.Directory / "models.jsonl.txt").empty());
}
#endif

TEST_CASE("generation record ranks are bounded by the combined parent and offspring count", "[generation-record]")
{
    // Retained parents keep ranks from the parents+offspring sort, e.g. offspring that replace dominated parents.
    RecordingFixture f("ranks");
    f.Options.Nsga2 = true;
    auto pop = f.Gp->Parents();
    auto const combined = f.Gp->Individuals().size();
    REQUIRE(combined > pop.size() + 3);
    pop[0].Rank = pop.size() + 3;
    GenerationRecorder recorder(f.Options);
    REQUIRE_NOTHROW(recorder.Capture(*f.Gp));
    auto const& ranks = recorder.Record().Ranks;
    REQUIRE(ranks.size() == pop.size() + 4);
    CHECK(ranks[0] == pop.size() - 1);
    CHECK(ranks[pop.size() + 3] == 1);
    {
        RecordingFixture invalid("invalid-rank");
        invalid.Options.Nsga2 = true;
        invalid.Gp->Parents()[1].Rank = invalid.Gp->Individuals().size();
        GenerationRecorder invalidRecorder(invalid.Options);
        CHECK_THROWS_WITH(invalidRecorder.Capture(*invalid.Gp), "invalid population rank");
        CHECK(invalidRecorder.Errors() == 1);
    }
}

TEST_CASE("generation shape validity is certified under the canonical scaling transform", "[generation-record]")
{
    // y = 2 * X1 on X1 in [1, 10]. The raw model X1 stays inside [-0.5, 10.5], but its canonical linear-scaling
    // transform (slope 2) reaches [2, 20], which is what the shape-constrained evaluator scored.
    RecordingFixture f("shape-scaling");
    ShapeConstraintSet constraints;
    constraints.Domains.insert_or_assign("X1", std::pair { Scalar { 1 }, Scalar { 10 } });
    constraints.Constraints.push_back({ .Op = ShapeConstraintOp::Identity,
        .Variable = "",
        .Sign = std::nullopt,
        .Bound = std::pair { static_cast<Scalar>(-0.5), static_cast<Scalar>(10.5) } });
    ShapeConstrainedValueEvaluator<Evaluator<DTable>> shapeEvaluator(
        ShapeConstraintSpec { .Constraints = constraints }, &f.Dtable, &f.Problem, &f.Dtable, MSE {});
    BasicOffspringGenerator generator(&shapeEvaluator, &f.Crossover, &f.Mutator, &f.FemSel, &f.MaleSel);
    auto gp = f.MakeGp(&generator);
    for (auto& ind : gp->Parents()) {
        ind.Genotype = InfixParser::ParseOrThrow("X1", f.Ds);
        ind.Fitness[0] = Scalar { 1 };
    }

    auto const& tree = gp->Parents()[0].Genotype;
    REQUIRE(shapeEvaluator.Certifier().Measure(tree, std::nullopt).Feasible);
    REQUIRE_FALSE(shapeEvaluator.Certifier().MeasureCanonical(tree).Feasible);

    GenerationRecorder recorder(f.Options);
    recorder.Capture(*gp);
    recorder.Write(*gp, gp->Parents()[0], 0.25, 0.5, 1.0);
    recorder.Close();
    auto const text = Contents(f.Directory / "generation-000000.jsonl");
    CHECK(text.find("\"shape_feasible\":false") != std::string::npos);
    CHECK(text.find("\"shape_feasible\":true") == std::string::npos);
}
} // namespace Operon::Test
