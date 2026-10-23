// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "final_record.hpp"

#include <algorithm>
#include <array>
#include <cerrno>
#include <cmath>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fmt/core.h>
#include <fmt/format.h>
#include <fstream>
#include <random>
#include <stdexcept>

#include "operon/hash/model_ref.hpp"
#include "operon/operators/generator.hpp"
#include "stop_signal.hpp"

#if !defined(_WIN32)
#include <fcntl.h>
#include <unistd.h>
#endif

namespace Operon::Cli {

namespace {

    // ---- JSON text -------------------------------------------------------------------------------------------

    auto Quote(std::string_view text) -> std::string
    {
        std::string out;
        out.reserve(text.size() + 2);
        out.push_back('"');
        for (auto const ch : text) {
            auto const c = static_cast<unsigned char>(ch);
            switch (c) {
            case '"':
                out += "\\\"";
                break;
            case '\\':
                out += "\\\\";
                break;
            case '\n':
                out += "\\n";
                break;
            case '\r':
                out += "\\r";
                break;
            case '\t':
                out += "\\t";
                break;
            default:
                if (c < 0x20U) {
                    out += fmt::format("\\u{:04x}", static_cast<unsigned>(c));
                } else {
                    out.push_back(ch); // UTF-8 passes through unchanged
                }
            }
        }
        out.push_back('"');
        return out;
    }

    auto Number(double value) -> std::string
    {
        return std::isfinite(value) ? fmt::format("{}", value) : std::string { "null" };
    }

    auto Uint(std::size_t value) -> std::string { return fmt::format("{}", value); }

    template <typename T, typename F> auto OrNull(std::optional<T> const& value, F&& format) -> std::string
    {
        return value ? std::forward<F>(format)(*value) : std::string { "null" };
    }

    auto CountersJson(Operon::EvaluationStats const& s) -> std::string
    {
        auto const misses = (s.CacheHits && s.CacheLookups)
            ? std::optional<std::size_t> { *s.CacheLookups >= *s.CacheHits ? *s.CacheLookups - *s.CacheHits : 0 }
            : std::nullopt;
        return fmt::format(
            R"({{"evaluator_calls":{},"residual_scoring":{},"residual_optimizer":{},"jacobian_scoring":{},)"
            R"("jacobian_optimizer":{},"residual_total":{},"jacobian_total":{},"cache_hits":{},"cache_misses":{},)"
            R"("optimizer_seconds":{}}})",
            s.Calls, s.ResidualScoring, s.ResidualOptimizer, s.JacobianScoring, s.JacobianOptimizer, s.ResidualTotal(),
            s.JacobianTotal(), OrNull(s.CacheHits, Uint), OrNull(misses, Uint),
            Number(static_cast<double>(s.OptimizerTimeUs) / 1e6));
    }

    auto Unlimited(std::size_t value) -> std::optional<std::size_t>
    {
        return static_cast<std::uint64_t>(value) >= UnlimitedThreshold ? std::nullopt : std::optional { value };
    }

    // ---- option values ---------------------------------------------------------------------------------------

    auto IsLowerHex(std::string_view s) -> bool
    {
        return std::ranges::all_of(s, [](char c) { return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f'); });
    }

    auto IsRunId(std::string_view s) -> bool { return s.size() == 64 && IsLowerHex(s); }

    auto IsUuid(std::string_view s) -> bool
    {
        if (s.size() != 36) {
            return false;
        }
        constexpr std::array<std::size_t, 4> dashes { 8, 13, 18, 23 };
        for (std::size_t i = 0; i < s.size(); ++i) {
            auto const isDash = std::ranges::find(dashes, i) != dashes.end();
            if (isDash ? s[i] != '-' : !IsLowerHex(s.substr(i, 1))) {
                return false;
            }
        }
        return true;
    }

} // namespace

// ------------------------------------------------------------------------------------------------------------------
// Snapshots

auto MakeFinalRecord(Operon::GeneticAlgorithmBase const& algorithm) -> FinalRecord
{
    auto const config = algorithm.GetConfig();
    auto const* evaluator = algorithm.GetGenerator()->Evaluator();

    FinalRecord record;
    record.Reasons = algorithm.TerminationReasons();
    record.GenerationsCompleted = algorithm.GenerationsCompleted();
    record.Incomplete = algorithm.IncompleteGeneration();
    record.GenerationLimit = Unlimited(config.Generations);
    record.EvaluationLimit = Unlimited(evaluator->Budget());
    record.TimeLimitSeconds = Unlimited(config.TimeLimit).transform([](auto v) { return static_cast<double>(v); });
    record.ElapsedSeconds = algorithm.Elapsed();
    if (auto const* os = dynamic_cast<Operon::OffspringSelectionGenerator const*>(algorithm.GetGenerator())) {
        record.PressureMax = os->MaxSelectionPressure();
        record.PressureLast = os->SelectionPressure();
    }
    record.Stats = algorithm.DetailedStats();
    return record;
}

auto MakeFinalRecord(Operon::GeneticAlgorithmConfig const& config) -> FinalRecord
{
    FinalRecord record;
    record.GenerationLimit = Unlimited(config.Generations);
    record.EvaluationLimit = Unlimited(config.Evaluations);
    record.TimeLimitSeconds = Unlimited(config.TimeLimit).transform([](auto v) { return static_cast<double>(v); });
    return record;
}

// ------------------------------------------------------------------------------------------------------------------
// Document

auto SerializeFinalRecord(FinalRecord const& r) -> std::string
{
    // Precedence order (D12), so `primary` is the first entry.
    std::string reasons;
    std::string primary { "null" };
    for (auto const reason : Operon::StopPrecedence) {
        if (!Operon::Contains(r.Reasons, reason)) {
            continue;
        }
        auto const name = Quote(Operon::ToString(reason));
        if (primary == "null") {
            primary = name;
        }
        reasons += (reasons.empty() ? "" : ",") + name;
    }

    auto const consumed = r.Stats.Consumed();
    std::optional<std::size_t> overshoot;
    if (r.EvaluationLimit) {
        overshoot = consumed > *r.EvaluationLimit ? consumed - *r.EvaluationLimit : 0;
    }
    auto const counters = CountersJson(r.Stats);

    auto const incomplete = OrNull(r.Incomplete, [](Operon::OffspringProgress const& p) {
        return fmt::format(R"({{"attempted_offspring":{},"completed_offspring":{},"pool_count":{}}})", p.Attempted,
            p.Completed, p.PoolSize);
    });
    auto const model = OrNull(r.Model, [](FinalModel const& m) {
        return fmt::format(R"({{"selection_rule":{},"model_ref":{},"infix":{}}})", Quote(m.SelectionRule),
            Quote(m.ModelRef), Quote(m.Infix));
    });

    std::string producer;
    if (!r.UnsupportedFields.empty()) {
        std::string fields;
        for (auto const& f : r.UnsupportedFields) {
            fields += (fields.empty() ? "" : ",") + Quote(f);
        }
        producer = fmt::format(
            R"(,"producer":{{"adapter":"operon","unsupported_fields":[{}],"derived_fields":[]}})", fields);
    }

    std::string segments;
    for (auto const& s : r.Segments) {
        if (!segments.empty()) {
            segments += ',';
        }
        segments += fmt::format(R"({{"segment_id":{},"first_seq":{},"last_seq":{},"count":{},"sha256":{}}})", s.Id,
            s.FirstSeq, s.LastSeq, s.Count, Quote(s.Sha256));
    }
    return fmt::format(
        R"({{"schema":"operon.final","schema_version":"1.1","run_id":{},"attempt_id":{},"exit_status":{},)"
        R"("termination":{{"primary":{},"reasons":[{}],"precedence_version":{}}},)"
        R"("generations_completed":{},"incomplete_generation":{},)"
        R"("budgets":{{"generations":{{"limit":{}}},)"
        R"("evaluations":{{"unit":"residual+jacobian","limit":{},"consumed":{},"overshoot":{}}},)"
        R"("time":{{"limit_s":{},"elapsed_s":{}}},"pressure":{{"max":{},"last":{}}}}},)"
        R"("counters":{{"logical":{},"since_restart":{}}},"final_model":{},"front_ref":{},)"
        R"("segments":[{}],"record_count":{},"instrumentation_errors":{}{}}})"
        "\n",
        Quote(r.RunId), Quote(r.AttemptId), r.Error ? R"("error")" : R"("ok")", primary, reasons,
        Operon::StopPrecedenceVersion, r.GenerationsCompleted, incomplete, OrNull(r.GenerationLimit, Uint),
        OrNull(r.EvaluationLimit, Uint), consumed, OrNull(overshoot, Uint), OrNull(r.TimeLimitSeconds, Number),
        Number(r.ElapsedSeconds), OrNull(r.PressureMax, Uint), OrNull(r.PressureLast, Number), counters, counters,
        model, OrNull(r.FrontRef, Quote), segments, r.RecordCount, r.InstrumentationErrors, producer);
}

auto WriteFileAtomically(std::filesystem::path const& path, std::string_view content) -> void
{
    auto tmp = path;
    tmp += fmt::format(".tmp-{}-{}",
        static_cast<unsigned long long>(
#if defined(_WIN32)
            0
#else
            ::getpid()
#endif
            ),
        std::random_device {}());

#if defined(_WIN32)
    {
        std::ofstream out(tmp, std::ios::binary | std::ios::trunc);
        out.write(content.data(), static_cast<std::streamsize>(content.size()));
        out.close();
        if (!out) {
            std::filesystem::remove(tmp);
            throw std::runtime_error(fmt::format("could not write {}", tmp.string()));
        }
    }
#else
    auto const fd = ::open(tmp.c_str(), O_WRONLY | O_CREAT | O_TRUNC | O_CLOEXEC, 0644); // NOLINT
    if (fd < 0) {
        throw std::runtime_error(fmt::format("could not create {}: {}", tmp.string(), std::strerror(errno)));
    }
    auto fail = [&](char const* what) {
        auto const err = errno;
        ::close(fd);
        std::filesystem::remove(tmp);
        throw std::runtime_error(fmt::format("{} {}: {}", what, tmp.string(), std::strerror(err)));
    };
    std::size_t written = 0;
    while (written < content.size()) {
        auto const n = ::write(fd, content.data() + written, content.size() - written);
        if (n < 0) {
            if (errno == EINTR) {
                continue;
            }
            fail("could not write");
        }
        written += static_cast<std::size_t>(n);
    }
    if (::fsync(fd) != 0) {
        fail("could not sync");
    }
    if (::close(fd) != 0) {
        std::filesystem::remove(tmp);
        throw std::runtime_error(fmt::format("could not close {}: {}", tmp.string(), std::strerror(errno)));
    }
#endif

    std::error_code ec;
    std::filesystem::rename(tmp, path, ec);
    if (ec) {
        std::filesystem::remove(tmp);
        throw std::runtime_error(fmt::format("could not publish {}: {}", path.string(), ec.message()));
    }
#if !defined(_WIN32)
    // Make the rename itself durable.
    auto dir = path.parent_path();
    if (dir.empty()) {
        dir = ".";
    }
    if (auto const dfd = ::open(dir.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); dfd >= 0) { // NOLINT
        static_cast<void>(::fsync(dfd));
        ::close(dfd);
    }
#endif
}

// ------------------------------------------------------------------------------------------------------------------
// CLI glue

FinalRun::FinalRun(cxxopts::ParseResult const& result, Operon::GenerationRecorder::Clock::time_point processStart)
    : processStart_(processStart)
    , streamRequested_(result.contains("generation-log"))
{
    if (!result.contains("final-json") && !streamRequested_) {
        if (result.contains("run-id") || result.contains("attempt-id")) {
            throw std::invalid_argument("--run-id and --attempt-id require --final-json or --generation-log");
        }
        return;
    }
    if (!result.contains("run-id") || !result.contains("attempt-id")) {
        throw std::invalid_argument("--final-json and --generation-log require --run-id and --attempt-id");
    }
    runId_ = result["run-id"].as<std::string>();
    attemptId_ = result["attempt-id"].as<std::string>();
    if (!IsRunId(runId_)) {
        throw std::invalid_argument("--run-id must be 64 lowercase hexadecimal characters");
    }
    if (!IsUuid(attemptId_)) {
        throw std::invalid_argument("--attempt-id must be a lowercase UUID (8-4-4-4-12 hexadecimal digits)");
    }
    if (result.contains("final-json")) {
        path_ = result["final-json"].as<std::string>();
    }
    enabled_ = true;
    StopSignals::Install();
}

auto FinalRun::StartGenerationLog(cxxopts::ParseResult const& result, bool nsga2, std::size_t objectiveCount)
    -> Operon::GenerationRecorder*
{
    if (!streamRequested_) {
        return nullptr;
    }
    try {
        if (result.contains("resume")) {
            throw std::invalid_argument("--generation-log requires a fresh run; --resume is unsupported");
        }
        auto const sync = result["sync"].as<std::string>();
        if (sync != "record" && sync != "segment") {
            throw std::invalid_argument("--sync must be record or segment");
        }
        Operon::GenerationRecorderOptions options;
        options.Directory = result["generation-log"].as<std::string>();
        options.RunId = runId_;
        options.AttemptId = attemptId_;
        options.ObjectiveNames.push_back(result["objective"].as<std::string>());
        if (nsga2) {
            options.ObjectiveNames.emplace_back("length");
        }
        if (objectiveCount > options.ObjectiveNames.size()) {
            options.ObjectiveNames.emplace_back("violation");
        }
        options.Generator = result["offspring-generator"].as<std::string>();
        options.Generator.resize(
            options.Generator.find(':') == std::string::npos ? options.Generator.size() : options.Generator.find(':'));
        options.SelectionRule = nsga2 ? result["model-selection"].as<std::string>() : "obj0";
        options.Nsga2 = nsga2;
        options.SegmentGenerations = result["segment-generations"].as<std::size_t>();
        options.SegmentSeconds = result["segment-seconds"].as<double>();
        options.SyncRecord = sync == "record";
        options.ProcessStart = processStart_;
        recorder_ = std::make_unique<Operon::GenerationRecorder>(std::move(options));
        return recorder_.get();
    } catch (...) {
        ++instrumentationErrors_;
        throw;
    }
}

FinalRun::~FinalRun()
{
    if (enabled_) {
        StopSignals::Uninstall();
    }
}

FinalRun::Watch::Watch(FinalRun& run, Operon::GeneticAlgorithmBase& algorithm) : run_(run), algorithm_(algorithm)
{
    if (run_.enabled_) {
        StopSignals::SetTarget(&algorithm_);
    }
}

FinalRun::Watch::~Watch()
{
    if (!run_.enabled_) {
        return;
    }
    StopSignals::SetTarget(nullptr);
    try {
        run_.snapshot_ = MakeFinalRecord(algorithm_);
    } catch (std::exception const& e) {
        fmt::print(stderr, "warning: could not collect the final counters: {}\n", e.what());
    }
}

auto FinalRun::Abandon() -> void
{
    if (enabled_) {
        StopSignals::Uninstall();
        enabled_ = false;
    }
}

auto FinalRun::SetModel(std::string selectionRule, Operon::Tree const& tree, std::string infix) -> void
{
    model_ = FinalModel { std::move(selectionRule), Operon::ModelRef(tree), std::move(infix) };
}

auto FinalRun::Finish(int exitCode) -> int
{
    if (!enabled_) {
        return exitCode;
    }
    if (recorder_) {
        try {
            recorder_->Close();
        } catch (std::exception const& e) {
            fmt::print(stderr, "error: could not seal generation log: {}\n", e.what());
            error_ = true;
            exitCode = EXIT_FAILURE;
        }
    }
    auto record = snapshot_ ? *snapshot_ : MakeFinalRecord(config_);
    record.RunId = runId_;
    record.AttemptId = attemptId_;
    record.Model = model_;
    record.FrontRef = frontRef_;
    record.InstrumentationErrors = instrumentationErrors_ + (recorder_ ? recorder_->Errors() : 0);
    if (streamRequested_) {
        if (recorder_) {
            record.Segments = recorder_->Segments();
            record.RecordCount = recorder_->RecordCount();
        }
    } else {
        record.UnsupportedFields = { "segments", "record_count" };
    }
    if (frontExpected_ && !record.FrontRef) {
        record.UnsupportedFields.emplace_back("front_ref");
    }

    auto const failed = error_ || exitCode != 0 || record.Reasons == 0 || record.InstrumentationErrors != 0;
    if (failed) {
        record.Error = true;
        record.Reasons |= Operon::MaskOf(Operon::StopReason::EngineError);
    }
    auto const signal = StopSignals::Received();
    try {
        if (!path_.empty()) {
            WriteFileAtomically(path_, SerializeFinalRecord(record));
        }
    } catch (std::exception const& e) {
        fmt::print(stderr, "error: could not write the final record: {}\n", e.what());
        return EXIT_FAILURE;
    }
    if (!failed && signal != 0 && Operon::Contains(record.Reasons, Operon::StopReason::Signal)) {
        return StopSignals::SignalExitCode(signal);
    }
    return exitCode;
}

} // namespace Operon::Cli
