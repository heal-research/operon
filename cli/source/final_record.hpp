// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CLI_FINAL_RECORD_HPP
#define OPERON_CLI_FINAL_RECORD_HPP

#include <cstddef>
#include <cstdint>
#include <cxxopts.hpp>
#include <filesystem>
#include <optional>
#include <string>
#include <string_view>
#include <vector>

#include "operon/algorithms/config.hpp"
#include "operon/algorithms/ga_base.hpp"
#include "operon/algorithms/generation_record.hpp"
#include "operon/algorithms/stop_reason.hpp"
#include "operon/core/tree.hpp"
#include "operon/operators/evaluator.hpp"

// The `operon.final` document written by `--final-json PATH` (schema: docs/schemas/final.schema.json): how a run
// ended, the budgets and counters it consumed (read from the evaluator after Run() returned, never copied from the
// last report), and the final model. The document is written atomically and always, also when the run failed.
//
// With --generation-log the stream is sealed before the final document, which carries its byte hashes and count.
// Without a stream, `segments` and `record_count` are explicitly declared unsupported.
namespace Operon::Cli {

// ---------------------------------------------------------------------------------------------------------------
// The document

struct FinalModel {
    std::string SelectionRule; // obj0 | mdl | bic | aic
    std::string ModelRef; // Operon::ModelRef of the final tree (lowercase hex SHA-256)
    std::string Infix; // the model as printed, with round-trip precision
};

struct FinalRecord {
    std::string RunId; // 64 lowercase hex characters, from --run-id
    std::string AttemptId; // lowercase UUID, from --attempt-id
    bool Error { false }; // exit_status "error" <=> StopReason::EngineError in Reasons
    Operon::StopMask Reasons {}; // every stop condition true at termination
    std::size_t GenerationsCompleted {}; // last generation whose pool was filled (initialization = 0)
    std::optional<Operon::OffspringProgress> Incomplete; // the generation body that was cut short, if any
    std::optional<std::size_t> GenerationLimit; // nullopt = unlimited
    std::optional<std::size_t> EvaluationLimit; // residual + Jacobian evaluations; nullopt = unlimited
    std::optional<double> TimeLimitSeconds; // nullopt = unlimited
    double ElapsedSeconds {};
    std::optional<std::size_t> PressureMax; // offspring selection only
    std::optional<double> PressureLast; // offspring selection only
    Operon::EvaluationStats Stats;
    std::optional<FinalModel> Model;
    std::optional<std::string> FrontRef; // SHA-256 of the written Pareto front (NSGA2 with --pareto-front)
    std::vector<Operon::GenerationSegment> Segments;
    std::size_t RecordCount {};
    std::size_t InstrumentationErrors {};
    std::vector<std::string> UnsupportedFields; // producer.unsupported_fields
};

// A limit at or above this is "unlimited" in the records (reported as null).
inline constexpr std::uint64_t UnlimitedThreshold = std::uint64_t { 1 } << 62U;

// Snapshot of an algorithm that is no longer running (or never got to run): limits, counters, progress, stop
// reasons. Model, front reference and identifiers are filled in by the caller. The reasons are the recorded ones
// plus every condition that holds in the final state. A generation whose pool was not filled is discarded by the
// algorithms, so `GenerationsCompleted` is Generation() and the cut-short generation is `Incomplete`.
[[nodiscard]] auto MakeFinalRecord(Operon::GeneticAlgorithmBase const& algorithm) -> FinalRecord;

// The record of a run that failed before an algorithm existed: nothing was evaluated, only the limits are known.
[[nodiscard]] auto MakeFinalRecord(Operon::GeneticAlgorithmConfig const& config) -> FinalRecord;

// The JSON text (one object, trailing newline). Non-finite numbers are written as null, never as NaN/Infinity.
[[nodiscard]] auto SerializeFinalRecord(FinalRecord const& record) -> std::string;

// Writes `content` to `path` so that readers see either the old file or the complete new one. The content goes to
// a temporary sibling file that is flushed to disk and then replaces `path` in one step: POSIX rename() followed by
// an fsync of the directory, Windows MoveFileExW(MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH). A temporary
// file is removed on failure. Throws std::runtime_error on failure.
auto WriteFileAtomically(std::filesystem::path const& path, std::string_view content) -> void;

// ---------------------------------------------------------------------------------------------------------------
// CLI glue

// --final-json / --run-id / --attempt-id of a program, plus its signal handling. One instance per run:
//
//   FinalRun finalRun { result };                       // throws std::invalid_argument on bad options
//   ... build everything ...
//   auto const watch = finalRun.Watch(algorithm);       // snapshots the algorithm when it goes out of scope,
//   ... algorithm.Run(...) with StopSignals::Forward(algorithm) in the report callback ...
//   finalRun.SetModel(...);  finalRun.SetFrontRef(...);
//   return finalRun.Finish(exitCode);                   // writes the record, returns the process exit status
//
// Without --final-json the object does nothing at all: no signal handlers, no file.
class FinalRun {
public:
    explicit FinalRun(cxxopts::ParseResult const& result,
        Operon::GenerationRecorder::Clock::time_point processStart = Operon::GenerationRecorder::Clock::now());
    ~FinalRun();
    FinalRun(FinalRun const&) = delete;
    auto operator=(FinalRun const&) -> FinalRun& = delete;
    FinalRun(FinalRun&&) = delete;
    auto operator=(FinalRun&&) -> FinalRun& = delete;

    [[nodiscard]] auto Enabled() const -> bool { return enabled_; }

    // Limits to report if the run fails before an algorithm is watched.
    auto SetConfig(Operon::GeneticAlgorithmConfig const& config) -> void { config_ = config; }
    auto StartGenerationLog(cxxopts::ParseResult const& result, bool nsga2, std::size_t objectiveCount)
        -> Operon::GenerationRecorder*;

    // Takes the snapshot of `algorithm` when the returned guard is destroyed (normal return or exception), and
    // forwards SIGINT/SIGTERM to it while it is watched. `algorithm` must outlive the guard.
    class Watch {
    public:
        Watch(FinalRun& run, Operon::GeneticAlgorithmBase& algorithm);
        ~Watch();
        Watch(Watch const&) = delete;
        auto operator=(Watch const&) -> Watch& = delete;
        Watch(Watch&&) = delete;
        auto operator=(Watch&&) -> Watch& = delete;

    private:
        FinalRun& run_;
        Operon::GeneticAlgorithmBase& algorithm_;
    };

    auto SetModel(std::string selectionRule, Operon::Tree const& tree, std::string infix) -> void;
    auto SetFrontRef(std::string sha256) -> void { frontRef_ = std::move(sha256); }
    // The algorithm produces a Pareto front object (NSGA2): a missing front reference is then declared as
    // unsupported instead of meaning "not applicable".
    auto ExpectFront() -> void { frontExpected_ = true; }
    // There is no run to record (the program only prints something and exits): write nothing, drop the handlers.
    auto Abandon() -> void;
    // The process is going to fail even if the algorithm itself ended normally.
    auto MarkError() -> void { error_ = true; }
    auto MarkInstrumentationError() -> void
    {
        ++instrumentationErrors_;
        error_ = true;
    }

    // Writes the record (if enabled) and returns the exit status for the process: `exitCode`, or, for a run that was
    // stopped by SIGINT/SIGTERM, 128 + the signal number. A non-zero `exitCode` makes the record an error record.
    [[nodiscard]] auto Finish(int exitCode) -> int;

private:
    bool enabled_ { false };
    std::filesystem::path path_;
    std::string runId_;
    std::string attemptId_;
    Operon::GeneticAlgorithmConfig config_ {};
    std::optional<FinalRecord> snapshot_;
    std::optional<FinalModel> model_;
    std::optional<std::string> frontRef_;
    bool frontExpected_ { false };
    bool error_ { false };
    Operon::GenerationRecorder::Clock::time_point processStart_;
    std::unique_ptr<Operon::GenerationRecorder> recorder_;
    std::size_t instrumentationErrors_ {};
    bool streamRequested_ { false };
};

} // namespace Operon::Cli

#endif
