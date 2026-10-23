// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#ifndef OPERON_GENERATION_RECORD_HPP
#define OPERON_GENERATION_RECORD_HPP

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <exception>
#include <filesystem>
#include <limits>
#include <mutex>
#include <optional>
#include <string>
#include <thread>
#include <vector>

#include "operon/algorithms/ga_base.hpp"
#include "operon/hash/sha256.hpp"
#include "operon/operon_export.hpp"

// One `operon.generation` JSON record per completed generation (schema: docs/schemas/generation.schema.json), written
// as lines of generation-NNNNNN.jsonl segments. GenerationRecord holds the native values the line is built from.
namespace Operon {

struct GenerationSegment {
    std::size_t Id {};
    std::size_t FirstSeq {};
    std::size_t LastSeq {};
    std::size_t Count {};
    std::string Sha256;
};

struct GenerationObjective {
    std::string Name;
    double Sum {};
    Scalar Min { std::numeric_limits<Scalar>::infinity() };
    Scalar Max { -std::numeric_limits<Scalar>::infinity() };
    std::size_t Finite {}, Rejected {}, Nan {}, PosInf {}, NegInf {};
};

struct OPERON_EXPORT GenerationStructure {
    double Sum {};
    std::size_t Min { std::numeric_limits<std::size_t>::max() }, Max {};
    auto Add(std::size_t value) -> void;
};

struct GenerationRecord {
    std::size_t Generation {}, PopulationCount {}, PoolCount {};
    OffspringProgress Progress;
    EvaluationStats Stats;
    std::vector<GenerationObjective> Objectives;
    GenerationStructure Length, Depth;
    std::vector<std::size_t> Ranks;
    std::optional<std::size_t> EvaluationLimit;
    std::optional<std::size_t> AttemptedCandidates, AcceptedCandidates, PressureMax;
    std::size_t PressureDenominator {};
    double ComparisonFactor {};
    double ProcessElapsed {}, LogicalElapsed {}, GenerationWall {}, SortSeconds {};
};

struct GenerationRecorderOptions {
    std::filesystem::path Directory;
    std::string RunId, AttemptId;
    std::vector<std::string> ObjectiveNames;
    std::string Generator { "basic" };
    std::string SelectionRule { "obj0" };
    bool Nsga2 { false };
    std::size_t SegmentGenerations { 10 };
    double SegmentSeconds { 5 };
    bool SyncRecord { false };
    std::chrono::steady_clock::time_point ProcessStart { std::chrono::steady_clock::now() };
};

// One callback-thread writer per attempt. Capture precedes reporting; Write consumes the reporter's exact model
// and metrics. A single seal worker overlaps durability I/O with computation, but Write waits before opening the
// next .part (at most one exists). Close is explicit and throwing; destruction joins already-requested seals but
// never seals the trailing unrequested segment.
class OPERON_EXPORT GenerationRecorder {
public:
    using Clock = std::chrono::steady_clock;
    explicit GenerationRecorder(GenerationRecorderOptions options);
    ~GenerationRecorder();
    GenerationRecorder(GenerationRecorder const&) = delete;
    auto operator=(GenerationRecorder const&) -> GenerationRecorder& = delete;
    auto Capture(GeneticAlgorithmBase const& algorithm) -> void;
    auto Write(GeneticAlgorithmBase const& algorithm, Individual const& selected, double nmseTrain, double nmseTest,
        double finitePredictionFraction) -> void;
    auto Close() -> void;
    // Wait for a count/time-triggered seal, propagating its error on the caller thread.
    auto Drain() -> void;
    [[nodiscard]] auto Record() const -> GenerationRecord const& { return record_; }
    [[nodiscard]] auto Segments() -> std::vector<GenerationSegment> const&
    {
        Drain();
        return segments_;
    }
    [[nodiscard]] auto RecordCount() const -> std::size_t { return seq_; }
    [[nodiscard]] auto Errors() const -> std::size_t { return errors_; }

private:
    auto Seal() -> void;
    auto SealWorker(std::stop_token stop) -> void;
    auto Serialize(Individual const& selected, double train, double test, double finiteFraction, bool emitted) -> void;
    GenerationRecorderOptions options_;
    GenerationRecord record_;
    std::vector<GenerationSegment> segments_;
    std::string buffer_, modelRef_;
    Tree previousModel_;
    std::optional<bool> shapeFeasible_;
    std::optional<Scalar> shapeViolation_;
    Sha256 hash_;
    Clock::time_point reportStart_, previousBoundary_, segmentStart_;
    std::filesystem::path partPath_, sealedPath_;
    std::size_t seq_ {}, firstSeq_ {};
    std::atomic_size_t errors_ {};
    int directoryFd_ { -1 }, segmentFd_ { -1 }, modelsFd_ { -1 };
    bool captured_ { false }, closed_ { false };
    std::mutex sealMutex_;
    std::condition_variable sealCondition_;
    bool sealPending_ { false };
    std::exception_ptr sealError_;
    std::jthread sealThread_;
};

} // namespace Operon
#endif
