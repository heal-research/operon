// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors
#include "operon/algorithms/generation_record.hpp"

#include <algorithm>
#include <array>
#include <bit>
#include <cerrno>
#include <charconv>
#include <climits>
#include <cmath>
#include <cstring>
#include <fcntl.h>
#include <fmt/format.h>
#include <limits>
#include <stdexcept>
#include <system_error>
#if defined(_WIN32)
#include <io.h>
#include <sys/stat.h>
#else
#include <unistd.h>
#endif

#include "operon/algorithms/phase_timer.hpp"
#include "operon/hash/model_ref.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"

namespace Operon {
namespace {
    auto IoError(char const* operation) -> void { throw std::system_error(errno, std::generic_category(), operation); }

    auto OpenDirectory(std::filesystem::path const& path) -> int
    {
#if defined(_WIN32)
        static_cast<void>(path);
        return -1; // Windows has no portable directory flush through the CRT.
#else
        return ::open(path.c_str(), O_RDONLY | O_DIRECTORY | O_CLOEXEC); // NOLINT
#endif
    }

    auto OpenExclusive(std::filesystem::path const& path) -> int
    {
#if defined(_WIN32)
        return ::_open(path.string().c_str(), _O_WRONLY | _O_CREAT | _O_EXCL | _O_BINARY, _S_IREAD | _S_IWRITE);
#else
        return ::open(path.c_str(), O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0644); // NOLINT
#endif
    }

    auto Sync(int fd) -> void
    {
        if (fd < 0) {
            return;
        }
#if defined(_WIN32)
        if (::_commit(fd) != 0) {
            IoError("generation log fsync");
        }
#else
        while (::fsync(fd) != 0) {
            if (errno != EINTR) {
                IoError("generation log fsync");
            }
        }
#endif
    }

    auto WriteAll(int fd, std::string_view bytes) -> void
    {
        while (!bytes.empty()) {
#if defined(_WIN32)
            auto const count = static_cast<unsigned>(std::min(bytes.size(), std::size_t { INT_MAX }));
            auto const n = ::_write(fd, bytes.data(), count);
#else
            auto const n = ::write(fd, bytes.data(), bytes.size());
#endif
            if (n < 0 && errno == EINTR) {
                continue;
            }
            if (n <= 0) {
                IoError("generation log write");
            }
            bytes.remove_prefix(static_cast<std::size_t>(n));
        }
    }

    auto CloseFd(int fd) -> int
    {
#if defined(_WIN32)
        return ::_close(fd);
#else
        return ::close(fd);
#endif
    }

    auto CloseFile(int& fd) -> void
    {
        auto const old = std::exchange(fd, -1);
        if (old >= 0 && CloseFd(old) != 0) {
            IoError("generation log close");
        }
    }

    auto SameModel(Tree const& lhs, Tree const& rhs) -> bool
    {
        return std::ranges::equal(lhs.Nodes(), rhs.Nodes(), [](Node const& a, Node const& b) {
            return a.Type == b.Type && a.HashValue == b.HashValue && a.Arity == b.Arity && a.IsEnabled == b.IsEnabled
                && a.Optimize == b.Optimize && (!a.IsRef() || a.RefTo == b.RefTo)
                && std::bit_cast<std::array<std::byte, sizeof(Scalar)>>(a.Value)
                == std::bit_cast<std::array<std::byte, sizeof(Scalar)>>(b.Value);
        });
    }

    // The numeric overload retains its argument's precision. No float-to-double promotion before to_chars.
    class Json {
    public:
        explicit Json(std::string& out) : out_(out) {}
        auto Raw(std::string_view value) -> void { out_.append(value); }
        auto Value(std::nullptr_t) -> void { Raw("null"); }
        auto Value(bool value) -> void { Raw(value ? "true" : "false"); }
        auto Value(std::string_view value) -> void
        {
            Raw("\"");
            constexpr char hex[] = "0123456789abcdef";
            for (unsigned char c : value) {
                if (c == '"' || c == '\\') {
                    out_.push_back('\\');
                    out_.push_back(static_cast<char>(c));
                } else if (c < 32) {
                    Raw("\\u00");
                    out_.push_back(hex[c >> 4U]);
                    out_.push_back(hex[c & 15U]);
                } else {
                    out_.push_back(static_cast<char>(c));
                }
            }
            Raw("\"");
        }
        auto Value(std::string const& value) -> void { Value(std::string_view(value)); }
        auto Value(char const* value) -> void { Value(std::string_view(value)); }
        template <typename T>
            requires std::is_arithmetic_v<T>
        auto Value(T value) -> void
        {
            if constexpr (std::is_floating_point_v<T>) {
                if (!std::isfinite(value)) {
                    Value(nullptr);
                    return;
                }
            }
            std::array<char, 64> chars {};
            auto const [end, ec] = std::to_chars(chars.data(), chars.data() + chars.size(), value);
            if (ec != std::errc {}) {
                throw std::runtime_error("generation number serialization failed");
            }
            out_.append(chars.data(), end);
        }
        template <typename T> auto Value(std::optional<T> const& value) -> void
        {
            if (value) {
                Value(*value);
            } else {
                Value(nullptr);
            }
        }
        auto Key(std::string_view key) -> void
        {
            if (out_.back() != '{') {
                Raw(",");
            }
            Value(key);
            Raw(":");
        }
        template <typename T> auto Field(std::string_view key, T const& value) -> void
        {
            Key(key);
            Value(value);
        }
        auto Begin(std::string_view key) -> void
        {
            Key(key);
            Raw("{");
        }
        auto End() -> void { Raw("}"); }

    private:
        std::string& out_;
    };

    auto Counters(Json& j, EvaluationStats const& s) -> void
    {
        j.Raw("{");
        j.Field("evaluator_calls", s.Calls);
        j.Field("residual_scoring", s.ResidualScoring);
        j.Field("residual_optimizer", s.ResidualOptimizer);
        j.Field("jacobian_scoring", s.JacobianScoring);
        j.Field("jacobian_optimizer", s.JacobianOptimizer);
        j.Field("residual_total", s.ResidualTotal());
        j.Field("jacobian_total", s.JacobianTotal());
        j.Field("cache_hits", s.CacheHits);
        j.Field("cache_misses",
            s.CacheHits && s.CacheLookups ? std::optional { *s.CacheLookups - *s.CacheHits } : std::nullopt);
        j.Field("optimizer_seconds", static_cast<double>(s.OptimizerTimeUs) / 1e6);
        j.End();
    }

    auto Structure(Json& j, GenerationStructure const& s, std::size_t count) -> void
    {
        j.Raw("{");
        j.Field("mean", count ? std::optional { s.Sum / static_cast<double>(count) } : std::nullopt);
        j.Field("min", count ? std::optional { s.Min } : std::nullopt);
        j.Field("max", count ? std::optional { s.Max } : std::nullopt);
        j.End();
    }
} // namespace

auto GenerationStructure::Add(std::size_t value) -> void
{
    Sum += static_cast<double>(value);
    Min = std::min(Min, value);
    Max = std::max(Max, value);
}

GenerationRecorder::GenerationRecorder(GenerationRecorderOptions options) : options_(std::move(options))
{
    if (options_.SegmentGenerations == 0 || !std::isfinite(options_.SegmentSeconds) || options_.SegmentSeconds <= 0) {
        throw std::invalid_argument("generation segment count and seconds must be positive and finite");
    }
    buffer_.reserve(8192);
    for (auto const& name : options_.ObjectiveNames) {
        record_.Objectives.push_back(GenerationObjective { .Name = name });
    }
    previousBoundary_ = options_.ProcessStart;
    try {
        std::filesystem::create_directories(options_.Directory);
        directoryFd_ = OpenDirectory(options_.Directory);
        if (directoryFd_ < 0) {
#if !defined(_WIN32)
            IoError("generation log directory open");
#endif
        }
        // Never replace an earlier attempt's records, even if the caller accidentally reuses its directory.
        if (std::filesystem::exists(options_.Directory / "generation-000000.jsonl")
            || std::filesystem::exists(options_.Directory / "generation-000000.jsonl.part")) {
            throw std::runtime_error("generation log directory already contains a stream");
        }
        modelsFd_ = OpenExclusive(options_.Directory / "models.jsonl.txt");
        if (modelsFd_ < 0) {
            IoError("generation model sidecar open");
        }
        sealThread_ = std::jthread([this](std::stop_token stop) { SealWorker(stop); });
    } catch (...) {
        if (modelsFd_ >= 0) {
            static_cast<void>(CloseFd(modelsFd_));
        }
        if (directoryFd_ >= 0) {
            static_cast<void>(CloseFd(directoryFd_));
        }
        throw;
    }
}

GenerationRecorder::~GenerationRecorder()
{
    {
        std::lock_guard lock(sealMutex_);
        sealThread_.request_stop();
    }
    sealCondition_.notify_all();
    if (sealThread_.joinable()) {
        sealThread_.join();
    }
    if (segmentFd_ >= 0) {
        static_cast<void>(CloseFd(segmentFd_));
    }
    if (modelsFd_ >= 0) {
        static_cast<void>(CloseFd(modelsFd_));
    }
    if (directoryFd_ >= 0) {
        static_cast<void>(CloseFd(directoryFd_));
    }
}

auto GenerationRecorder::Capture(GeneticAlgorithmBase const& algorithm) -> void
{
    try {
        if (closed_ || captured_) {
            throw std::logic_error("generation recorder callback out of order");
        }
        reportStart_ = Clock::now();
        auto const pop = algorithm.Parents();
        auto& r = record_;
        r.Generation = algorithm.Generation();
        if (r.Generation != seq_) {
            throw std::logic_error("generation log requires a fresh, dense generation stream");
        }
        r.PopulationCount = pop.size();
        r.PoolCount = algorithm.GetConfig().PoolSize;
        r.Progress = algorithm.Progress();
        r.Stats = algorithm.DetailedStats();
        auto const budget = algorithm.GetGenerator()->Evaluator()->Budget();
        r.EvaluationLimit = budget >= (std::uint64_t { 1 } << 62U) ? std::nullopt : std::optional { budget };
        r.Length = {};
        r.Depth = {};
        for (auto& o : r.Objectives) {
            o.Sum = 0;
            o.Min = std::numeric_limits<Scalar>::infinity();
            o.Max = -o.Min;
            o.Finite = o.Rejected = o.Nan = o.PosInf = o.NegInf = 0;
        }
        r.Ranks.clear();
        // Fixed population order, double sum: reproducible independently of Taskflow scheduling.
        for (auto const& ind : pop) {
            if (ind.Size() != r.Objectives.size()) {
                throw std::logic_error("generation objective names/count mismatch");
            }
            r.Length.Add(ind.Genotype.Length());
            r.Depth.Add(ind.Genotype.Depth());
            if (options_.Nsga2) {
                if (ind.Rank >= pop.size()) {
                    throw std::logic_error("invalid population rank");
                }
                if (r.Ranks.size() <= ind.Rank) {
                    r.Ranks.resize(ind.Rank + 1);
                }
                ++r.Ranks[ind.Rank];
            }
            for (std::size_t k = 0; k < ind.Size(); ++k) {
                auto& o = r.Objectives[k];
                auto const v = ind[k];
                if (ind.ShapeRejected || v == EvaluatorBase::ErrMax) {
                    ++o.Rejected;
                } else if (std::isnan(v)) {
                    ++o.Nan;
                } else if (std::isinf(v)) {
                    v > 0 ? ++o.PosInf : ++o.NegInf;
                } else {
                    ++o.Finite;
                    o.Sum += static_cast<double>(v);
                    o.Min = std::min(o.Min, v);
                    o.Max = std::max(o.Max, v);
                }
            }
        }
        if (auto const* os = dynamic_cast<OffspringSelectionGenerator const*>(algorithm.GetGenerator())) {
            r.AttemptedCandidates = r.Generation ? std::optional { os->AttemptedCandidates() } : std::nullopt;
            r.AcceptedCandidates = r.Generation ? std::optional { r.Progress.Completed } : std::nullopt;
            r.PressureDenominator = os->FemaleSelector()->Population().size();
            r.PressureMax = os->MaxSelectionPressure();
            r.ComparisonFactor = os->ComparisonFactor();
        }
        r.ProcessElapsed = std::chrono::duration<double>(reportStart_ - options_.ProcessStart).count();
        r.GenerationWall = std::chrono::duration<double>(reportStart_ - previousBoundary_).count();
        previousBoundary_ = reportStart_;
        r.LogicalElapsed = algorithm.Elapsed();
        auto const it = algorithm.Timings().find(std::string { SortTaskName });
        r.SortSeconds = it == algorithm.Timings().end() ? 0 : it->second;
        captured_ = true;
    } catch (...) {
        ++errors_;
        throw;
    }
}

auto GenerationRecorder::Write(GeneticAlgorithmBase const& algorithm, Individual const& selected, double nmseTrain,
    double nmseTest, double finitePredictionFraction) -> void
{
    Drain();
    try {
        if (!captured_ || closed_) {
            throw std::logic_error("generation recorder write without capture");
        }
        if (segmentFd_ < 0) {
            sealedPath_ = options_.Directory / fmt::format("generation-{:06}.jsonl", segments_.size());
            partPath_ = sealedPath_;
            partPath_ += ".part";
            segmentFd_ = OpenExclusive(partPath_);
            if (segmentFd_ < 0) {
                IoError("generation segment open");
            }
            segmentStart_ = Clock::now();
            firstSeq_ = seq_;
            hash_.Reset();
        }
        auto const changed = modelRef_.empty() || !SameModel(previousModel_, selected.Genotype);
        if (changed) {
            auto const serialization = ModelSerialization(selected.Genotype);
            modelRef_ = Sha256Hex(serialization);
            buffer_.clear();
            Json j(buffer_);
            j.Raw("{");
            j.Field("model_ref", modelRef_);
            j.Field("serialization", serialization);
            j.End();
            j.Raw("\n");
            WriteAll(modelsFd_, buffer_);
            previousModel_ = selected.Genotype;
            if (auto const* shape
                = dynamic_cast<detail::ShapeConstraintEnforcer const*>(algorithm.GetGenerator()->Evaluator())) {
                auto const summary = shape->Certifier().Measure(selected.Genotype, std::nullopt);
                shapeFeasible_ = summary.Feasible;
                shapeViolation_ = summary.Violation;
            }
        }
        Serialize(selected, nmseTrain, nmseTest, finitePredictionFraction, changed);
        WriteAll(segmentFd_, buffer_);
        hash_.Update(buffer_);
        ++seq_;
        captured_ = false;
        if (options_.SyncRecord) {
            Sync(modelsFd_);
            Sync(segmentFd_);
        }
        if (seq_ - firstSeq_ >= options_.SegmentGenerations
            || std::chrono::duration<double>(Clock::now() - segmentStart_).count() >= options_.SegmentSeconds) {
            {
                std::lock_guard lock(sealMutex_);
                sealPending_ = true;
            }
            sealCondition_.notify_all();
        }
    } catch (...) {
        ++errors_;
        throw;
    }
}

auto GenerationRecorder::Seal() -> void
{
    if (segmentFd_ < 0) {
        return;
    }
    Sync(modelsFd_);
    Sync(segmentFd_);
    CloseFile(segmentFd_);
    if (std::filesystem::exists(sealedPath_)) {
        throw std::runtime_error("generation segment already exists");
    }
    std::filesystem::rename(partPath_, sealedPath_);
    Sync(directoryFd_);
    segments_.push_back({ segments_.size(), firstSeq_, seq_ - 1, seq_ - firstSeq_, Sha256::ToHex(hash_.Finalize()) });
}

auto GenerationRecorder::SealWorker(std::stop_token stop) -> void
{
    std::unique_lock lock(sealMutex_);
    for (;;) {
        sealCondition_.wait(lock, [&] { return sealPending_ || stop.stop_requested(); });
        if (!sealPending_) {
            return;
        }
        lock.unlock();
        std::exception_ptr error;
        try {
            Seal();
        } catch (...) {
            ++errors_;
            error = std::current_exception();
        }
        lock.lock();
        sealError_ = error;
        sealPending_ = false;
        sealCondition_.notify_all();
    }
}

auto GenerationRecorder::Drain() -> void
{
    std::unique_lock lock(sealMutex_);
    sealCondition_.wait(lock, [&] { return !sealPending_; });
    if (auto error = std::exchange(sealError_, {})) {
        std::rethrow_exception(error);
    }
}

auto GenerationRecorder::Close() -> void
{
    if (closed_) {
        return;
    }
    Drain();
    try {
        Seal();
        Sync(modelsFd_);
        CloseFile(modelsFd_);
        Sync(directoryFd_);
        CloseFile(directoryFd_);
        closed_ = true;
        if (captured_ && errors_ == 0) {
            throw std::runtime_error("generation report did not complete");
        }
    } catch (...) {
        ++errors_;
        throw;
    }
}

auto GenerationRecorder::Serialize(
    Individual const& selected, double train, double test, double finiteFraction, bool emitted) -> void
{
    auto const& r = record_;
    buffer_.clear();
    Json j(buffer_);
    j.Raw("{");
    j.Field("schema", "operon.generation");
    j.Field("schema_version", "1.1");
    j.Field("run_id", options_.RunId);
    j.Field("attempt_id", options_.AttemptId);
    j.Field("segment_id", segments_.size());
    j.Field("seq", seq_);
    j.Field("generation", r.Generation);
    j.Field("event", r.Generation ? "generation_complete" : "initialized");
    j.Field("checkpoint_parent", nullptr);
    j.Begin("budget");
    j.Field("unit", "residual+jacobian");
    j.Field("limit", r.EvaluationLimit);
    j.Field("consumed", r.Stats.Consumed());
    j.End();
    j.Begin("counters");
    j.Key("logical");
    Counters(j, r.Stats);
    j.Key("since_restart");
    Counters(j, r.Stats);
    j.End();
    j.Key("objectives");
    j.Raw("[");
    for (std::size_t k = 0; k < r.Objectives.size(); ++k) {
        auto const& o = r.Objectives[k];
        if (k) {
            j.Raw(",");
        }
        j.Raw("{");
        j.Field("name", o.Name);
        j.Field("direction", "min");
        j.Field("best", o.Min);
        // Accumulate in double, then round once to the build precision like the extrema. Mixing a double mean
        // with shortest-float extrema can put the parsed JSON mean outside [min,max] for a uniform population.
        j.Field("mean",
            o.Finite ? std::optional { static_cast<Scalar>(o.Sum / static_cast<double>(o.Finite)) } : std::nullopt);
        j.Field("min", o.Min);
        j.Field("max", o.Max);
        j.Field("finite_count", o.Finite);
        j.Field("rejected_count", o.Rejected);
        j.Field("nan_count", o.Nan);
        j.Field("posinf_count", o.PosInf);
        j.Field("neginf_count", o.NegInf);
        j.End();
    }
    j.Raw("]");
    j.Begin("structure");
    j.Field("population_count", r.PopulationCount);
    j.Field("pool_count", r.PoolCount);
    j.Field("attempted_offspring", r.Generation ? std::optional { r.Progress.Attempted } : std::nullopt);
    j.Field("completed_offspring", r.Generation ? std::optional { r.Progress.Completed } : std::nullopt);
    j.Key("length");
    Structure(j, r.Length, r.PopulationCount);
    j.Key("depth");
    Structure(j, r.Depth, r.PopulationCount);
    j.End();
    j.Begin("selection");
    j.Field("generator", options_.Generator);
    j.Field("applicable", r.PressureMax.has_value());
    j.Field("attempted_candidates", r.AttemptedCandidates);
    j.Field("accepted_candidates", r.AcceptedCandidates);
    j.Key("pressure");
    if (r.AttemptedCandidates) {
        j.Raw("{");
        j.Field("numerator", *r.AttemptedCandidates);
        j.Field("denominator", r.PressureDenominator);
        j.Field("value",
            r.PressureDenominator ? std::optional { static_cast<double>(*r.AttemptedCandidates)
                                        / static_cast<double>(r.PressureDenominator) }
                                  : std::nullopt);
        j.Field("max", r.PressureMax);
        j.End();
    } else {
        j.Value(nullptr);
    }
    j.Field("comparison_factor", r.PressureMax ? std::optional { r.ComparisonFactor } : std::nullopt);
    j.Field("success_ratio",
        r.AttemptedCandidates && *r.AttemptedCandidates
            ? std::optional { static_cast<double>(*r.AcceptedCandidates) / static_cast<double>(*r.AttemptedCandidates) }
            : std::nullopt);
    j.End();
    j.Key("nsga2");
    if (options_.Nsga2) {
        j.Raw("{");
        j.Key("rank_histogram");
        j.Raw("[");
        for (std::size_t k = 0; k < r.Ranks.size(); ++k) {
            if (k) {
                j.Raw(",");
            }
            j.Value(r.Ranks[k]);
        }
        j.Raw("]");
        j.End();
    } else {
        j.Value(nullptr);
    }
    j.Begin("selected_model");
    j.Field("selection_rule", options_.SelectionRule);
    j.Field("model_ref", modelRef_);
    j.Field("model_object_emitted", emitted);
    j.Field("node_count", selected.Genotype.Length());
    j.Key("objectives");
    j.Raw("[");
    for (std::size_t k = 0; k < selected.Size(); ++k) {
        if (k) {
            j.Raw(",");
        }
        j.Value(selected[k]);
    }
    j.Raw("]");
    j.Field("nmse_train", train);
    j.Field("nmse_validation", nullptr);
    j.Field("nmse_test", test);
    j.Field("metric_status", std::isfinite(train) ? "ok" : std::isnan(train) ? "nan" : train > 0 ? "+inf" : "-inf");
    j.End();
    j.Begin("validity");
    j.Field("finite_prediction_fraction", finiteFraction);
    j.Field("shape_feasible", shapeFeasible_);
    j.Field("shape_violation", shapeViolation_);
    j.Key("unsupported_fields");
    j.Raw("[\"selected_model.nmse_validation\"");
    if (!shapeFeasible_) {
        j.Raw(",\"validity.shape_feasible\",\"validity.shape_violation\"");
    }
    j.Raw("]");
    j.End();
    j.Begin("time");
    j.Field("process_elapsed_s", r.ProcessElapsed);
    j.Field("logical_elapsed_s", r.LogicalElapsed);
    j.Field("generation_wall_s", r.GenerationWall);
    j.Field("optimizer_worker_s", static_cast<double>(r.Stats.OptimizerTimeUs) / 1e6);
    j.Field("sort_s", options_.Nsga2 ? std::optional { r.SortSeconds } : std::nullopt);
    // Includes selection, metrics, model snapshots and JSON construction up to this field. The final write and
    // seal necessarily occur after the bytes containing this measurement are fixed; benchmark those end-to-end.
    j.Field("report_s", std::chrono::duration<double>(Clock::now() - reportStart_).count());
    j.End();
    j.End();
    j.Raw("\n");
}
} // namespace Operon
