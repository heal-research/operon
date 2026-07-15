// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_CHAIN_HPP
#define OPERON_ALGORITHMS_PROBES_CHAIN_HPP

#include <cstddef>
#include <memory>
#include <string_view>
#include <type_traits>
#include <vector>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Consumes one generation's ResultRecord. Separate from GenerationProbe so
// probes stay agnostic of the output format and can share one sink/file.
struct RecordSink {
    virtual ~RecordSink() = default;
    RecordSink() = default;
    RecordSink(RecordSink const&) = default;
    RecordSink(RecordSink&&) noexcept = default;
    auto operator=(RecordSink const&) -> RecordSink& = default;
    auto operator=(RecordSink&&) noexcept -> RecordSink& = default;

    virtual auto Write(ResultRecord const& record) -> void = 0;
    virtual auto Flush() -> void { }
};

// Writes one JSON object per line (NDJSON) - chosen over a fixed-column
// format like CSV because ResultRecord's keys can differ per generation.
class OPERON_EXPORT JsonlSink final : public RecordSink {
public:
    explicit JsonlSink(std::string_view path);
    ~JsonlSink() override;
    JsonlSink(JsonlSink const&) = delete;
    JsonlSink(JsonlSink&&) noexcept;
    auto operator=(JsonlSink const&) -> JsonlSink& = delete;
    auto operator=(JsonlSink&&) noexcept -> JsonlSink&;

    auto Write(ResultRecord const& record) -> void override;
    auto Flush() -> void override;

    // False if the file failed to open - Write() then silently no-ops.
    [[nodiscard]] auto IsOpen() const -> bool;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Runs a set of GenerationProbes, each on its own (every, offset) schedule,
// and hands the combined per-generation record to one shared sink. Not
// integrated with taskflow's ObserverInterface: that observes per-task
// entry/exit with no domain data, whereas the existing ReportCallback
// closure already gives direct, generation-granular access to algorithm
// state. ProbeChain is meant to be invoked from inside that report lambda.
class OPERON_EXPORT ProbeChain {
public:
    ProbeChain() = default;
    ~ProbeChain() { Finish(); }
    ProbeChain(ProbeChain const&) = delete;
    // Move-construction only, not move-assignment: this needs to stay
    // movable so it can be returned by value from a future config-loading
    // factory, but a defaulted move-assign would member-wise overwrite
    // *this's entries_ without ever calling their Finish() - move-construct
    // into a fresh chain instead.
    ProbeChain(ProbeChain&&) noexcept = default;
    auto operator=(ProbeChain const&) -> ProbeChain& = delete;
    auto operator=(ProbeChain&&) -> ProbeChain& = delete;

    // every == 0 disables the probe without removing it, so config-driven
    // callers can toggle a probe off without editing the entry list. Note
    // Finish() below still runs a disabled probe's Finish() - only the
    // per-generation call is skipped.
    auto Add(std::unique_ptr<GenerationProbe> probe, std::size_t every = 1, std::size_t offset = 0) -> void;
    auto SetSink(std::unique_ptr<RecordSink> sink) -> void;

    [[nodiscard]] auto Empty() const noexcept -> bool { return entries_.empty(); }

    // Call once per generation (e.g. from an algorithm's ReportCallback).
    // Only writes to the sink if at least one probe actually ran. Keys
    // "generation" and "elapsed" are reserved (set here before any probe
    // runs); a probe that emits either overwrites it via insert_or_assign.
    auto operator()(GeneticAlgorithmBase const& algo) -> void;

    // Call once after the algorithm's Run() returns. Idempotent - the
    // destructor also calls this as a safety net.
    auto Finish() -> void;

private:
    struct Entry {
        std::unique_ptr<GenerationProbe> Probe;
        std::size_t Every{1};
        std::size_t Offset{0};
    };

    std::vector<Entry> entries_;
    std::unique_ptr<RecordSink> sink_;
    bool finished_{false};
};

static_assert(std::is_move_constructible_v<ProbeChain> && !std::is_move_assignable_v<ProbeChain>,
              "ProbeChain must stay move-constructible but not move-assignable - see the class comment");

} // namespace Operon

#endif
