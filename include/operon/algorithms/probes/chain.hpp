// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_CHAIN_HPP
#define OPERON_ALGORITHMS_PROBES_CHAIN_HPP

#include <cstddef>
#include <memory>
#include <string_view>
#include <vector>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Consumes one generation's ResultRecord. Separate from GenerationProbe so
// probes stay agnostic of the output format (NDJSON today, something else
// later) and multiple probes can share one sink/one file.
struct RecordSink {
    virtual ~RecordSink() = default;
    RecordSink() = default;
    RecordSink(RecordSink const&) = default;
    RecordSink(RecordSink&&) noexcept = default;
    auto operator=(RecordSink const&) -> RecordSink& = default;
    auto operator=(RecordSink&&) noexcept -> RecordSink& = default;

    virtual auto Write(std::size_t generation, ResultRecord const& record) -> void = 0;
    virtual auto Flush() -> void { }
};

// Writes one JSON object per line (NDJSON). Chosen over a fixed-column
// format (e.g. CSV) because ResultRecord is an unordered, self-describing
// bag of keys that can differ from one generation to the next depending on
// which probes ran that generation - NDJSON records their keys inline
// instead of requiring a stable schema up front.
class OPERON_EXPORT JsonlSink final : public RecordSink {
public:
    explicit JsonlSink(std::string_view path);
    ~JsonlSink() override;
    JsonlSink(JsonlSink const&) = delete;
    JsonlSink(JsonlSink&&) noexcept;
    auto operator=(JsonlSink const&) -> JsonlSink& = delete;
    auto operator=(JsonlSink&&) noexcept -> JsonlSink&;

    auto Write(std::size_t generation, ResultRecord const& record) -> void override;
    auto Flush() -> void override;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

// Runs a set of GenerationProbes, each on its own (every, offset) schedule,
// and hands the combined per-generation record to one shared sink.
// Deliberately not integrated with taskflow's ObserverInterface: that
// observes per-task entry/exit across worker threads (used today only for
// PhaseTimer's wall-clock profiling) and carries no domain data - a probe
// would still need a side channel back to the algorithm's state, which the
// existing ReportCallback closure already provides directly at exactly
// generation granularity. ProbeChain is meant to be invoked from inside
// that same report lambda (see cli/source/probes_config.hpp), one call per
// generation, on whichever thread runs the report task.
class OPERON_EXPORT ProbeChain {
public:
    // every == 0 disables the probe (kept registered but never invoked) -
    // this is deliberately not rejected/asserted against so config-driven
    // callers can toggle a probe off without removing its entry.
    auto Add(std::unique_ptr<GenerationProbe> probe, std::size_t every = 1, std::size_t offset = 0) -> void;
    auto SetSink(std::unique_ptr<RecordSink> sink) -> void;

    [[nodiscard]] auto Empty() const noexcept -> bool { return entries_.empty(); }

    // Intended to be called once per generation (e.g. from an algorithm's
    // ReportCallback). Builds this generation's record, runs every probe
    // whose schedule matches, and writes the record via the sink - but only
    // if at least one probe actually ran, so generations with nothing
    // scheduled don't produce empty lines.
    auto operator()(GeneticAlgorithmBase const& algo) -> void;

    // Call once after the algorithm's Run() returns.
    auto Finish() -> void;

private:
    struct Entry {
        std::unique_ptr<GenerationProbe> Probe;
        std::size_t Every{1};
        std::size_t Offset{0};
    };

    std::vector<Entry> entries_;
    std::unique_ptr<RecordSink> sink_;
};

} // namespace Operon

#endif
