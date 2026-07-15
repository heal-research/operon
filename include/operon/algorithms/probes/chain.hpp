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
// probes stay agnostic of the output format and can share one sink/file.
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

    auto Write(std::size_t generation, ResultRecord const& record) -> void override;
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
    ProbeChain(ProbeChain&&) noexcept = default;
    auto operator=(ProbeChain const&) -> ProbeChain& = delete;
    // Spelled out explicitly because the user-declared destructor otherwise
    // suppresses the implicit move ops, and ProbeChain needs to stay
    // movable (e.g. returned by value from a future config-loading factory).
    auto operator=(ProbeChain&&) noexcept -> ProbeChain& = default;

    // every == 0 disables the probe without removing it, so config-driven
    // callers can toggle a probe off without editing the entry list.
    auto Add(std::unique_ptr<GenerationProbe> probe, std::size_t every = 1, std::size_t offset = 0) -> void;
    auto SetSink(std::unique_ptr<RecordSink> sink) -> void;

    [[nodiscard]] auto Empty() const noexcept -> bool { return entries_.empty(); }

    // Call once per generation (e.g. from an algorithm's ReportCallback).
    // Only writes to the sink if at least one probe actually ran.
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

} // namespace Operon

#endif
