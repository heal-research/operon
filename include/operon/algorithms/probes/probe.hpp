// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_PROBE_HPP
#define OPERON_ALGORITHMS_PROBES_PROBE_HPP

#include <cstdint>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include "operon/algorithms/ga_base.hpp"
#include "operon/core/types.hpp"

namespace Operon {

// A single named value emitted by a probe for one generation. Vectors cover
// per-individual/per-variable readings (e.g. a diversity probe's raw
// pairwise distances) that don't collapse to one scalar; scalars cover
// everything else (counts, rates, sizes).
using ResultValue = std::variant<
    std::int64_t,
    double,
    bool,
    std::string,
    std::vector<std::int64_t>,
    std::vector<double>>;

// One generation's worth of named values, handed to a RecordSink. Deliberately
// not a fixed-column struct: Operon::Map is unordered (ankerl::unordered_dense),
// so probes are free to emit whatever keys are relevant that generation and the
// sink (NDJSON) serializes each record as a self-describing object.
using ResultRecord = Operon::Map<std::string, ResultValue>;

// Read-only view into the running algorithm plus a handle to the current
// generation's record. Deliberately built on GeneticAlgorithmBase rather
// than a narrower per-algorithm type: GeneticProgrammingAlgorithm and NSGA2
// both derive from it and share this exact surface, so a probe written
// against ProbeContext works for either without modification.
//
// GrammarEnumerationAlgorithm is intentionally not supported here: it has no
// population and no Parents()/Offspring(), and its ReportCallback fires per
// budget level, not per generation - forcing it under this interface would
// be the wrong abstraction rather than a missing feature.
class ProbeContext {
public:
    ProbeContext(GeneticAlgorithmBase const& algo, ResultRecord& record)
        : algo_(algo)
        , record_(record)
    {
    }

    [[nodiscard]] auto Algorithm() const -> GeneticAlgorithmBase const& { return algo_; }
    [[nodiscard]] auto Generation() const -> std::size_t { return algo_.Generation(); }
    [[nodiscard]] auto Parents() const { return algo_.Parents(); }
    [[nodiscard]] auto Offspring() const { return algo_.Offspring(); }
    [[nodiscard]] auto Config() const { return algo_.GetConfig(); }
    [[nodiscard]] auto Problem() const -> Operon::Problem const* { return algo_.GetProblem(); }

    // Sets (or overwrites) a named value in this generation's record.
    auto Emit(std::string key, ResultValue value) -> void
    {
        record_.insert_or_assign(std::move(key), std::move(value));
    }

private:
    // Reference members are intentional: ProbeContext is a throwaway view
    // constructed fresh by ProbeChain once per generation and passed by
    // reference to each probe in turn - it never outlives that call and is
    // never copied or stored, so there's no dangling-reference hazard to
    // guard against by switching to pointers.
    GeneticAlgorithmBase const& algo_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    ResultRecord& record_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

// A pluggable, per-generation diagnostic. Deliberately one interface, not
// split by output kind (e.g. "emits a metric" vs "writes an artifact"): a
// probe may do either or both via the same call - see PopulationTraceProbe
// (writes its own file) vs CacheHitRateProbe (only calls Emit) vs
// StructuralDiversityProbe (could do both) in the sibling headers.
struct GenerationProbe {
    virtual ~GenerationProbe() = default;
    GenerationProbe() = default;
    GenerationProbe(GenerationProbe const&) = default;
    GenerationProbe(GenerationProbe&&) noexcept = default;
    auto operator=(GenerationProbe const&) -> GenerationProbe& = default;
    auto operator=(GenerationProbe&&) noexcept -> GenerationProbe& = default;

    // Called once per generation, subject to the scheduling interval a
    // ProbeChain applies before invoking this.
    virtual auto operator()(ProbeContext& ctx) -> void = 0;

    // Called once after the algorithm's Run() returns, for probes that
    // buffer state or hold an open file (e.g. flushing/closing a trace).
    virtual auto Finish() -> void { }
};

} // namespace Operon

#endif
