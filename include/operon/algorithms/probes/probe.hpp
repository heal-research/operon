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
// per-individual/per-variable readings that don't collapse to one scalar.
using ResultValue = std::variant<
    std::int64_t,
    double,
    bool,
    std::string,
    std::vector<std::int64_t>,
    std::vector<double>>;

// One generation's worth of named values, handed to a RecordSink. Not a
// fixed-column struct: probes are free to emit whatever keys apply that
// generation, and the sink (NDJSON) serializes each record self-describingly.
using ResultRecord = Operon::Map<std::string, ResultValue>;

// Read-only view into the running algorithm plus a handle to the current
// generation's record. Built on GeneticAlgorithmBase (not a narrower
// per-algorithm type) so a probe works for both GeneticProgrammingAlgorithm
// and NSGA2 unmodified. GrammarEnumerationAlgorithm has no population and
// reports per budget level, not per generation, so it's out of scope here.
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
    // Reference members are fine here: ProbeContext is a throwaway view
    // built fresh by ProbeChain each generation and never stored or copied.
    GeneticAlgorithmBase const& algo_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    ResultRecord& record_; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
};

// A pluggable, per-generation diagnostic. One interface, not split by output
// kind: a probe may emit a scalar, write its own artifact, or both.
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
