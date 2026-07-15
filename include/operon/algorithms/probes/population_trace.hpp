// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_POPULATION_TRACE_HPP
#define OPERON_ALGORITHMS_PROBES_POPULATION_TRACE_HPP

#include <memory>
#include <string_view>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// Appends the population (ctx.Parents()) to one binary trace file each
// invocation, as [u64 generation][u64 byte-length][BEVE bytes] frames - an
// offline (generation x population) matrix for structural/diversity
// analysis. Uses Serialization::ToBeve, so glaze stays confined to the .cpp,
// same as core/serialization.cpp.
class OPERON_EXPORT PopulationTraceProbe final : public GenerationProbe {
public:
    explicit PopulationTraceProbe(std::string_view path);
    ~PopulationTraceProbe() override;
    PopulationTraceProbe(PopulationTraceProbe const&) = delete;
    PopulationTraceProbe(PopulationTraceProbe&&) noexcept;
    auto operator=(PopulationTraceProbe const&) -> PopulationTraceProbe& = delete;
    auto operator=(PopulationTraceProbe&&) noexcept -> PopulationTraceProbe&;

    auto operator()(ProbeContext& ctx) -> void override;
    auto Finish() -> void override;

    // False if the file failed to open - operator() then silently no-ops.
    [[nodiscard]] auto IsOpen() const -> bool;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

} // namespace Operon

#endif
