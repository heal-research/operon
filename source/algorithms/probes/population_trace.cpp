// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/probes/population_trace.hpp"

#include <cstdint>
#include <fstream>

#include <fmt/core.h>

#include "operon/core/serialization.hpp"

namespace Operon {

struct PopulationTraceProbe::Impl {
    std::ofstream Out;
};

PopulationTraceProbe::PopulationTraceProbe(std::string_view path)
    : impl_(std::make_unique<Impl>())
{
    impl_->Out.open(std::string(path), std::ios::out | std::ios::binary | std::ios::trunc);
    if (!impl_->Out) {
        fmt::print(stderr, "PopulationTraceProbe: failed to open '{}' for writing\n", path);
    }
}

PopulationTraceProbe::~PopulationTraceProbe() = default;
PopulationTraceProbe::PopulationTraceProbe(PopulationTraceProbe&&) noexcept = default;
auto PopulationTraceProbe::operator=(PopulationTraceProbe&&) noexcept -> PopulationTraceProbe& = default;

auto PopulationTraceProbe::operator()(ProbeContext& ctx) -> void
{
    if (!impl_->Out) { return; }

    auto const bytes = Serialization::ToBeve(ctx.Parents());
    auto const generation = static_cast<std::uint64_t>(ctx.Generation());
    auto const length = static_cast<std::uint64_t>(bytes.size());

    impl_->Out.write(reinterpret_cast<char const*>(&generation), sizeof(generation));
    impl_->Out.write(reinterpret_cast<char const*>(&length), sizeof(length));
    impl_->Out.write(bytes.data(), static_cast<std::streamsize>(bytes.size()));

    ctx.Emit("trace_bytes", static_cast<std::int64_t>(bytes.size()));
}

auto PopulationTraceProbe::Finish() -> void
{
    if (impl_->Out) { impl_->Out.flush(); }
}

auto PopulationTraceProbe::IsOpen() const -> bool
{
    return static_cast<bool>(impl_->Out);
}

} // namespace Operon
