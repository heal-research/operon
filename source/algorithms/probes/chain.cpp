// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/probes/chain.hpp"

#include <fstream>
#include <utility>

#include <fmt/core.h>
#include <glaze/glaze.hpp>

namespace Operon {

// ---- JsonlSink ----

struct JsonlSink::Impl {
    std::ofstream Out;
    std::string Path;
};

JsonlSink::JsonlSink(std::string_view path)
    : impl_(std::make_unique<Impl>())
{
    impl_->Path = std::string(path);
    // Truncates by default rather than appending, to avoid silently mixing
    // an old trace's lines into a new run.
    impl_->Out.open(impl_->Path, std::ios::out | std::ios::trunc);
    if (!impl_->Out) {
        fmt::print(stderr, "JsonlSink: failed to open '{}' for writing\n", impl_->Path);
    }
}

JsonlSink::~JsonlSink() = default;
JsonlSink::JsonlSink(JsonlSink&&) noexcept = default;
auto JsonlSink::operator=(JsonlSink&&) noexcept -> JsonlSink& = default;

auto JsonlSink::Write(std::size_t /*generation*/, ResultRecord const& record) -> void
{
    if (!impl_->Out) { return; }
    // No glz::meta needed: ResultRecord's map and ResultValue's variant are
    // both generically supported by glaze, unlike the proxies in
    // core/serialization.cpp.
    auto result = glz::write_json(record);
    if (!result) {
        fmt::print(stderr, "JsonlSink: serialization error: {}\n", glz::format_error(result.error()));
        return;
    }
    impl_->Out << *result << '\n';
}

auto JsonlSink::Flush() -> void
{
    if (impl_->Out) { impl_->Out.flush(); }
}

auto JsonlSink::IsOpen() const -> bool
{
    return static_cast<bool>(impl_->Out);
}

// ---- ProbeChain ----

auto ProbeChain::Add(std::unique_ptr<GenerationProbe> probe, std::size_t every, std::size_t offset) -> void
{
    entries_.push_back(Entry{ .Probe = std::move(probe), .Every = every, .Offset = offset });
}

auto ProbeChain::SetSink(std::unique_ptr<RecordSink> sink) -> void
{
    sink_ = std::move(sink);
}

auto ProbeChain::operator()(GeneticAlgorithmBase const& algo) -> void
{
    auto const generation = algo.Generation();
    ResultRecord record;
    record.insert_or_assign("generation", static_cast<std::int64_t>(generation));
    record.insert_or_assign("elapsed", algo.Elapsed());

    ProbeContext ctx{ algo, record };
    bool ran = false;
    for (auto& entry : entries_) {
        if (entry.Every == 0) { continue; }
        if (generation < entry.Offset) { continue; }
        if ((generation - entry.Offset) % entry.Every != 0) { continue; }
        (*entry.Probe)(ctx);
        ran = true;
    }
    if (ran && sink_) { sink_->Write(generation, record); }
}

auto ProbeChain::Finish() -> void
{
    if (finished_) { return; }
    finished_ = true;
    for (auto& entry : entries_) { entry.Probe->Finish(); }
    if (sink_) { sink_->Flush(); }
}

} // namespace Operon
