// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/probes/registry.hpp"

#include <stdexcept>

#include <fmt/format.h>

#include "operon/algorithms/probes/cache_hit_rate.hpp"
#include "operon/algorithms/probes/diversity.hpp"
#include "operon/algorithms/probes/population_trace.hpp"
#include "operon/core/constants.hpp"

namespace Operon {

auto RegisterBuiltinProbes(ProbeRegistry& registry) -> void
{
    registry.Register("population_trace", [](ProbeParams const& params) -> std::unique_ptr<GenerationProbe> {
        if (!params.contains("path")) {
            throw std::runtime_error("population_trace probe requires a 'path' param");
        }
        return std::make_unique<PopulationTraceProbe>(params.at("path").Get<std::string>());
    });

    registry.Register("cache_hit_rate", [](ProbeParams const& /*params*/) -> std::unique_ptr<GenerationProbe> {
        return std::make_unique<CacheHitRateProbe>();
    });

    registry.Register("structural_diversity", [](ProbeParams const& params) -> std::unique_ptr<GenerationProbe> {
        auto mode = HashMode::Strict;
        if (params.contains("hash_mode")) {
            auto const& m = params.at("hash_mode").Get<std::string>();
            if (m == "relaxed") {
                mode = HashMode::Relaxed;
            } else if (m != "strict") {
                throw std::runtime_error(fmt::format("structural_diversity: unknown hash_mode '{}' (expected 'strict' or 'relaxed')", m));
            }
        }
        return std::make_unique<StructuralDiversityProbe>(mode);
    });
}

} // namespace Operon
