// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_CACHE_HIT_RATE_HPP
#define OPERON_ALGORITHMS_PROBES_CACHE_HIT_RATE_HPP

#include <cstddef>
#include <cstdint>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/hash/zobrist.hpp"

namespace Operon {

// Reports per-generation deltas of the algorithm's Zobrist transposition
// cache (GeneticAlgorithmConfig::Cache): hits, lookups, hit rate, and the
// cache's cumulative size. Emits nothing if no cache is configured.
class CacheHitRateProbe final : public GenerationProbe {
public:
    auto operator()(ProbeContext& ctx) -> void override
    {
        auto const* cache = ctx.Config().Cache;
        if (cache == nullptr) {
            return;
        }

        // The baseline belongs to one cache within one run. It restarts from zero (the counters accumulated since the
        // cache was created or last cleared) when the active cache is a different object than the previously sampled
        // one (OffspringGeneratorBase::SetCache), and when the generation counter went backwards (the probe is being
        // reused for a new run, whose cache may have been cleared and caught up with the old baseline already).
        auto const generation = ctx.Generation();
        if (cache != cache_ || generation < prevGeneration_) {
            prevHits_ = 0;
            prevLookups_ = 0;
        }
        cache_ = cache;
        prevGeneration_ = generation;

        auto const hits = cache->Hits();
        auto const lookups = cache->Lookups();
        // A caller may Clear() the cache between generations (resetting its
        // counters to 0), which would otherwise underflow these unsigned
        // deltas into a huge value; treat a backward jump as "counting
        // resumed from zero" instead.
        auto const deltaHits = hits >= prevHits_ ? hits - prevHits_ : hits;
        auto const deltaLookups = lookups >= prevLookups_ ? lookups - prevLookups_ : lookups;
        prevHits_ = hits;
        prevLookups_ = lookups;

        ctx.Emit("cache_hits", static_cast<std::int64_t>(deltaHits));
        ctx.Emit("cache_lookups", static_cast<std::int64_t>(deltaLookups));
        ctx.Emit("cache_hit_rate",
            deltaLookups != 0 ? static_cast<double>(deltaHits) / static_cast<double>(deltaLookups) : 0.0);
        ctx.Emit("cache_size", static_cast<std::int64_t>(cache->Size()));
    }

private:
    Zobrist const* cache_ { nullptr }; // identity only, never dereferenced; the cache the baseline belongs to
    std::size_t prevGeneration_ { 0 };
    std::size_t prevHits_ { 0 };
    std::size_t prevLookups_ { 0 };
};

} // namespace Operon

#endif
