// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_ALGORITHMS_PROBES_FITNESS_STATS_HPP
#define OPERON_ALGORITHMS_PROBES_FITNESS_STATS_HPP

#include <algorithm>
#include <cstdint>
#include <vector>

#include "operon/algorithms/probes/probe.hpp"
#include "operon/core/comparison.hpp"

namespace Operon {

// Reports per-generation best/median/worst training fitness (objective 0)
// and best/median/worst tree length across ctx.Parents(), so a
// --probes-config run can produce a convergence curve (fitness and size
// over generations) without decoding population_trace's binary frames.
class FitnessStatsProbe final : public GenerationProbe {
public:
    auto operator()(ProbeContext& ctx) -> void override
    {
        auto parents = ctx.Parents();
        if (parents.empty()) { return; }

        fitness_.clear();
        fitness_.reserve(parents.size());
        length_.clear();
        length_.reserve(parents.size());
        for (auto const& ind : parents) {
            fitness_.push_back(static_cast<double>(ind[0]));
            length_.push_back(static_cast<std::int64_t>(ind.Genotype.Length()));
        }

        std::ranges::sort(fitness_, Operon::Less<true>{});
        std::ranges::sort(length_);

        auto const n = fitness_.size();
        auto const fitnessMedian = (n % 2 == 0) ? (fitness_[n/2 - 1] + fitness_[n/2]) / 2 : fitness_[n/2];
        auto const lengthMedian = (n % 2 == 0) ? (static_cast<double>(length_[n/2 - 1]) + static_cast<double>(length_[n/2])) / 2 : static_cast<double>(length_[n/2]);

        ctx.Emit("fitness_best", fitness_.front());
        ctx.Emit("fitness_median", fitnessMedian);
        ctx.Emit("fitness_worst", fitness_.back());
        ctx.Emit("length_min", length_.front());
        ctx.Emit("length_median", lengthMedian);
        ctx.Emit("length_max", length_.back());
    }

private:
    std::vector<double> fitness_;
    std::vector<std::int64_t> length_;
};

} // namespace Operon

#endif
