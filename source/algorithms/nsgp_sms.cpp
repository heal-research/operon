// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <limits>

#include "operon/algorithms/nsgp_sms.hpp"
#include "operon/core/contracts.hpp"

namespace Operon {

// Assigns hypervolume contribution as the Distance field for each individual.
// For a 2-objective front sorted by obj0 ascending (obj1 descending on a
// proper Pareto front), the contribution of interior point i is:
//   (obj0[i+1] - obj0[i]) * (obj1[i-1] - obj1[i])
// Boundary points receive infinity so they are never discarded by CrowdedComparison.
auto NSGPSMS::UpdateDistance(Operon::Span<Individual> pop) -> void
{
    ENSURE(!pop.empty() && pop.front().Fitness.size() == 2);

    auto const inf = std::numeric_limits<Operon::Scalar>::infinity();
    auto const& fronts = Fronts();

    for (size_t fi = 0; fi < fronts.size(); ++fi) {
        auto const& front = fronts[fi];

        for (auto idx : front) {
            pop[idx].Rank = fi;
            pop[idx].Distance = 0;
        }

        if (front.size() <= 2) {
            for (auto idx : front) { pop[idx].Distance = inf; }
            continue;
        }

        // sort by obj0 ascending; on a Pareto front this gives obj1 descending
        auto sorted = front;
        std::stable_sort(sorted.begin(), sorted.end(), [&](auto a, auto b) {
            return pop[a][0] < pop[b][0];
        });

        pop[sorted.front()].Distance = inf;
        pop[sorted.back()].Distance  = inf;

        for (size_t j = 1; j < sorted.size() - 1; ++j) {
            auto const dx = pop[sorted[j + 1]][0] - pop[sorted[j]][0];     // ≥ 0
            auto const dy = pop[sorted[j - 1]][1] - pop[sorted[j]][1];     // ≥ 0 on a Pareto front
            pop[sorted[j]].Distance = dx * dy;
        }
    }
}

} // namespace Operon
