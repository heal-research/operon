// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <cmath>
#include <limits>

#include "operon/algorithms/nsgp2.hpp"

namespace Operon {

auto NSGP2::UpdateDistance(Operon::Span<Individual> pop) -> void
{
    size_t const m = pop.front().Fitness.size();
    auto const inf = std::numeric_limits<Operon::Scalar>::max();
    auto const& fronts = Fronts();

    for (size_t i = 0; i < fronts.size(); ++i) {
        auto const& front = fronts[i];
        for (size_t obj = 0; obj < m; ++obj) {
            SingleObjectiveComparison comp(obj);
            // work on a sorted copy of the front indices for this objective
            auto sorted = front;
            std::stable_sort(sorted.begin(), sorted.end(), [&](auto a, auto b) -> auto { return comp(pop[a], pop[b]); });

            auto const fmin = pop[sorted.front()][obj];
            auto const fmax = pop[sorted.back()][obj];

            for (size_t j = 0; j < sorted.size(); ++j) {
                auto const idx = sorted[j];
                pop[idx].Rank = i;
                if (obj == 0) { pop[idx].Distance = 0; }

                if (j == 0 || j == sorted.size() - 1) {
                    pop[idx].Distance += inf;
                } else {
                    auto const mPrev = pop[sorted[j - 1]][obj];
                    auto const mNext = pop[sorted[j + 1]][obj];
                    auto const range = fmax - fmin;
                    auto const distance = range > 0 ? (mNext - mPrev) / range : 0;
                    pop[idx].Distance += std::isfinite(distance) ? distance : 0;
                }
            }
        }
    }
}

} // namespace Operon
