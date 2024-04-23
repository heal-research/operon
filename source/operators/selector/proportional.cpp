// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cstddef>
#include <numeric>
#include <algorithm>
#include <functional>
#include <limits>
#include <random>
#include <span>
#include <utility>
#include <vector>

#include "operon/operators/selector.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"

namespace Operon {

auto ProportionalSelector::operator()(Operon::RandomGenerator& random) const -> size_t
{
    std::uniform_real_distribution<Operon::Scalar> uniformReal(0, fitness_.back().first - std::numeric_limits<Operon::Scalar>::epsilon());
    return std::lower_bound(fitness_.begin(), fitness_.end(), std::make_pair(uniformReal(random), size_t{0}), std::less {})->second;
}

void ProportionalSelector::Prepare(const Operon::Span<const Individual> pop) const
{
    SelectorBase::Prepare(pop);
    Prepare();
}

void ProportionalSelector::Prepare() const
{
    fitness_.clear();
    fitness_.reserve(Population().size());

    const size_t populationSize = Population().size();
    auto population = Population();

    Operon::Scalar vmin = Population().front()[idx_];
    Operon::Scalar vmax = vmin;

    for (size_t i = 0; i < populationSize; ++i) {
        auto f = population[i][idx_];
        fitness_.emplace_back( f, i );
        vmin = std::min(vmin, f);
        vmax = std::max(vmax, f);
    }
    auto prepare = [=](auto p) { return std::make_pair(vmax - p.first, p.second); };
    std::transform(fitness_.begin(), fitness_.end(), fitness_.begin(), prepare);
    std::sort(fitness_.begin(), fitness_.end());
    std::inclusive_scan(fitness_.begin(), fitness_.end(), fitness_.begin(), [](auto lhs, auto rhs) { return std::make_pair(lhs.first + rhs.first, rhs.second); });
}
}  // namespace Operon
