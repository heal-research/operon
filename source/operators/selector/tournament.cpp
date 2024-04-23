// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cstddef>
#include <numeric>
#include <algorithm>
#include <random>
#include <span>
#include <vector>

#include "operon/operators/selector.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/types.hpp"

namespace Operon {

auto TournamentSelector::operator()(Operon::RandomGenerator& random) const -> size_t
{
    auto population = Population();
    std::uniform_int_distribution<size_t> uniformInt(0, population.size() - 1);
    auto best = uniformInt(random);
    auto tournamentSize = GetTournamentSize();

    for (size_t i = 1; i < tournamentSize; ++i) {
        auto curr = uniformInt(random);
        if (this->Compare(population[curr], population[best])) {
            best = curr;
        }
    }
    return best;
}

auto RankTournamentSelector::operator()(Operon::RandomGenerator& random) const -> size_t
{
    auto population = Population();
    std::uniform_int_distribution<size_t> uniformInt(0, population.size() - 1);
    auto best = uniformInt(random);
    auto tournamentSize = GetTournamentSize();

    for (size_t i = 1; i < tournamentSize; ++i) {
        auto curr = uniformInt(random);
        if (best < curr) {
            best = curr;
        }
    }
    return best;
}

void RankTournamentSelector::Prepare(const Operon::Span<const Individual> pop) const
{
    SelectorBase::Prepare(pop);
    indices_.resize(pop.size());
    std::iota(indices_.begin(), indices_.end(), 0);
    std::sort(indices_.begin(), indices_.end(), [&](auto i, auto j) { return Compare(pop[i], pop[j]); });
}
} // namespace Operon
