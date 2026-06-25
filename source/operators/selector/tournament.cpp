// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <cstddef>
#include <random>

#include "operon/operators/selector.hpp"
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

} // namespace Operon
