/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef TOURNAMENT_SELECTOR_HPP
#define TOURNAMENT_SELECTOR_HPP

#include <algorithm>
#include <execution>
#include <random>
#include <vector>

#include "core/operator.hpp"
#include "gsl/span"

namespace Operon {

template <typename T, gsl::index Idx>
class TournamentSelector : public SelectorBase<T, Idx> {
public:
    TournamentSelector(size_t tSize)
        : tournamentSize(tSize)
    {
    }

    gsl::index operator()(operon::rand_t& random) const override
    {
        std::uniform_int_distribution<gsl::index> uniformInt(0, this->population.size() - 1);
        auto best = uniformInt(random);
        for (size_t i = 1; i < tournamentSize; ++i) {
            auto curr = uniformInt(random);
            if (this->population[best][Idx] > this->population[curr][Idx]) {
                best = curr;
            }
        }
        return best;
    }

    void TournamentSize(size_t size) { tournamentSize = size; }
    size_t TournamentSize() const { return tournamentSize; }

private:
    size_t tournamentSize;
};

template <typename T, gsl::index Idx>
class RankTournamentSelector : public SelectorBase<T, Idx> {
public:
    RankTournamentSelector(size_t tSize)
        : tournamentSize(tSize)
    {
    }

    gsl::index operator()(operon::rand_t& random) const override
    {
        std::uniform_int_distribution<gsl::index> uniformInt(0, this->population.size() - 1);
        auto best = uniformInt(random);
        for (size_t i = 1; i < tournamentSize; ++i) {
            auto curr = uniformInt(random);
            if (best < curr) {
                best = curr;
            }
        }
        return best;
    }

    void Prepare(const gsl::span<const T> pop) const override
    {
        SelectorBase<T, Idx>::Prepare(pop);
        indices.resize(pop.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](auto lhs, auto rhs) { return pop[lhs][Idx] < pop[rhs][Idx]; });
    }

    void TournamentSize(size_t size)
    {
        tournamentSize = size;
    }

    size_t TournamentSize() const { return tournamentSize; }

private:
    size_t tournamentSize;
    mutable std::vector<size_t> indices;
};


} // namespace Operon

#endif

