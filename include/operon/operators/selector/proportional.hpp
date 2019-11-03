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

#ifndef PROPORTIONAL_SELECTOR_HPP
#define PROPORTIONAL_SELECTOR_HPP

#include <algorithm>
#include <execution>
#include <random>
#include <vector>

#include "core/operator.hpp"
#include "gsl/span"

namespace Operon {
template <typename T, gsl::index Idx, bool Max>
class ProportionalSelector : public SelectorBase<T, Idx, Max> {
public:
    gsl::index operator()(operon::rand_t& random) const override
    {
        std::uniform_real_distribution<operon::scalar_t> uniformReal(0, fitness.back().first - std::numeric_limits<double>::epsilon());
        return std::lower_bound(fitness.begin(), fitness.end(), std::make_pair(uniformReal(random), 0L), std::less {})->second;
    }

    void Prepare(const gsl::span<const T> pop) const override
    {
        SelectorBase<T, Idx, Max>::Prepare(pop);
        Prepare();
    }

private:
    void Prepare() const 
    {
        fitness.clear();
        fitness.reserve(this->population.size());

        operon::scalar_t vmin = this->population[0][Idx], vmax = vmin;
        for (gsl::index i = 0; i < this->population.size(); ++i) {
            auto f = this->population[i][Idx];
            fitness.push_back({ f, i });
            vmin = std::min(vmin, f);
            vmax = std::max(vmax, f);
        }
        auto prepare = [=](auto p) {
            auto f = p.first;
            if constexpr (Max)
                return std::make_pair(f - vmin, p.second);
            else
                return std::make_pair(vmax - f, p.second);
        };
        std::transform(fitness.begin(), fitness.end(), fitness.begin(), prepare);
        std::sort(fitness.begin(), fitness.end());
        std::inclusive_scan(std::execution::seq, fitness.begin(), fitness.end(), fitness.begin(), [](auto lhs, auto rhs) { return std::make_pair(lhs.first + rhs.first, rhs.second); });
    }

    // discrete CDF of the population fitness values
    mutable std::vector<std::pair<operon::scalar_t, gsl::index>> fitness;
};
} // namespace Operon

#endif

