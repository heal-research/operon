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

#ifndef SELECTION_HPP
#define SELECTION_HPP

#include "core/operator.hpp"

namespace Operon {

class TournamentSelector : public SelectorBase {
public:
    explicit TournamentSelector(ComparisonCallback cb) : SelectorBase(cb){ } 

    gsl::index operator()(Operon::Random& random) const override;
    void SetTournamentSize(size_t size) { tournamentSize = size; }
    size_t GetTournamentSize() const { return tournamentSize; }

private:
    size_t tournamentSize;
};

class RankTournamentSelector : public SelectorBase {
public:
    explicit RankTournamentSelector(ComparisonCallback cb) : SelectorBase(cb){ } 

    gsl::index operator()(Operon::Random& random) const override;

    void Prepare(const gsl::span<const Individual> pop) const override;

    void SetTournamentSize(size_t size) { tournamentSize = size; }

    size_t GetTournamentSize() const { return tournamentSize; }

private:
    size_t tournamentSize;
    mutable std::vector<size_t> indices;
};

class ProportionalSelector : public SelectorBase {
public:
    explicit ProportionalSelector(ComparisonCallback cb) : SelectorBase(cb), idx(0) { } 

    gsl::index operator()(Operon::Random& random) const override;
    
    void Prepare(const gsl::span<const Individual> pop) const override;

    void SetObjIndex(gsl::index objIndex) { idx = objIndex; } 

private:
    void Prepare() const; 

    // discrete CDF of the population fitness values
    mutable std::vector<std::pair<Operon::Scalar, gsl::index>> fitness;
    gsl::index idx = 0;
};

class RandomSelector : public SelectorBase {
public:
    RandomSelector() : SelectorBase(nullptr) { }

    gsl::index operator()(Operon::Random& random) const override
    {
        return std::uniform_int_distribution<gsl::index>(0, this->population.size() - 1)(random);
    }
};

} //namespace

#endif
