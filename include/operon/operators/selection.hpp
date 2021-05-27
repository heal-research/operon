// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef SELECTION_HPP
#define SELECTION_HPP

#include "core/operator.hpp"

namespace Operon {

class TournamentSelector : public SelectorBase {
public:
    explicit TournamentSelector(ComparisonCallback&& cb) 
        : SelectorBase(cb)
        , tournamentSize(5)
    { } 
    explicit TournamentSelector(ComparisonCallback const& cb) 
        : SelectorBase(cb)
        , tournamentSize(5)
    { } 

    size_t operator()(Operon::RandomGenerator& random) const override;
    void SetTournamentSize(size_t size) { tournamentSize = size; }
    size_t GetTournamentSize() const { return tournamentSize; }

private:
    size_t tournamentSize;
};

class RankTournamentSelector : public SelectorBase {
public:
    explicit RankTournamentSelector(ComparisonCallback&& cb) : SelectorBase(cb){ } 
    explicit RankTournamentSelector(ComparisonCallback const& cb) : SelectorBase(cb){ } 

    size_t operator()(Operon::RandomGenerator& random) const override;

    void Prepare(const Operon::Span<const Individual> pop) const override;

    void SetTournamentSize(size_t size) { tournamentSize = size; }

    size_t GetTournamentSize() const { return tournamentSize; }

private:
    size_t tournamentSize;
    mutable std::vector<size_t> indices;
};

class ProportionalSelector : public SelectorBase {
public:
    explicit ProportionalSelector(ComparisonCallback&& cb) : SelectorBase(cb), idx(0) { } 
    explicit ProportionalSelector(ComparisonCallback const& cb) : SelectorBase(cb), idx(0) { } 

    size_t operator()(Operon::RandomGenerator& random) const override;
    
    void Prepare(const Operon::Span<const Individual> pop) const override;

    void SetObjIndex(size_t objIndex) { idx = objIndex; }

private:
    void Prepare() const; 

    // discrete CDF of the population fitness values
    mutable std::vector<std::pair<Operon::Scalar, size_t>> fitness;
    size_t idx = 0;
};

class RandomSelector : public SelectorBase {
public:
    RandomSelector() : SelectorBase(nullptr) { }

    size_t operator()(Operon::RandomGenerator& random) const override
    {
        return std::uniform_int_distribution<size_t>(0, this->population.size() - 1)(random);
    }
};

} //namespace

#endif
