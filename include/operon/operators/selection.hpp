#ifndef SELECTION_HPP
#define SELECTION_HPP

#include <algorithm>
#include <execution>
#include <random>
#include <vector>

#include "core/operator.hpp"
#include "gsl/span"

namespace Operon {
template <typename T, gsl::index Idx, bool Max>
class TournamentSelector : public SelectorBase<T, Idx, Max> {
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
            if (comparison(this->population[best][Idx], this->population[curr][Idx])) {
                best = curr;
            }
        }
        return best;
    }

    void Prepare(const gsl::span<const T> pop) override
    {
        this->population = gsl::span<const T>(pop);
    }

    void TournamentSize(size_t size) { tournamentSize = size; }
    size_t TournamentSize() const { return tournamentSize; }

private:
    std::conditional_t<Max, std::less<>, std::greater<>> comparison;
    size_t tournamentSize;
};

template <typename T, gsl::index Idx, bool Max>
class RankedTournamentSelector : public SelectorBase<T, Idx, Max> {
public:
    RankedTournamentSelector(size_t tSize)
        : tournamentSize(tSize)
    {
    }

    gsl::index operator()(operon::rand_t& random) const override
    {
        std::uniform_int_distribution<gsl::index> uniformInt(0, this->population.size() - 1);
        auto best = uniformInt(random);
        for (size_t i = 1; i < tournamentSize; ++i) {
            auto curr = uniformInt(random);
            if (best < curr)
                best = curr;
        }
        return indices[best];
    }

    void Prepare(const gsl::span<const T> pop) override
    {
        this->population = gsl::span<const T>(pop);
        indices.resize(pop.size());
        std::iota(indices.begin(), indices.end(), 0);
        std::sort(indices.begin(), indices.end(), [&](auto lhs, auto rhs) { return comparison(pop[lhs][Idx], pop[rhs][Idx]); });
    }

    void TournamentSize(size_t size)
    {
        tournamentSize = size;
    }

    size_t TournamentSize() const { return tournamentSize; }

private:
    size_t tournamentSize;
    std::conditional_t<Max, std::less<>, std::greater<>> comparison;
    std::vector<size_t> indices;
};

template <typename T, gsl::index Idx, bool Max>
class ProportionalSelector : public SelectorBase<T, Idx, Max> {
public:
    gsl::index operator()(operon::rand_t& random) const override
    {
        std::uniform_real_distribution<double> uniformReal(0, fitness.back().first - std::numeric_limits<double>::epsilon());
        return std::lower_bound(fitness.begin(), fitness.end(), std::make_pair(uniformReal(random), 0L), std::less {})->second;
    }

    void Prepare(const gsl::span<const T> pop)
    {
        this->population = gsl::span<const T>(pop);
        Prepare();
    }

private:
    void Prepare()
    {
        fitness.clear();
        fitness.reserve(this->population.size());

        double vmin = this->population[0][Idx], vmax = vmin;
        for (gsl::index i = 0; i < this->population.size(); ++i) {
            auto f = this->population[i].Fitness[Idx];
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
    std::vector<std::pair<double, gsl::index>> fitness;
};

template <typename T, gsl::index Idx, bool Max>
class RandomSelector : public SelectorBase<T, Idx, Max> {
public:
    gsl::index operator()(operon::rand_t& random) const override
    {
        std::uniform_int_distribution<gsl::index> uniformInt(0, this->population.size() - 1);
        return uniformInt(random);
    }

    void Prepare(const gsl::span<const T> pop) override
    {
        this->population = gsl::span<const T>(pop);
    }
};
}
#endif
