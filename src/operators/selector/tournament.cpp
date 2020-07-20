#include "operators/selection.hpp"

namespace Operon {
gsl::index
TournamentSelector::operator()(Operon::Random& random) const
{
    std::uniform_int_distribution<gsl::index> uniformInt(0, this->population.size() - 1);
    auto best = uniformInt(random);

    for (size_t i = 1; i < tournamentSize; ++i) {
        auto curr = uniformInt(random);
        if (this->comp(population[curr], population[best])) {
            best = curr;
        }
    }
    return best;
}

gsl::index
RankTournamentSelector::operator()(Operon::Random& random) const
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

void RankTournamentSelector::Prepare(const gsl::span<const Individual> pop) const
{
    SelectorBase::Prepare(pop);
    indices.resize(pop.size());
    std::iota(indices.begin(), indices.end(), 0);
    std::sort(indices.begin(), indices.end(), [&](auto i, auto j) { return this->comp(pop[i], pop[j]); });
}
}
