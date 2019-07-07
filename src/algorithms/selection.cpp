#include "operators.hpp"

using namespace std;

namespace Operon
{
    template<typename T, size_t Index>
    vector<T> SelectTournament(Rand& random, const std::vector<T>& population, bool maximization[], size_t tournamentSize)
    {
        uniform_int_distribution<size_t> uniformInt(0, population.size() - 1);
        vector<T> selected(population.size());

        auto fitness = [&](size_t i) { return population[i].Fitness[Index]; };

        for(size_t i = 0; i < population.size(); ++i)
        {
            auto best = uniformInt(random);
            for(size_t j = 1; j < tournamentSize; ++j)
            {
                auto curr = uniformInt(random);

                if (maximization != fitness[best] > fitness[curr])
                {
                    best = curr; 
                }
            }

            selected.push_back(population[best]);
        }

        return selected;
    }
}

