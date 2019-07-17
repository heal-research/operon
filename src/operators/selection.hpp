#ifndef SELECTION_HPP
#define SELECTION_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <execution>

#include "operator.hpp"

namespace Operon
{
    namespace 
    {
        template<typename T>
        struct Population
        {
            T*     Individuals;
            size_t Size;

            inline T& operator[](size_t i) { return Individuals[i]; }
            inline const T& operator[](size_t i) const { return Individuals[i]; }
        };
    }

    template<typename T, size_t Index>
    class TournamentSelector : public SelectorBase
    {
        public:
            TournamentSelector(const std::vector<T>& population, size_t s, bool m) : pop { population.data(), population.size() }, tournamentSize(s), maximization(m) { }  
            size_t operator()(RandomDevice& random) const
            {
                std::uniform_int_distribution<size_t> uniformInt(0, pop.Size - 1);
                auto fitness = [&](size_t i) { return pop[i].Fitness[Index]; };

                auto best = uniformInt(random);
                for(size_t j = 1; j < tournamentSize; ++j)
                {
                    auto curr = uniformInt(random);
                    if (maximization != fitness[best] > fitness[curr])
                    {
                        best = curr; 
                    }
                }
                return best;
            }

        private:
            const  Population<T> pop;
            size_t tournamentSize = 2;
            bool   maximization   = true;
    };

    template<typename T, size_t Index>
    class ProportionalSelector : public SelectorBase
    {
        public:
            ProportionalSelector(const std::vector<T>& population, bool m) : pop{ population.data(), population.size() }, maximization(m) { }
            size_t operator()(RandomDevice& random) const
            {
                std::uniform_real_distribution<double> uniformReal(0, fitness.back() - std::numeric_limits<double>::epsilon());
                auto r = uniformReal(random);
                auto it = std::find_if(fitness.begin(), fitness.end(), [=](double v) { return r < v; });
                return std::distance(fitness.begin(), it);
            }

        private:
            void Prepare()
            {
                fitness.clear();
                fitness.reserve(pop.Size);
                double vmin = pop[0].Fitness[Index], vmax = vmin;
                for (size_t i = 0; i < pop.Size; ++i)
                {
                    auto f = pop.Individuals[i].Fitness[Index];
                    fitness.push_back(f);
                    if (vmin > f) vmin = f;
                    if (vmax < f) vmax = f;
                }
                auto prepare = maximization ? [=](auto f) { return f - vmin; } : [=](auto f) { return vmax - f; };
                std::transform(fitness.begin(), fitness.end(), fitness.begin(), prepare);
                std::inclusive_scan(std::execution::seq, fitness.begin(), fitness.end(), fitness.begin());
            }

            const Population<T> pop;
            std::vector<double> fitness;
            bool maximization;
    };
}

#endif

