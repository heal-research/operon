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
            T const* Individuals;
            size_t Size;

            inline T& operator[](size_t i) { return Individuals[i]; }
            inline const T& operator[](size_t i) const { return Individuals[i]; }
        };
    }

    template<typename T, size_t I = 0>
    class TournamentSelector : public SelectorBase<T, I>
    {
        public:
            TournamentSelector(size_t s, bool m) : TournamentSelector({ nullptr, 0UL }, s, m) { }

            TournamentSelector(const std::vector<T>& p, size_t s, bool m) : TournamentSelector(Population { p.data(), p.size() }, s, m) { }  

            TournamentSelector(Population<T> p, size_t s, size_t m) : pop(p), tournamentSize(s), maximization(m) { }

            size_t operator()(RandomDevice& random) const
            {
                std::uniform_int_distribution<size_t> uniformInt(0, pop.Size - 1);
                auto fitness = [&](size_t i) { return pop[i].Fitness[I]; };

                auto best = uniformInt(random);
                assert(best < pop.Size);
                for(size_t j = 1; j < tournamentSize; ++j)
                {
                    auto curr = uniformInt(random);
                    assert(curr < pop.Size);
                    if (maximization != (fitness(best) > fitness(curr)))
                    {
                        best = curr; 
                    }
                }
                return best;
            }

            void Reset(const std::vector<T>& population)
            {
                pop = { population.data(), population.size() };
            }

        private:
            Population<T> pop;
            size_t tournamentSize = 2;
            bool   maximization   = true;
    };

    template<typename T, size_t I = 0>
    class ProportionalSelector : public SelectorBase<T, I>
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

            void Reset(const std::vector<T>& population)
            {
                *this = ProportionalSelector(population, maximization);
            }

        private:
            void Prepare()
            {
                fitness.clear();
                fitness.reserve(pop.Size);
                double vmin = pop[0].Fitness[I], vmax = vmin;
                for (size_t i = 0; i < pop.Size; ++i)
                {
                    auto f = pop[i].Fitness[I];
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

