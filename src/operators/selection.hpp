#ifndef SELECTION_HPP
#define SELECTION_HPP

#include <vector>
#include <random>
#include <algorithm>
#include <execution>
#include "gsl/span"

#include "operator.hpp"

namespace Operon
{
    template<typename T, size_t I = 0>
    class TournamentSelector : public SelectorBase<T, I>
    {
        public:
            TournamentSelector(size_t s, bool m) : TournamentSelector(gsl::span<const T>{ }, s, m) { }
            TournamentSelector(const std::vector<T>& p, size_t s, bool m) : TournamentSelector(gsl::span<const T>(p), s, m) { }  
            TournamentSelector(gsl::span<const T> p, size_t s, size_t m) : pop(p), tournamentSize(s), maximization(m) { }

            size_t operator()(RandomDevice& random) const
            {
                std::uniform_int_distribution<size_t> uniformInt(0, pop.size() - 1);
                auto fitness = [&](size_t i) { return pop[i].Fitness[I]; };

                auto best = uniformInt(random);
                assert(best < pop.size());
                for(size_t j = 1; j < tournamentSize; ++j)
                {
                    auto curr = uniformInt(random);
                    assert(curr < pop.size());
                    if (maximization != (fitness(best) > fitness(curr)))
                    {
                        best = curr; 
                    }
                }
                return best;
            }

            void Reset(const std::vector<T>& population)
            {
                pop = gsl::span<const T>(population);
            }

        private:
            gsl::span<const T> pop;
            size_t tournamentSize = 2;
            bool   maximization   = true;
    };

    template<typename T, size_t I = 0>
    class ProportionalSelector : public SelectorBase<T, I>
    {
        public:
            ProportionalSelector(const gsl::span<const T> p, bool m) : pop(p), maximization(m) { }
            ProportionalSelector(bool m) : ProportionalSelector(gsl::span<const T>{ }, m) { }
            ProportionalSelector(const std::vector<T>& p, bool m) : ProportionalSelector(gsl::span<const T>(p), m) { }
            size_t operator()(RandomDevice& random) const
            {
                std::uniform_real_distribution<double> uniformReal(0, fitness.back() - std::numeric_limits<double>::epsilon());
                auto r = uniformReal(random);
                auto it = std::find_if(fitness.begin(), fitness.end(), [=](double v) { return r < v; });
                return std::distance(fitness.begin(), it);
            }

            void Reset(const std::vector<T>& population)
            {
                pop = gsl::span<const T>(population);    
            }

        private:
            void Prepare()
            {
                fitness.clear();
                fitness.reserve(pop.size());
                double vmin = pop[0].Fitness[I], vmax = vmin;
                for (size_t i = 0; i < pop.size(); ++i)
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

            gsl::span<const T> pop;
            std::vector<double> fitness;
            bool maximization;
    };
}
#endif

