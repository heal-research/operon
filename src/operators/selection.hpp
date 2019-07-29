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
    template<typename T, size_t Idx, bool Max>
    class TournamentSelector : public SelectorBase<T, Idx, Max>
    {
        public:
            TournamentSelector(size_t tSize) : tournamentSize(tSize) {} 

            size_t operator()(RandomDevice& random) const
            {
                std::uniform_int_distribution<size_t> uniformInt(0, this->population.size() - 1);

                auto best = uniformInt(random);
                for(size_t j = 1; j < tournamentSize; ++j)
                {
                    auto curr = uniformInt(random);
                    if (this->Compare(best, curr))
                    {
                        best = curr; 
                    }
                }
                return best;
            }

            void Reset(const std::vector<T>& pop) override 
            {
                this->population = gsl::span<const T>(pop);
            }

        private:
            size_t tournamentSize;
    };

    template<typename T, size_t Idx, bool Max>
    class ProportionalSelector : public SelectorBase<T, Idx, Max>
    {
        public:
            size_t operator()(RandomDevice& random) const
            {
                std::uniform_real_distribution<double> uniformReal(0, fitness.back() - std::numeric_limits<double>::epsilon());
                auto r = uniformReal(random);
                auto it = std::find_if(fitness.begin(), fitness.end(), [=](double v) { return r < v; });
                return std::distance(fitness.begin(), it);
            }

            void Reset(const std::vector<T>& pop)
            {
                this->population = gsl::span<const T>(pop);    
                Prepare();
            }

        private:
            void Prepare()
            {
                fitness.clear();
                fitness.reserve(this->population.size());

                double vmin = this->population[0].Fitness[Idx], vmax = vmin;
                for (size_t i = 0; i < this->population.size(); ++i)
                {
                    auto f = this->population[i].Fitness[Idx];
                    fitness.push_back(f);
                    if (vmin > f) vmin = f;
                    if (vmax < f) vmax = f;
                }
                auto prepare = [=](auto f) 
                {  
                    if constexpr (Max) return f - vmin; 
                    else return vmax - f;
                };
                std::transform(fitness.begin(), fitness.end(), fitness.begin(), prepare);
                std::inclusive_scan(std::execution::par_unseq, fitness.begin(), fitness.end(), fitness.begin());
            }

            std::vector<double> fitness;
    };
}
#endif

