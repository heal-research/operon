#ifndef BROOD_RECOMBINATOR_HPP
#define BROOD_RECOMBINATOR_HPP

#include "core/operator.hpp"

namespace Operon
{
    template<typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
    class BroodRecombinator : public RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>
    {
        public:
            explicit BroodRecombinator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut) : RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut) { }

            using T = typename TSelector::SelectableType;
            std::optional<T> operator()(operon::rand_t& random, double pCrossover, double pMutation) const override 
            {
                std::uniform_real_distribution<double> uniformReal;

                constexpr bool Max       = TSelector::Maximization;
                constexpr gsl::index Idx = TSelector::SelectableIndex;

                auto population = this->Selector().Population();
                auto first = this->selector(random);
                auto second = this->selector(random);

                T child;

                std::vector<T> brood; brood.reserve(broodSize);
                for (size_t i = 0; i < broodSize; ++i)
                {
                    bool doCrossover = uniformReal(random) < pCrossover;
                    bool doMutation  = uniformReal(random) < pMutation;

                    if (!(doCrossover || doMutation)) 
                    {
                        child = population[first];
                    }
                    else
                    {
                        if (doCrossover)
                        {
                            child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
                        }

                        if (doMutation)
                        {
                            child.Genotype = doCrossover 
                                ? this->mutator(random, std::move(child.Genotype))
                                : this->mutator(random, population[first].Genotype);
                        }
                    }
                    brood.push_back(child);
                }

                auto eval = [&](gsl::index idx) {
                    auto f = this->evaluator(random, brood[idx]);
                    brood[idx].Fitness[Idx] = std::isfinite(f) ? f : (Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max());
                };

                std::uniform_int_distribution<gsl::index> uniformInt(0, brood.size() - 1);
                auto bestIdx = uniformInt(random);
                eval(bestIdx);
                for (size_t i = 1; i < broodTournamentSize; ++i)
                {
                    auto currIdx = uniformInt(random);
                    eval(currIdx);
                    if ((Max && brood[bestIdx].Fitness[Idx] < brood[currIdx].Fitness[Idx]) || (!Max && brood[bestIdx].Fitness[Idx] > brood[currIdx].Fitness[Idx]))
                    {
                        bestIdx = currIdx;
                    }
                }
                return std::make_optional(brood[bestIdx]);
            }

            void Prepare(const gsl::span<const T> pop) const override
            {
                this->Selector().Prepare(pop);
            }

            void BroodSize(size_t value) { broodSize = value; }
            size_t BroodSize() const     { return broodSize;  }

            void BroodTournamentSize(size_t value) { broodTournamentSize = value; }
            size_t BroodTournamentSize() const     { return broodTournamentSize;  }

        private:
            size_t broodSize;
            size_t broodTournamentSize;
    };
} // namespace Operon
#endif

