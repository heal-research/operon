#ifndef RECOMBINATOR_HPP
#define RECOMBINATOR_HPP

#include "core/operator.hpp"
#include "core/eval.hpp"

namespace Operon
{
    // TODO: think of a way to eliminate duplicated code between the different recombinators
    template<typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
    class BasicRecombinator : public RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>
    {
        public:
            explicit BasicRecombinator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut) : RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut) { }

            using T = typename TSelector::SelectableType;
            std::optional<T> operator()(operon::rand_t& random, double pCrossover, double pMutation)
            {
                std::uniform_real_distribution<double> uniformReal;

                constexpr bool Max       = TSelector::Maximization;
                constexpr gsl::index Idx = TSelector::SelectableIndex;

                auto population = this->Selector().Population();

                auto first = this->selector(random);

                typename TSelector::SelectableType child;

                bool doCrossover = uniformReal(random) < pCrossover;
                bool doMutation  = uniformReal(random) < pMutation;

                if (!(doCrossover || doMutation)) return std::nullopt;

                if (doCrossover)
                {
                    auto second = this->selector(random);
                    child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
                }

                if (doMutation)
                {
                    if (!doCrossover)
                    {
                        // if no crossover was performed we take a copy of the first parent and mutate it
                        child.Genotype = population[first].Genotype;
                    }
                    this->mutator(random, child.Genotype);
                }

                auto f = this->evaluator(random, child);
                child.Fitness[Idx] = ceres::IsFinite(f) ? f : (Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max());

                return std::make_optional(child);
            }

            void Prepare(const gsl::span<const T> pop) override
            {
                this->Selector().Prepare(pop);
            }
    };

    template<typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
    class PlusRecombinator : public RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>
    {
        public:
            explicit PlusRecombinator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut) : RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut) { } 

            using T = typename TSelector::SelectableType;
            std::optional<T> operator()(operon::rand_t& random, double pCrossover, double pMutation) 
            {
                std::uniform_real_distribution<double> uniformReal;
                bool doCrossover = uniformReal(random) < pCrossover;
                bool doMutation  = uniformReal(random) < pMutation;

                if (!(doCrossover || doMutation)) return std::nullopt;

                constexpr bool Max       = TSelector::Maximization;
                constexpr gsl::index Idx = TSelector::SelectableIndex;

                auto population = this->Selector().Population();

                auto first = this->selector(random);
                auto second = this->selector(random);

                typename TSelector::SelectableType child;

                if (doCrossover)
                {
                    child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);
                }

                if (doMutation)
                {
                    // make a copy
                    if (!doCrossover)
                    {
                        // if no crossover was performed we take a copy of the first parent and mutate it
                        child.Genotype = population[first].Genotype;
                    }
                    this->mutator(random, child.Genotype);
                }

                auto f = this->evaluator(random, child);
                child.Fitness[Idx] = ceres::IsFinite(f) ? f : (Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max());

                if (doCrossover)
                {
                    // we have two parents
                    if (Max && child.Fitness[Idx] < std::max(population[first].Fitness[Idx], population[second].Fitness[Idx]))
                    {
                        child = population[first].Fitness[Idx] > population[second].Fitness[Idx] ? population[first] : population[second];
                    }
                    else if (!Max && child.Fitness[Idx] > std::min(population[first].Fitness[Idx], population[second].Fitness[Idx]))
                    {
                        child = population[first].Fitness[Idx] < population[second].Fitness[Idx] ? population[first] : population[second];
                    }
                }
                else 
                {
                    // we have one parent
                    if (Max && child.Fitness[Idx] < population[first].Fitness[Idx])
                    {
                        child = population[first];
                    }
                    else if (!Max && child.Fitness[Idx] > population[first].Fitness[Idx])
                    {
                        child = population[first];
                    }
                }

                return std::make_optional(child);
            }

            void Prepare(const gsl::span<const T> pop) override
            {
                this->Selector().Prepare(pop);
            }
    };

    template<typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
    class BroodRecombinator : public RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>
    {
        public:
            explicit BroodRecombinator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut) : RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut) { }

            using T = typename TSelector::SelectableType;
            std::optional<T> operator()(operon::rand_t& random, double pCrossover, double pMutation) 
            {
                std::uniform_real_distribution<double> uniformReal;

                constexpr bool Max       = TSelector::Maximization;
                constexpr gsl::index Idx = TSelector::SelectableIndex;

                auto population = this->Selector().Population();
                auto first = this->selector(random);
                auto second = this->selector(random);

                T child;

                std::vector<T> brood;
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
                            // make a copy
                            if (!doCrossover)
                            {
                                // if no crossover was performed we take a copy of the first parent and mutate it
                                child.Genotype = population[first].Genotype;
                            }
                            this->mutator(random, child.Genotype);
                        }
                    }
                    brood.push_back(child);
                }

                auto eval = [&](gsl::index idx) {
                    auto f = this->evaluator(random, brood[idx]);
                    brood[idx].Fitness[Idx] = ceres::IsFinite(f) ? f : (Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max());
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

            void Prepare(const gsl::span<const T> pop) override
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

    template<typename TEvaluator, typename TSelector, typename TCrossover, typename TMutator>
    class OffspringSelectionRecombinator : public RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>
    {
        public:
            explicit OffspringSelectionRecombinator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut) : RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut) { }

            using T = typename TSelector::SelectableType;
            std::optional<T> operator()(operon::rand_t& random, double pCrossover, double pMutation) 
            {
                std::uniform_real_distribution<double> uniformReal;
                bool doCrossover = uniformReal(random) < pCrossover;
                bool doMutation  = uniformReal(random) < pMutation;

                if (!(doCrossover || doMutation)) return std::nullopt;

                constexpr bool Max       = TSelector::Maximization;
                constexpr gsl::index Idx = TSelector::SelectableIndex;

                auto population = this->Selector().Population();

                auto first = this->selector(random);
                auto fit   = population[first].Fitness[Idx];

                typename TSelector::SelectableType child;

                if (doCrossover)
                {
                    auto second = this->selector(random);
                    child.Genotype = this->crossover(random, population[first].Genotype, population[second].Genotype);

                    if constexpr(TSelector::Maximization) { fit = std::max(fit, population[second].Fitness[Idx]); }
                    else                                  { fit = std::min(fit, population[second].Fitness[Idx]); } 
                }

                if (doMutation)
                {
                    // make a copy
                    if (!doCrossover)
                    {
                        // if no crossover was performed we take a copy of the first parent and mutate it
                        child.Genotype = population[first].Genotype;
                    }
                    this->mutator(random, child.Genotype);
                }

                auto f = this->evaluator(random, child);
                child.Fitness[Idx] = ceres::IsFinite(f) ? f : (Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max());

                if ((Max && child.Fitness[Idx] > fit) || (!Max && child.Fitness[Idx] < fit))
                {
                    return std::make_optional(child);
                }
                return std::nullopt;
            }

            void Prepare(const gsl::span<const T> pop) override
            {
                this->Selector().Prepare(pop);
            }
    };

}

#endif

