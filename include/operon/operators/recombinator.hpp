#ifndef RECOMBINATOR_HPP
#define RECOMBINATOR_HPP

#include "core/operator.hpp"
#include "core/eval.hpp"

namespace Operon
{
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
    class BroodRecombinator : public RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>
    {
        public:
            explicit BroodRecombinator(TEvaluator& eval, TSelector& sel, TCrossover& cx, TMutator& mut) : RecombinatorBase<TEvaluator, TSelector, TCrossover, TMutator>(eval, sel, cx, mut) { }

            using T = typename TSelector::SelectableType;
            std::optional<T> operator()(operon::rand_t& random, double pCrossover, double pMutation, size_t broodSize) 
            {
                std::uniform_real_distribution<double> uniformReal;

                constexpr bool Max       = TSelector::Maximization;
                constexpr gsl::index Idx = TSelector::SelectableIndex;

                auto population = this->Selector().Population();

                std::optional<T> best;

                for (size_t i = 0; i < broodSize; ++i)
                {
                    auto first = this->selector(random);
                    auto fit   = population[first].Fitness[Idx];

                    typename TSelector::SelectableType child;

                    bool doCrossover = uniformReal(random) < pCrossover;
                    bool doMutation  = uniformReal(random) < pMutation;

                    if (!(doCrossover || doMutation)) return std::nullopt;
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

                    if (!best.has_value() || (Max && child.Fitness[Idx] > best.value().Fitness[Idx]) || (!Max && child.Fitness[Idx] < best.value().Fitness[Idx]))
                    {
                        best = std::make_optional(child);
                    }
                }

                return best;
            }

            void Prepare(const gsl::span<const T> pop) override
            {
                this->Selector().Prepare(pop);
            }
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

