#ifndef GP_HPP
#define GP_HPP

#include <chrono>

#include "core/eval.hpp"
#include "operators/recombinator.hpp"
#include "operators/crossover.hpp"
#include "operators/mutation.hpp"
#include "operators/initialization.hpp"
#include "algorithms/config.hpp"

namespace Operon
{
    template<typename TCreator, typename TRecombinator, typename ExecutionPolicy = std::execution::parallel_unsequenced_policy>
    class GeneticProgrammingAlgorithm
    {
        using T                   = typename TRecombinator::SelectorType::SelectableType;
        static constexpr bool Idx = TRecombinator::SelectorType::SelectableIndex;
        static constexpr bool Max = TRecombinator::SelectorType::Maximization;

        private:
            std::reference_wrapper<const Problem> problem_;
            std::reference_wrapper<const GeneticAlgorithmConfig> config_;

            std::reference_wrapper<const TCreator> creator_;
            std::reference_wrapper<const TRecombinator> recombinator_;

            std::vector<T>         parents;
            std::vector<T>         offspring;

            size_t                 generation;

        public:
            explicit GeneticProgrammingAlgorithm(const Problem& problem, const GeneticAlgorithmConfig& config, const TCreator& creator, const TRecombinator& recombinator) : 
                problem_(problem), config_(config), creator_(creator), recombinator_(recombinator), parents(config.PopulationSize), offspring(config.PopulationSize), generation(0UL) { }

            const gsl::span<const T> Parents() const { return gsl::span<const T>(parents); }
            const gsl::span<const T> Offspring() const { return gsl::span<const T>(offspring); }

            const Problem& GetProblem() const { return problem_.get(); }
            const GeneticAlgorithmConfig& GetConfig() const { return config_.get(); }

            const TCreator& GetCreator() const { return creator_.get(); }
            const TRecombinator& GetRecombinator() const { return recombinator_.get(); }

            size_t Generation() const { return generation; }

            void Reset()
            {
                generation = 0;
            }

            void Run(operon::rand_t& random, std::function<void()> report = nullptr)
            {
                auto& config       = GetConfig();
                auto& creator      = GetCreator();
                auto& recombinator = GetRecombinator();
                auto& problem      = GetProblem();
                auto& grammar      = problem.GetGrammar();
                auto targetValues  = problem.TargetValues();

                // easier to work with indices 
                std::vector<gsl::index> indices(config.PopulationSize);
                std::iota(indices.begin(), indices.end(), 0L);
                // random seeds for each thread
                std::vector<operon::rand_t::result_type> seeds(config.PopulationSize);
                std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

                // flag to signal algorithm termination
                std::atomic_bool terminate = false;

                const auto& inputs = problem.InputVariables();

                thread_local operon::rand_t rndlocal = random;

                const auto worst = Max ? std::numeric_limits<double>::min() : std::numeric_limits<double>::max();
                auto create = [&](gsl::index i)
                {
                    // create one random generator per thread
                    rndlocal.Seed(seeds[i]);
                    parents[i].Genotype     = creator(rndlocal, grammar, inputs);
                    parents[i].Fitness[Idx] = worst;
                };
                const auto& evaluator = recombinator.Evaluator();
                auto evaluate = [&](T& ind) 
                {
                    auto fitness = evaluator(rndlocal, ind);
                    ind.Fitness[Idx] = std::isfinite(fitness) ? fitness : worst;
                };

                ExecutionPolicy executionPolicy;

                std::for_each(executionPolicy, indices.begin(), indices.end(), create);
                std::for_each(executionPolicy, parents.begin(), parents.end(), evaluate);

                std::uniform_real_distribution<double> uniformReal(0, 1); // for crossover and mutation
                for (generation = 0; generation < config.Generations; ++generation)
                {
                    // get some new seeds
                    std::generate(seeds.begin(), seeds.end(), [&](){ return random(); });

                    // preserve one elite
                    auto [minElem, maxElem] = std::minmax_element(parents.begin(), parents.end(), [&](const auto& lhs, const auto& rhs) { return lhs.Fitness[Idx] < rhs.Fitness[Idx]; });
                    auto best = Max ? maxElem : minElem;

                    // this assumes fitness is in [0, 1]
                    if ((Max && std::abs(1 - best->Fitness[Idx]) < 1e-6) || (!Max && std::abs(best->Fitness[Idx]) < 1e-6))
                    {
                        terminate = true;
                    }

                    if (terminate)
                    {
                        return;
                    }

                    if (report)
                    {
                        std::invoke(report);
                    }

                    offspring[0] = *best;
                    recombinator.Prepare(parents);

                    // produce some offspring
                    auto iterate = [&](gsl::index i) 
                    {
                        rndlocal.Seed(seeds[i]);

                        while (!(terminate = recombinator.Terminate())) 
                        {
                            auto recombinant  = recombinator(rndlocal, config.CrossoverProbability, config.MutationProbability);

                            if (recombinant.has_value())
                            {
                                offspring[i]  = std::move(recombinant.value());
                                return;
                            }
                        }
                    };
                    std::for_each(executionPolicy, indices.cbegin() + 1, indices.cend(), iterate);
                    // we check for empty offspring (in case the recombinator terminated early) and fill them with the parents
                    for(auto i : indices)
                    {
                        if (offspring[i].Genotype.Nodes().empty())
                        {
                            offspring[i] = parents[i];
                        }
                    }
                    // the offspring become the parents
                    parents.swap(offspring);
                }
            }
    };
}

#endif

