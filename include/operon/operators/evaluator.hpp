#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "core/operator.hpp"

namespace Operon
{
    template<typename T>
    class NormalizedMeanSquaredErrorEvaluator : public EvaluatorBase<T>
    {
        public:
            NormalizedMeanSquaredErrorEvaluator(Problem &problem) : EvaluatorBase<T>(problem) { }

            double operator()(operon::rand_t&, T& ind, size_t iter = 0)
            {
                ++this->fitnessEvaluations;
                auto& problem  = this->problem.get();
                auto& dataset  = problem.GetDataset();
                auto& genotype = ind.Genotype;

                auto trainingRange = problem.TrainingRange();
                auto targetValues  = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start, trainingRange.Size());

                if (iter > 0)
                {
                    auto summary = OptimizeAutodiff(genotype, dataset, targetValues, trainingRange, iter);
                    this->localEvaluations += summary.num_unsuccessful_steps + summary.num_successful_steps;
                }

                auto estimatedValues = Evaluate<double>(genotype, dataset, trainingRange);
                return NormalizedMeanSquaredError(estimatedValues.begin(), estimatedValues.end(), targetValues.begin());
            }

            void Prepare(const gsl::span<const T> pop) 
            {
                this->population = pop;
            }
    };
}

#endif

