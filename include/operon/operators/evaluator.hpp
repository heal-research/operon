#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "core/operator.hpp"
#include "core/eval.hpp"

namespace Operon
{
    template<typename T>
    class NormalizedMeanSquaredErrorEvaluator : public EvaluatorBase<T>
    {
        public:
            NormalizedMeanSquaredErrorEvaluator(Problem &problem) : EvaluatorBase<T>(problem) { }

            double operator()(operon::rand_t&, T& ind)
            {
                ++this->fitnessEvaluations;
                auto& problem  = this->problem.get();
                auto& dataset  = problem.GetDataset();
                auto& genotype = ind.Genotype;

                auto trainingRange = problem.TrainingRange();
                auto targetValues  = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start, trainingRange.Size());

                if (this->iterations > 0)
                {
                    auto summary = OptimizeAutodiff(genotype, dataset, targetValues, trainingRange, this->iterations);
                    this->localEvaluations += summary.iterations.size();
                }

                auto estimatedValues = Evaluate<double>(genotype, dataset, trainingRange, nullptr);
                return NormalizedMeanSquaredError(estimatedValues.begin(), estimatedValues.end(), targetValues.begin());
            }

            void Prepare(const gsl::span<const T> pop) 
            {
                this->population = pop;
            }
    };
}

#endif

