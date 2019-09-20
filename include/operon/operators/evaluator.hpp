#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "core/eval.hpp"
#include "core/operator.hpp"
#include "core/stats.hpp"
#include "core/metrics.hpp"

namespace Operon {
template <typename T>
class NormalizedMeanSquaredErrorEvaluator : public EvaluatorBase<T> {
public:
    static constexpr bool Maximization = false;

    NormalizedMeanSquaredErrorEvaluator(Problem& problem)
        : EvaluatorBase<T>(problem)
    {
    }

    double operator()(operon::rand_t&, T& ind) const
    {
        ++this->fitnessEvaluations;
        auto& problem = this->problem.get();
        auto& dataset = problem.GetDataset();
        auto& genotype = ind.Genotype;

        auto trainingRange = problem.TrainingRange();
        auto targetValues = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start, trainingRange.Size());

        if (this->iterations > 0) {
            auto summary = OptimizeAutodiff(genotype, dataset, targetValues, trainingRange, this->iterations);
            this->localEvaluations += summary.iterations.size();
        }

        auto estimatedValues = Evaluate<double>(genotype, dataset, trainingRange);
        auto nmse = NormalizedMeanSquaredError(estimatedValues.begin(), estimatedValues.end(), targetValues.begin());
        if (!std::isfinite(nmse)) {
            nmse = std::numeric_limits<double>::max();
        }
        return nmse;
    }

    void Prepare(const gsl::span<const T> pop)
    {
        this->population = pop;
    }
};

template <typename T>
class RSquaredEvaluator : public EvaluatorBase<T> {
public:
    static constexpr bool Maximization = true;

    RSquaredEvaluator(Problem& problem)
        : EvaluatorBase<T>(problem)
    {
    }

    double operator()(operon::rand_t&, T& ind) const
    {
        ++this->fitnessEvaluations;
        auto& problem = this->problem.get();
        auto& dataset = problem.GetDataset();
        auto& genotype = ind.Genotype;

        auto trainingRange = problem.TrainingRange();
        auto targetValues = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start(), trainingRange.Size());

        if (this->iterations > 0) {
            auto summary = OptimizeAutodiff(genotype, dataset, targetValues, trainingRange, this->iterations);
            this->localEvaluations += summary.iterations.size();
        }

        auto estimatedValues = Evaluate<double>(genotype, dataset, trainingRange);
        auto r2 = RSquared(estimatedValues.begin(), estimatedValues.end(), targetValues.begin());
        if (!std::isfinite(r2)) {
            r2 = 0;
        }
        return r2;
    }

    void Prepare(const gsl::span<const T> pop)
    {
        this->population = pop;
    }
};
}

#endif
