/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
 */

#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "core/eval.hpp"
#include "core/metrics.hpp"
#include "core/operator.hpp"
#include "core/stat/pearson.hpp"
#include "core/stat/meanvariance.hpp"

namespace Operon {
template <typename T>
class NormalizedMeanSquaredErrorEvaluator : public EvaluatorBase<T> {
public:
    static constexpr bool Maximization = false;

    NormalizedMeanSquaredErrorEvaluator(Problem& problem)
        : EvaluatorBase<T>(problem)
    {
    }

    operon::scalar_t operator()(operon::rand_t&, T& ind) const
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

        auto estimatedValues = Evaluate<operon::scalar_t>(genotype, dataset, trainingRange);
        auto nmse = NormalizedMeanSquaredError(estimatedValues, targetValues);
        if (!std::isfinite(nmse)) {
            nmse = std::numeric_limits<operon::scalar_t>::max();
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

    operon::scalar_t operator()(operon::rand_t&, T& ind) const
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

        auto estimatedValues = Evaluate<operon::scalar_t>(genotype, dataset, trainingRange);
        auto r2 = RSquared(estimatedValues, targetValues);
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
