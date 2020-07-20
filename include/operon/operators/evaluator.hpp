/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "core/eval.hpp"
#include "core/nnls.hpp"
#include "core/nnls_tiny.hpp"
#include "core/metrics.hpp"
#include "core/operator.hpp"
#include "stat/meanvariance.hpp"
#include "stat/pearson.hpp"

namespace Operon {
class NormalizedMeanSquaredErrorEvaluator : public EvaluatorBase {
public:
    NormalizedMeanSquaredErrorEvaluator(Problem& problem)
        : EvaluatorBase(problem)
    {
    }

    typename EvaluatorBase::ReturnType
    operator()(Operon::Random&, Individual& ind) const override
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

        auto estimatedValues = Evaluate<Operon::Scalar>(genotype, dataset, trainingRange);
        // scale values
        auto [a, b] = LinearScalingCalculator::Calculate(estimatedValues.begin(), estimatedValues.end(), targetValues.begin());
        std::transform(estimatedValues.begin(), estimatedValues.end(), estimatedValues.begin(), [a = a, b = b](Operon::Scalar v) { return b * v + a; });
        auto nmse = NormalizedMeanSquaredError(estimatedValues, targetValues);
        if (!std::isfinite(nmse)) {
            nmse = Operon::Numeric::Max<Operon::Scalar>();
        }
        return nmse;
    }
};

class RSquaredEvaluator : public EvaluatorBase {
public:
    static constexpr Operon::Scalar LowerBound = 0.0;
    static constexpr Operon::Scalar UpperBound = 1.0;

    RSquaredEvaluator(Problem& problem)
        : EvaluatorBase(problem)
    {
    }

    typename EvaluatorBase::ReturnType
    operator()(Operon::Random&, Individual& ind) const
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
            //auto coeff = genotype.GetCoefficients();
            //Eigen::Matrix<double, Eigen::Dynamic, 1> param(coeff.size());

            //// use the tree coefficients as a starting point
            //for (size_t i = 0; i < coeff.size(); ++i) {
            //    param(i) = coeff[i];
            //}

            //TinyCostFunction func(genotype, dataset, targetValues, trainingRange);
            //ceres::TinySolver<TinyCostFunction> solver;
            //solver.options.max_num_iterations = this->iterations;
            //auto summary = solver.Solve(func, &param);

            //for (size_t i = 0; i < coeff.size(); ++i) {
            //   coeff[i] =  param(i);
            //}
            //genotype.SetCoefficients(coeff);

            //this->localEvaluations += summary.iterations;
        }

        auto estimatedValues = Evaluate<Operon::Scalar>(genotype, dataset, trainingRange);

        MeanVarianceCalculator mv;
        mv.Add(estimatedValues);

        auto variance = mv.NaiveVariance();
        
        double r2 = 0;
        if (variance > 1e-12) {
            r2 = RSquared(estimatedValues, targetValues);
            if (!std::isfinite(r2) || r2 > UpperBound || r2 < LowerBound) { 
                r2 = 0; 
            }
        }
        return UpperBound - r2 + LowerBound;
    }
};
}
#endif

