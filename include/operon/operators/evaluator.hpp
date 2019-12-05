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
#include "core/metrics.hpp"
#include "core/operator.hpp"
#include "stat/pearson.hpp"
#include "stat/meanvariance.hpp"

namespace Operon {
template <typename T>
class NormalizedMeanSquaredErrorEvaluator : public EvaluatorBase<T, false> {
public:
    NormalizedMeanSquaredErrorEvaluator(Problem& problem)
        : EvaluatorBase<T, false>(problem)
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
            nmse = operon::scalar::max();
        }
        return nmse;
    }

    void Prepare(const gsl::span<const T> pop)
    {
        this->population = pop;
    }
};

template <typename T>
class RSquaredEvaluator : public EvaluatorBase<T, true> {
public:
    static constexpr bool Maximization = true;

    RSquaredEvaluator(Problem& problem)
        : EvaluatorBase<T, true>(problem)
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
        //if (!std::isfinite(r2)) {
        //    r2 = 0;
        //}
        return r2;
    }

    void Prepare(const gsl::span<const T> pop)
    {
        this->population = pop;
    }
};
}

#endif
