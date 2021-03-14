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

#include "core/metrics.hpp"
#include "core/operator.hpp"
#include "core/types.hpp"
#include "nnls/nnls.hpp"
#include "stat/meanvariance.hpp"
#include "stat/pearson.hpp"

namespace Operon {

class UserDefinedEvaluator : public EvaluatorBase {
public:
    UserDefinedEvaluator(Problem& problem, std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual&)> && func)
        : EvaluatorBase(problem), fref(std::move(func))
    {
    }

    UserDefinedEvaluator(Problem& problem, std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual&)> const& func)
        : EvaluatorBase(problem), fref(func)
    {
    }

    // the func signature taking a pointer to the rng is a workaround for pybind11, since the random generator is non-copyable we have to pass a pointer
    UserDefinedEvaluator(Problem& problem, std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual&)> && func)
        : EvaluatorBase(problem), fptr(std::move(func))
    {
    }

    UserDefinedEvaluator(Problem& problem, std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual&)> const& func)
        : EvaluatorBase(problem), fptr(func)
    {
    }

    typename EvaluatorBase::ReturnType
    operator()(Operon::RandomGenerator& rng, Individual& ind) const override
    {
        return fptr ? fptr(&rng, ind) : fref(rng, ind);
    }

private:
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual&)> fref;
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual&)> fptr; // workaround for pybind11
};

template<typename ErrorMetric, bool LinearScaling = false>
class Evaluator : public EvaluatorBase {

public:
    Evaluator(Problem& problem)
        : EvaluatorBase(problem)
    {
    }

    typename EvaluatorBase::ReturnType
    operator()(Operon::RandomGenerator&, Individual& ind) const override
    {
        ++this->fitnessEvaluations;
        auto& problem_ = this->problem.get();
        auto& dataset = problem_.GetDataset();
        auto& genotype = ind.Genotype;

        auto trainingRange = problem_.TrainingRange();
        auto targetValues = dataset.GetValues(problem_.TargetVariable()).subspan(trainingRange.Start(), trainingRange.Size());

        if (this->iterations > 0) {
            auto summary = Optimize(genotype, dataset, targetValues, trainingRange, this->iterations);
            this->localEvaluations += summary.Iterations;
        }

        auto estimatedValues = Evaluate<Operon::Scalar>(genotype, dataset, trainingRange);

        if constexpr (LinearScaling) {
            // scale values
            Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 2, Eigen::ColMajor> a(trainingRange.Size(), 2);
            Eigen::Map<Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1>> x(estimatedValues.data(), estimatedValues.size());
            a.col(0) = x;
            a.col(1).setConstant(1);

            Eigen::Map<const Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>> y(targetValues.data(), targetValues.size());
            Eigen::ColPivHouseholderQR<decltype(a)> hh(a);
            auto v = hh.solve(y);
            x = v(0) + x * v(1);
        }
        auto fit = metric(gsl::span<Operon::Scalar const>{ estimatedValues }, targetValues);

        if (!std::isfinite(fit)) {
            return Operon::Numeric::Max<Operon::Scalar>();
        }
        return fit;
    }


private:
    ErrorMetric metric;
};

using MeanSquaredErrorEvaluator           = Evaluator<MSE, true>;
using NormalizedMeanSquaredErrorEvaluator = Evaluator<NMSE, true>;
using RootMeanSquaredErrorEvaluator       = Evaluator<RMSE, true>;
using MeanAbsoluteErrorEvaluator          = Evaluator<MAE, true>;
using RSquaredEvaluator                   = Evaluator<R2, false>;

}
#endif

