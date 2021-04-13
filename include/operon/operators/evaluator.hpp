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
    Evaluator(Problem& problem, Interpreter& interp)
        : EvaluatorBase(problem), interpreter(interp)
    {
    }

    Interpreter& GetInterpreter() { return interpreter; }
    Interpreter const& GetInterpreter() const { return interpreter; }

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
#if defined(CERES_TINY_SOLVER) || !defined(HAVE_CERES)
        NonlinearLeastSquaresOptimizer<OptimizerType::TINY> opt(interpreter.get(), genotype, dataset);
#else
        NonlinearLeastSquaresOptimizer<OptimizerType::CERES> opt(interpreter.get(), genotype, dataset);
#endif
            auto summary = opt.Optimize(targetValues, trainingRange, this->iterations);
            this->localEvaluations += summary.Iterations;
        }

        auto estimatedValues = GetInterpreter().template Evaluate<Operon::Scalar>(genotype, dataset, trainingRange);

        //Operon::Scalar scale = 1.0;
        //Operon::Scalar offset = 0.0;
        if constexpr (LinearScaling) {
            auto [m, c] = Operon::LinearScalingCalculator::Calculate(gsl::span<Operon::Scalar const>{ estimatedValues }, targetValues);
            Eigen::Map<Eigen::Array<Operon::Scalar, Eigen::Dynamic, 1>> x(estimatedValues.data(), estimatedValues.size());
            x = x * m + c;
            //auto stats = bivariate::accumulate<Operon::Scalar>(estimatedValues.data(), targetValues.data(), estimatedValues.size());
            //scale = stats.covariance / stats.sample_variance_x;
            //offset = stats.mean_y - scale * stats.mean_x;
        }
        auto fit = ErrorMetric{}(gsl::span<Operon::Scalar const>{ estimatedValues }, targetValues);

        if (!std::isfinite(fit)) {
            return Operon::Numeric::Max<Operon::Scalar>();
        }
        return fit;
    }


private:
    std::reference_wrapper<Interpreter> interpreter;
};

using MeanSquaredErrorEvaluator           = Evaluator<MSE,  true>;
using NormalizedMeanSquaredErrorEvaluator = Evaluator<NMSE, true>;
using RootMeanSquaredErrorEvaluator       = Evaluator<RMSE, true>;
using MeanAbsoluteErrorEvaluator          = Evaluator<MAE,  true>;
using RSquaredEvaluator                   = Evaluator<R2,   false>;

}
#endif

