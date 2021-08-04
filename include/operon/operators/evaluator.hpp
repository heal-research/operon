// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef EVALUATOR_HPP
#define EVALUATOR_HPP

#include "collections/projection.hpp"
#include "core/metrics.hpp"
#include "core/operator.hpp"
#include "core/types.hpp"
#include "core/format.hpp"
#include "nnls/nnls.hpp"
#include "nnls/tiny_optimizer.hpp"

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
    operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar>) const override
    {
        ++this->fitnessEvaluations;
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
    operator()(Operon::RandomGenerator&, Individual& ind, Operon::Span<Operon::Scalar> buf = Operon::Span<Operon::Scalar>{}) const override
    {
        ++this->fitnessEvaluations;
        auto& problem_ = this->problem.get();
        auto& dataset = problem_.GetDataset();
        auto& genotype = ind.Genotype;

        auto trainingRange = problem_.TrainingRange();
        auto targetValues = dataset.GetValues(problem_.TargetVariable()).subspan(trainingRange.Start(), trainingRange.Size());

        auto computeFitness = [&]() {
            Operon::Vector<Operon::Scalar> estimatedValues;
            if (buf.size() < trainingRange.Size()) {
                estimatedValues.resize(trainingRange.Size());
                buf = Operon::Span<Operon::Scalar>(estimatedValues.data(), estimatedValues.size());
            }
            GetInterpreter().template Evaluate<Operon::Scalar>(genotype, dataset, trainingRange, buf);

            if constexpr (LinearScaling) {
                auto stats = bivariate::accumulate<double>(buf.data(), targetValues.data(), buf.size());
                auto a = static_cast<Operon::Scalar>(stats.covariance / stats.variance_x); // scale
                if (!std::isfinite(a)) { a = 1; }
                auto b = static_cast<Operon::Scalar>(stats.mean_y - a * stats.mean_x);     // offset

                Projection p(buf, [&](auto x) { return a * x + b; });
                return ErrorMetric{}(p.begin(), p.end(), targetValues.begin());
            }
            auto err = ErrorMetric{}(buf.begin(), buf.end(), targetValues.begin());
            return err;
        };

        if (this->iterations > 0) {
#if defined(CERES_TINY_SOLVER) || !defined(HAVE_CERES)
            NonlinearLeastSquaresOptimizer<OptimizerType::TINY> opt(interpreter.get(), genotype, dataset);
#else
            NonlinearLeastSquaresOptimizer<OptimizerType::CERES> opt(interpreter.get(), genotype, dataset);
#endif
            auto coeff = genotype.GetCoefficients();
            auto summary = opt.Optimize(targetValues, trainingRange, this->iterations);
            this->localEvaluations += summary.Iterations;

            if (summary.InitialCost < summary.FinalCost) {
                // if optimization failed, restore the original coefficients
                genotype.SetCoefficients(coeff);
            }
        }

        auto fit = Operon::Vector<Operon::Scalar>{ computeFitness() };
        for (auto& v : fit) {
            if (!std::isfinite(v)) {
                v = Operon::Numeric::Max<Operon::Scalar>();
            }
        }
        return fit;
    }

private:
    std::reference_wrapper<Interpreter> interpreter;
};

class MultiEvaluator : public EvaluatorBase {
    public:
    MultiEvaluator(Problem& problem)
        : EvaluatorBase(problem)
    {
    }

    void Add(EvaluatorBase const& evaluator) {
        evaluators_.push_back(std::ref(evaluator));
    }

    typename EvaluatorBase::ReturnType
    operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf = Operon::Span<Operon::Scalar>{}) const override
    {
        EXPECT(evaluators_.size() > 1);
        Operon::Vector<Operon::Scalar> fit;
        for (size_t i = 0; i < evaluators_.size(); ++i) {
            auto fit_i = (evaluators_[i])(rng, ind, buf);
            std::copy(fit_i.begin(), fit_i.end(), std::back_inserter(fit));
        }
        // proxy the budget of the first evaluator (the one that actually calls the tree interpreter)
        this->fitnessEvaluations = evaluators_[0].get().FitnessEvaluations();
        this->localEvaluations = evaluators_[0].get().LocalEvaluations();
        return fit;
    }

    private:
    std::vector<std::reference_wrapper<EvaluatorBase const>> evaluators_;
};

using MeanSquaredErrorEvaluator           = Evaluator<MSE,  true>;
using NormalizedMeanSquaredErrorEvaluator = Evaluator<NMSE, true>;
using RootMeanSquaredErrorEvaluator       = Evaluator<RMSE, true>;
using MeanAbsoluteErrorEvaluator          = Evaluator<MAE,  true>;
using RSquaredEvaluator                   = Evaluator<R2,   false>;
using L2NormEvaluator                     = Evaluator<L2,   true>;

}
#endif

