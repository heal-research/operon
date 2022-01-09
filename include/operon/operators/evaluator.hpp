// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_EVALUATOR_HPP
#define OPERON_EVALUATOR_HPP

#include <atomic>
#include <utility>

#include "operon/collections/projection.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/metrics.hpp"
#include "operon/core/operator.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/types.hpp"
#include "operon/nnls/nnls.hpp"

namespace Operon {

class EvaluatorBase : public OperatorBase<Operon::Vector<Operon::Scalar>, Individual&, Operon::Span<Operon::Scalar>> {
    Operon::Span<const Individual> population_;
    std::reference_wrapper<const class Problem> problem_;
    mutable std::atomic_ulong fitnessEvaluations_ = 0;
    mutable std::atomic_ulong localEvaluations_ = 0;
    size_t iterations_ = DefaultLocalOptimizationIterations;
    size_t budget_ = DefaultEvaluationBudget;

public:
    static constexpr size_t DefaultLocalOptimizationIterations = 50;
    static constexpr size_t DefaultEvaluationBudget = 100'000;

    using ReturnType = OperatorBase::ReturnType;

    explicit EvaluatorBase(Problem& problem)
        : problem_(problem)
    {
    }

    virtual void Prepare(const Operon::Span<const Individual> pop)
    {
        population_ = pop;
    }

    auto TotalEvaluations() const -> size_t { return fitnessEvaluations_ + localEvaluations_; }
    auto FitnessEvaluations() const -> size_t { return fitnessEvaluations_; }
    auto LocalEvaluations() const -> size_t { return localEvaluations_; }

    void SetFitnessEvaluations(size_t value) const { fitnessEvaluations_ = value; }
    void SetLocalEvaluations(size_t value) const { localEvaluations_ = value; }

    void IncrementFitnessEvaluations() const { ++fitnessEvaluations_; }
    void IncrementLocalEvaluations() const { ++localEvaluations_; }

    void IncrementFitnessEvaluations(size_t inc) const { fitnessEvaluations_ += inc; }
    void IncrementLocalEvaluations(size_t inc) const { localEvaluations_ += inc; }

    void SetLocalOptimizationIterations(size_t value) { iterations_ = value; }
    auto LocalOptimizationIterations() const -> size_t { return iterations_; }

    void SetBudget(size_t value) { budget_ = value; }
    auto Budget() const -> size_t { return budget_; }
    auto BudgetExhausted() const -> bool { return TotalEvaluations() > Budget(); }

    auto Population() const -> Operon::Span<Individual const> { return population_; }
    auto GetProblem() const -> Problem const& { return problem_; }

    void Reset()
    {
        fitnessEvaluations_ = 0;
        localEvaluations_ = 0;
    }
};

class UserDefinedEvaluator : public EvaluatorBase {
public:
    UserDefinedEvaluator(Problem& problem, std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual&)> func)
        : EvaluatorBase(problem)
        , fref_(std::move(func))
    {
    }

    // the func signature taking a pointer to the rng is a workaround for pybind11, since the random generator is non-copyable we have to pass a pointer
    UserDefinedEvaluator(Problem& problem, std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual&)> func)
        : EvaluatorBase(problem)
        , fptr_(std::move(func))
    {
    }

    auto
    operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> /*args*/) const -> typename EvaluatorBase::ReturnType override
    {
        //++this->FitnessEvaluations();
        IncrementFitnessEvaluations();
        return fptr_ ? fptr_(&rng, ind) : fref_(rng, ind);
    }

private:
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual&)> fref_;
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual&)> fptr_; // workaround for pybind11
};

template <typename ErrorMetric, bool LinearScaling = false>
class Evaluator : public EvaluatorBase {

public:
    Evaluator(Problem& problem, Interpreter& interp)
        : EvaluatorBase(problem)
        , interpreter_(interp)
    {
    }

    auto GetInterpreter() -> Interpreter& { return interpreter_; }
    auto GetInterpreter() const -> Interpreter const& { return interpreter_; }

    auto
    operator()(Operon::RandomGenerator& /*random*/, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override
    {
        IncrementFitnessEvaluations();
        auto& problem = GetProblem();
        auto& dataset = problem.GetDataset();
        auto& genotype = ind.Genotype;

        auto trainingRange = problem.TrainingRange();
        auto targetValues = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start(), trainingRange.Size());

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
                if (!std::isfinite(a)) {
                    a = 1;
                }
                auto b = static_cast<Operon::Scalar>(stats.mean_y - a * stats.mean_x); // offset

                Projection p(buf, [&](auto x) { return a * x + b; });
                return ErrorMetric {}(p.begin(), p.end(), targetValues.begin());
            }
            auto err = ErrorMetric {}(buf.begin(), buf.end(), targetValues.begin());
            return err;
        };

        auto const iter = LocalOptimizationIterations();

        if (iter > 0) {
#if defined(CERES_TINY_SOLVER) || !defined(HAVE_CERES)
            NonlinearLeastSquaresOptimizer<OptimizerType::TINY> opt(interpreter_.get(), genotype, dataset);
#else
            NonlinearLeastSquaresOptimizer<OptimizerType::CERES> opt(interpreter_.get(), genotype, dataset);
#endif
            auto coeff = genotype.GetCoefficients();
            auto summary = opt.Optimize(targetValues, trainingRange, iter);
            //this->localEvaluations += summary.Iterations;
            IncrementLocalEvaluations(summary.Iterations);

            if (summary.InitialCost < summary.FinalCost) {
                // if optimization failed, restore the original coefficients
                genotype.SetCoefficients(coeff);
            }
        }

        auto fit = Operon::Vector<Operon::Scalar> { static_cast<Operon::Scalar>(computeFitness()) };
        for (auto& v : fit) {
            if (!std::isfinite(v)) {
                v = Operon::Numeric::Max<Operon::Scalar>();
            }
        }
        return fit;
    }

private:
    std::reference_wrapper<Interpreter> interpreter_;
};

class MultiEvaluator : public EvaluatorBase {
public:
    explicit MultiEvaluator(Problem& problem)
        : EvaluatorBase(problem)
    {
    }

    void Add(EvaluatorBase const& evaluator)
    {
        evaluators_.push_back(std::ref(evaluator));
    }

    auto
    operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override
    {
        EXPECT(evaluators_.size() > 1);
        Operon::Vector<Operon::Scalar> fit;
        for (auto const& evaluator : evaluators_) {
            auto fitI = evaluator(rng, ind, buf);
            std::copy(fitI.begin(), fitI.end(), std::back_inserter(fit));
        }
        // proxy the budget of the first evaluator (the one that actually calls the tree interpreter)
        SetFitnessEvaluations(evaluators_[0].get().FitnessEvaluations());
        SetLocalEvaluations(evaluators_[0].get().LocalEvaluations());
        return fit;
    }

private:
    std::vector<std::reference_wrapper<EvaluatorBase const>> evaluators_;
};

using MeanSquaredErrorEvaluator = Evaluator<MSE, true>;
using NormalizedMeanSquaredErrorEvaluator = Evaluator<NMSE, true>;
using RootMeanSquaredErrorEvaluator = Evaluator<RMSE, true>;
using MeanAbsoluteErrorEvaluator = Evaluator<MAE, true>;
using SquaredCorrelationEvaluator = Evaluator<C2, false>;
using R2Evaluator = Evaluator<R2, true>;
using L2NormEvaluator = Evaluator<L2, true>;

// a couple of useful user-defined evaluators (mostly to avoid calling lambdas from python)
// TODO: think about a better design
class LengthEvaluator : public UserDefinedEvaluator {
public:
    explicit LengthEvaluator(Operon::Problem& problem)
        : UserDefinedEvaluator(problem, [](Operon::RandomGenerator& /*unused*/, Operon::Individual& ind) {
            return EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(ind.Genotype.Length()) };
        })
    {
    }
};

class ShapeEvaluator : public UserDefinedEvaluator {
public:
    explicit ShapeEvaluator(Operon::Problem& problem)
        : UserDefinedEvaluator(problem, [](Operon::RandomGenerator& /*unused*/, Operon::Individual& ind) {
            return EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(ind.Genotype.VisitationLength()) };
        })
    {
    }
};
} // namespace Operon
#endif
