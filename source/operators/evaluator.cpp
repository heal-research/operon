// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/operators/evaluator.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/error_metrics/normalized_mean_squared_error.hpp"
#include "operon/error_metrics/r2_score.hpp"
#include "operon/error_metrics/correlation_coefficient.hpp"
#include "operon/error_metrics/mean_absolute_error.hpp"

namespace Operon {
    auto MSE::operator()(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target) const noexcept -> double
    {
        return MeanSquaredError(estimated.begin(), estimated.end(), target.begin());
    }
    
    auto MSE::operator()(Iterator beg1, Iterator end1, Iterator beg2) const noexcept -> double
    {
        return MeanSquaredError(beg1, end1, beg2);
    }

    auto RMSE::operator()(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target) const noexcept -> double
    {
        return RootMeanSquaredError(estimated.begin(), estimated.end(), target.begin());
    }
    
    auto RMSE::operator()(Iterator beg1, Iterator end1, Iterator beg2) const noexcept -> double
    {
        return RootMeanSquaredError(beg1, end1, beg2);
    }

    auto NMSE::operator()(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target) const noexcept -> double
    {
        return NormalizedMeanSquaredError(estimated.begin(), estimated.end(), target.begin());
    }
    
    auto NMSE::operator()(Iterator beg1, Iterator end1, Iterator beg2) const noexcept -> double
    {
        return NormalizedMeanSquaredError(beg1, end1, beg2);
    }

    auto MAE::operator()(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target) const noexcept -> double
    {
        return MeanAbsoluteError(estimated.begin(), estimated.end(), target.begin());
    }

    auto MAE::operator()(Iterator beg1, Iterator end1, Iterator beg2) const noexcept -> double
    {
        return MeanAbsoluteError(beg1, end1, beg2);
    }

    auto R2::operator()(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target) const noexcept -> double
    {
        return -R2Score(estimated.begin(), estimated.end(), target.begin());
    }
    
    auto R2::operator()(Iterator beg1, Iterator end1, Iterator beg2) const noexcept -> double
    {
        return -R2Score(beg1, end1, beg2);
    }

    auto C2::operator()(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target) const noexcept -> double
    {
        auto r = CorrelationCoefficient(estimated.begin(), estimated.end(), target.begin());
        return r * r;
    }
    
    auto C2::operator()(Iterator beg1, Iterator end1, Iterator beg2) const noexcept -> double
    {
        auto r = CorrelationCoefficient(beg1, end1, beg2);
        return r * r;
    }

    auto
    Evaluator::operator()(Operon::RandomGenerator& /*random*/, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType
    {
        IncrementFitnessEvaluations();
        auto const& problem = GetProblem();
        auto const& dataset = problem.GetDataset();
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

            if (scaling_) {
                auto stats = bivariate::accumulate<double>(buf.data(), targetValues.data(), buf.size());
                auto a = static_cast<Operon::Scalar>(stats.covariance / stats.variance_x); // scale
                if (!std::isfinite(a)) {
                    a = 1;
                }
                auto b = static_cast<Operon::Scalar>(stats.mean_y - a * stats.mean_x); // offset
                std::transform(buf.begin(), buf.end(), buf.begin(), [&](auto x) { return a * x + b; });
            }
            return error_(buf.begin(), buf.end(), targetValues.begin());
        };

        auto const iter = LocalOptimizationIterations();

        if (iter > 0) {
#if defined(HAVE_CERES)
            NonlinearLeastSquaresOptimizer<OptimizerType::CERES> opt(interpreter_.get(), genotype, dataset);
#else
            NonlinearLeastSquaresOptimizer<OptimizerType::EIGEN> opt(interpreter_.get(), genotype, dataset);
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

} // namespace Operon
