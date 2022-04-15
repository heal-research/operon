// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#include "operon/core/distance.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/error_metrics/normalized_mean_squared_error.hpp"
#include "operon/error_metrics/r2_score.hpp"
#include "operon/error_metrics/correlation_coefficient.hpp"
#include "operon/error_metrics/mean_absolute_error.hpp"
#include "operon/nnls/nnls.hpp"

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
        return -(r * r);
    }

    auto C2::operator()(Iterator beg1, Iterator end1, Iterator beg2) const noexcept -> double
    {
        auto r = CorrelationCoefficient(beg1, end1, beg2);
        return -(r * r);
    }

    template<typename T, std::enable_if_t<std::is_arithmetic_v<T>, bool> = true>
    auto FitLeastSquaresImpl(Operon::Span<T const> estimated, Operon::Span<T const> target) -> std::pair<double, double> {
        auto stats = vstat::bivariate::accumulate<T>(estimated.data(), target.data(), estimated.size());
        auto a = stats.covariance / stats.variance_x; // scale
        if (!std::isfinite(a)) {
            a = 1;
        }
        auto b = stats.mean_y - a * stats.mean_x; // offset
        return {a, b};
    }

    auto FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target) noexcept -> std::pair<double, double> {
        return FitLeastSquaresImpl<float>(estimated, target);
    }

    auto FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target) noexcept -> std::pair<double, double> {
        return FitLeastSquaresImpl<double>(estimated, target);
    }

    auto
    Evaluator::operator()(Operon::RandomGenerator& /*random*/, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType
    {
        IncrementEvaluationCounter();
        auto const& problem = GetProblem();
        auto const& dataset = problem.GetDataset();
        auto& genotype = ind.Genotype;

        auto trainingRange = problem.TrainingRange();
        auto targetValues = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start(), trainingRange.Size());

        auto computeFitness = [&]() {
            IncrementResidualEvaluations();
            Operon::Vector<Operon::Scalar> estimatedValues;
            if (buf.size() < trainingRange.Size()) {
                estimatedValues.resize(trainingRange.Size());
                buf = Operon::Span<Operon::Scalar>(estimatedValues.data(), estimatedValues.size());
            }
            GetInterpreter().template Evaluate<Operon::Scalar>(genotype, dataset, trainingRange, buf);

            if (scaling_) {
                auto [a, b] = FitLeastSquaresImpl<Operon::Scalar>(buf, targetValues);
                std::transform(buf.begin(), buf.end(), buf.begin(), [a=a,b=b](auto x) { return a * x + b; });
            }
            return error_(buf, targetValues);
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
            IncrementResidualEvaluations(summary.FunctionEvaluations);
            IncrementJacobianEvaluations(summary.JacobianEvaluations);

            if (summary.Success) {
                genotype.SetCoefficients(coeff);
            }
        }

        auto fit = Operon::Vector<Operon::Scalar> { static_cast<Operon::Scalar>(computeFitness()) };
        for (auto& v : fit) {
            if (!std::isfinite(v)) {
                v = std::numeric_limits<Operon::Scalar>::max();
            }
        }
        return fit;
    }

    auto DiversityEvaluator::Prepare(Operon::Span<Operon::Individual const> pop) const -> void {
        if (hashes_.size() < pop.size()) {
            hashes_.resize(pop.size());
        }

        // hash the trees
        for (size_t i = 0; i < pop.size(); ++i) {
            auto const& nodes = pop[i].Genotype.Hash(Operon::HashMode::Strict).Nodes();
            hashes_[i].reserve(nodes.size());
            std::transform(std::begin(nodes), std::end(nodes), std::back_inserter(hashes_[i]), [](auto const& n) { return n.CalculatedHashValue; });
            std::stable_sort(std::begin(hashes_[i]), std::end(hashes_[i]));
        }

        std::vector<vstat::univariate_accumulator<Operon::Scalar>> stats;
        stats.reserve(pop.size());
        for (auto i = 0UL; i < pop.size(); ++i) {
            stats.emplace_back(Operon::Scalar{0});
        }

        auto const n = pop.size() * (pop.size() - 1) / 2;

        for (auto i = 0UL; i < pop.size() - 1; ++i) {
            for (auto j = i+1; j < pop.size(); ++j) {
                auto d = Operon::Distance::Jaccard(hashes_[i], hashes_[j]);
                stats[i](static_cast<Operon::Scalar>(d));
                stats[j](static_cast<Operon::Scalar>(d));
            }
        }

        divmap_.clear();
        for (auto i = 0UL; i < pop.size(); ++i) {
            [[maybe_unused]] auto [it, ok] = divmap_.insert({
                pop[i].Genotype.HashValue(),
                static_cast<Operon::Scalar>(vstat::univariate_statistics(stats[i]).mean)
            });
        }
    }

    auto
    DiversityEvaluator::operator()(Operon::RandomGenerator& /*random*/, Individual& ind, Operon::Span<Operon::Scalar>  /*buf*/) const -> typename EvaluatorBase::ReturnType
    {
        Operon::Scalar f{std::numeric_limits<Operon::Scalar>::max()};
        if (auto it = divmap_.find(ind.Genotype.HashValue()); it != divmap_.end()) {
             f = it->second;
        }
        return EvaluatorBase::ReturnType { f };
    }

} // namespace Operon
