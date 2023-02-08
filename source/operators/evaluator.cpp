// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/autodiff/autodiff.hpp"
#include "operon/core/distance.hpp"
#include "operon/error_metrics/correlation_coefficient.hpp"
#include "operon/error_metrics/mean_absolute_error.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/error_metrics/normalized_mean_squared_error.hpp"
#include "operon/error_metrics/r2_score.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/optimizer.hpp"

#include <taskflow/taskflow.hpp>
#include <chrono>

namespace Operon {
    auto SSE::operator()(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target) const noexcept -> double
    {
        return vstat::univariate::accumulate<Operon::Scalar>(estimated.data(), target.data(), estimated.size(), [](auto a, auto b) { auto e = a - b; return e * e; }).sum;
    }

    auto SSE::operator()(Iterator beg1, Iterator end1, Iterator beg2) const noexcept -> double
    {
        using T = typename std::iterator_traits<Iterator>::value_type;
        return vstat::univariate::accumulate<T>(beg1, end1, beg2, [](auto a, auto b) { auto e = a - b; return e * e; }).sum;
    }

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

    template<typename T>
    auto FitLeastSquaresImpl(Operon::Span<T const> estimated, Operon::Span<T const> target) -> std::pair<double, double>
    requires std::is_arithmetic_v<T>
    {
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
        ++CallCount;
        auto const& problem = GetProblem();
        auto const& dataset = problem.GetDataset();
        auto& genotype = ind.Genotype;

        auto trainingRange = problem.TrainingRange();
        auto targetValues = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start(), trainingRange.Size());

        auto const& interpreter = GetInterpreter();
        auto computeFitness = [&]() {
            ++ResidualEvaluations;
            Operon::Vector<Operon::Scalar> estimatedValues;
            if (buf.size() < trainingRange.Size()) {
                estimatedValues.resize(trainingRange.Size());
                buf = { estimatedValues.data(), estimatedValues.size() };
            }
            interpreter(genotype, dataset, trainingRange, buf);

            if (scaling_) {
                auto [a, b] = FitLeastSquaresImpl<Operon::Scalar>(buf, targetValues);
                std::transform(buf.begin(), buf.end(), buf.begin(), [a=a,b=b](auto x) { return a * x + b; });
            }
            ENSURE(!buf.empty() && buf.size() == targetValues.size());
            return error_(buf, targetValues);
        };

        auto const iter = LocalOptimizationIterations();

        if (iter > 0) {
            auto t0 = std::chrono::steady_clock::now();
            //Autodiff::Forward::DerivativeCalculator calc(this->GetInterpreter());
            Autodiff::Reverse::DerivativeCalculator calc{ this->GetInterpreter() };
            NonlinearLeastSquaresOptimizer<decltype(calc), OptimizerType::Eigen> opt(calc, ind.Genotype, dataset);
            OptimizerSummary summary{};
            auto coefficients = opt.Optimize(targetValues, trainingRange, iter, summary);
            ResidualEvaluations += summary.FunctionEvaluations;
            JacobianEvaluations += summary.JacobianEvaluations;

            if (summary.Success) {
                genotype.SetCoefficients(coefficients);
            }
            auto t1 = std::chrono::steady_clock::now();
            CostFunctionTime += std::chrono::duration_cast<std::chrono::microseconds>(t1-t0).count();
        }

        auto fit = static_cast<Operon::Scalar>(computeFitness());
        if (!std::isfinite(fit)) {
            fit = EvaluatorBase::ErrMax;
        }
        return typename EvaluatorBase::ReturnType{ fit };
    }

    auto DiversityEvaluator::Prepare(Operon::Span<Operon::Individual const> pop) const -> void {
        divmap_.clear();
        for (auto const& individual : pop) {
            auto const& tree = individual.Genotype;
            auto const& nodes = tree.Nodes();
            (void) tree.Hash(hashmode_);
            std::vector<Operon::Hash> hash(nodes.size());;
            std::transform(nodes.begin(), nodes.end(), hash.begin(), [](auto const& n) { return n.CalculatedHashValue; });
            std::stable_sort(hash.begin(), hash.end());
            divmap_[tree.HashValue()] = std::move(hash);
        }
    }

    auto
    DiversityEvaluator::operator()(Operon::RandomGenerator& random, Individual& ind, Operon::Span<Operon::Scalar>  /*buf*/) const -> typename EvaluatorBase::ReturnType
    {
        (void)ind.Genotype.Hash(hashmode_);
        std::vector<Operon::Hash> lhs(ind.Genotype.Length());
        auto const& nodes = ind.Genotype.Nodes();
        std::transform(nodes.begin(), nodes.end(), lhs.begin(), [](auto const& n) { return n.CalculatedHashValue; });
        std::stable_sort(lhs.begin(), lhs.end());
        auto const& values = divmap_.values();

        Operon::Scalar distance{0};
        std::vector<double> distances(sampleSize_);
        for (auto i = 0; i < sampleSize_; ++i) {
            auto const& rhs = Operon::Random::Sample(random, values.begin(), values.end())->second;
            distance += static_cast<Operon::Scalar>(Operon::Distance::Jaccard(lhs, rhs));
        }
        return EvaluatorBase::ReturnType { -distance / static_cast<Operon::Scalar>(sampleSize_) };
    }

    auto MinimumDescriptionLengthEvaluator::operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType {
        // call the base method of the evaluator in order to optimize the coefficients
        // this also returns the error which we are going to use
        auto error = Evaluator::operator()(rng, ind, buf).front();

        // compute the minimum description length
        Operon::Set<Operon::Hash> uniqueFunctions;
        for (auto const& n : ind.Genotype.Nodes()) {
            if (n.IsLeaf()) { continue; }
            uniqueFunctions.insert(n.HashValue);
        }
        auto const n { uniqueFunctions.size() };
        auto const k { ind.Genotype.Length() };

        auto const& interpreter = Evaluator::GetInterpreter();
        Autodiff::Reverse::DerivativeCalculator calc{ interpreter };
        auto const& problem = Evaluator::GetProblem();
        auto const& dataset = problem.GetDataset();
        auto const range = problem.TrainingRange();
        auto const coeff = ind.Genotype.GetCoefficients();
        auto const p { coeff.size() };
        Eigen::Matrix<Operon::Scalar, -1, -1> j = calc(ind.Genotype, dataset, range, coeff);

        auto values = interpreter(ind.Genotype, dataset, range);
        auto target = dataset.GetValues(problem.TargetVariable()).subspan(range.Start(), range.Size());
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> c(coeff.data(), std::ssize(coeff));
        auto s = (j.transpose() * j).diagonal().array().log().sum() / 2 + c.abs().log().sum();
        auto mld = std::log(error) + static_cast<double>(k) * std::log(n) - static_cast<double>(p) / 2 * std::log(3) + s;
        if (!std::isfinite(mld)) { mld = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(mld) };
    }

    auto BayesianInformationCriterionEvaluator::operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType {
        auto const& tree = ind.Genotype;
        auto ncoef = static_cast<Operon::Scalar>(std::ranges::count_if(tree.Nodes(), &Operon::Node::Optimize));
        auto nrows = static_cast<Operon::Scalar>(Evaluator::GetProblem().TrainingRange().Size());
        auto error = Evaluator::operator()(rng, ind, buf).front();
        auto bic = nrows * std::log(error) + ncoef * std::log(nrows);
        if (!std::isfinite(bic)) { bic = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(bic) };
    }

    auto AkaikeInformationCriterionEvaluator::operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType {
        auto const& tree = ind.Genotype;
        auto ncoef = static_cast<Operon::Scalar>(std::ranges::count_if(tree.Nodes(), &Operon::Node::Optimize));
        auto nrows = static_cast<Operon::Scalar>(Evaluator::GetProblem().TrainingRange().Size());
        auto error = Evaluator::operator()(rng, ind, buf).front();
        auto aik = 2 * ncoef + nrows * std::log(error);
        if (!std::isfinite(aik)) { aik = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(aik) };
    }
} // namespace Operon
