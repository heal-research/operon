// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/core/distance.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/error_metrics/mean_squared_error.hpp"
#include "operon/error_metrics/normalized_mean_squared_error.hpp"
#include "operon/error_metrics/r2_score.hpp"
#include "operon/error_metrics/correlation_coefficient.hpp"
#include "operon/error_metrics/mean_absolute_error.hpp"
#include "operon/optimizer/optimizer.hpp"

#include <taskflow/taskflow.hpp>

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
        ++CallCount;
        auto const& problem = GetProblem();
        auto const& dataset = problem.GetDataset();
        auto& genotype = ind.Genotype;

        auto trainingRange = problem.TrainingRange();
        auto targetValues = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start(), trainingRange.Size());

        auto computeFitness = [&]() {
            ++ResidualEvaluations;
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
            constexpr OptimizerType TYPE = OptimizerType::CERES;
#else
            constexpr OptimizerType TYPE = OptimizerType::EIGEN;
#endif
            NonlinearLeastSquaresOptimizer<TYPE> opt(interpreter_.get(), genotype, dataset);
            OptimizerSummary summary{};
            auto coefficients = opt.Optimize(targetValues, trainingRange, iter, summary);
            ResidualEvaluations += summary.FunctionEvaluations;
            JacobianEvaluations += summary.JacobianEvaluations;

            if (summary.Success) {
                genotype.SetCoefficients(coefficients);
            }
        }

        auto fit = EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(computeFitness()) };
        for (auto& v : fit) {
            if (!std::isfinite(v)) {
                v = std::numeric_limits<Operon::Scalar>::max();
            }
        }
        return fit;
    }

    auto DiversityEvaluator::Prepare(Operon::Span<Operon::Individual const> pop) const -> void {
        divmap_.clear();
        for (auto i = 0UL; i < pop.size(); ++i) {
            auto const& tree = pop[i].Genotype;
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

    auto
    ComplexityEvaluator::operator()(Operon::RandomGenerator& random, Individual& ind, Operon::Span<Operon::Scalar>  /*buf*/) const -> typename EvaluatorBase::ReturnType
    {
        // NOLINTBEGIN(*)
        static Operon::Map<NodeType, int> costMap {
            { NodeType::Add,  3 },
            { NodeType::Mul,  3 },
            { NodeType::Sub,  3 },
            { NodeType::Div,  3 },
            { NodeType::Fmin, 3 },
            { NodeType::Fmax, 3 },
            { NodeType::Aq,   3 },
            { NodeType::Pow,  9 },
            { NodeType::Abs,  3 },
            { NodeType::Acos, 9 },
            { NodeType::Asin, 9 },
            { NodeType::Atan, 28 },
            { NodeType::Cbrt, 22 },
            { NodeType::Ceil,  3 },
            { NodeType::Cos,   5 },
            { NodeType::Cosh, 37 },
            { NodeType::Exp,   5 },
            { NodeType::Floor, 3 },
            { NodeType::Log, 6 },
            { NodeType::Logabs, 6 },
            { NodeType::Log1p, 6 },
            { NodeType::Sin, 5 },
            { NodeType::Sinh, 37 },
            { NodeType::Sqrt, 4 },
            { NodeType::Sqrtabs, 4 },
            { NodeType::Tan, 65 },
            { NodeType::Tanh, 27 },
            { NodeType::Square, 3 },
            { NodeType::Constant, 0 },
            { NodeType::Variable, 0 },
            { NodeType::Dynamic, 1000 }
        };
        // NOLINTEND(*)
        auto const& nodes = ind.Genotype.Nodes();
        auto complexity = std::transform_reduce(nodes.begin(), nodes.end(), 0, std::plus{}, [&](auto const& node) { return costMap[node.Type]; });
        return EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(complexity) };
    }
} // namespace Operon
