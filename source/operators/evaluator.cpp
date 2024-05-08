// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/core/distance.hpp"
#include "operon/formatter/formatter.hpp"
#include "operon/interpreter/dispatch_table.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/optimizer/solvers/sgd.hpp"

#include <operon/operon_export.hpp>
#include <taskflow/taskflow.hpp>
#include <chrono>
#include <type_traits>

namespace Operon {
    template<typename T>
    auto FitLeastSquaresImpl(Operon::Span<T const> estimated, Operon::Span<T const> target) -> std::pair<double, double>
    requires std::is_arithmetic_v<T>
    {
        auto stats = vstat::bivariate::accumulate<T>(std::cbegin(estimated), std::cend(estimated), std::cbegin(target));
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

    template<> auto OPERON_EXPORT
    Evaluator<DefaultDispatch>::operator()(Operon::RandomGenerator& /*rng*/, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType
    {
        ++CallCount;
        auto const& problem = GetProblem();
        auto const& dataset = problem.GetDataset();

        auto trainingRange = problem.TrainingRange();
        auto targetValues = dataset.GetValues(problem.TargetVariable()).subspan(trainingRange.Start(), trainingRange.Size());

        auto& tree = ind.Genotype;
        auto const& dtable = GetDispatchTable();
        TInterpreter const interpreter{dtable, dataset, tree};

        auto computeFitness = [&]() {
            ++ResidualEvaluations;
            Operon::Vector<Operon::Scalar> estimatedValues;
            if (buf.size() != trainingRange.Size()) {
                estimatedValues.resize(trainingRange.Size());
                buf = { estimatedValues.data(), estimatedValues.size() };
            }
            auto coeff = tree.GetCoefficients();
            interpreter.Evaluate(coeff, trainingRange, buf);
            if (scaling_) {
                auto [a, b] = FitLeastSquaresImpl<Operon::Scalar>(buf, targetValues);
                std::transform(buf.begin(), buf.end(), buf.begin(), [a=a,b=b](auto x) { return a * x + b; });
            }
            ENSURE(buf.size() >= targetValues.size());
            return error_(buf, targetValues);
        };

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
        for (auto i = 0UL; i < sampleSize_; ++i) {
            auto const& rhs = Operon::Random::Sample(random, values.begin(), values.end())->second;
            distance += static_cast<Operon::Scalar>(Operon::Distance::Jaccard(lhs, rhs));
        }
        return EvaluatorBase::ReturnType { -distance / static_cast<Operon::Scalar>(sampleSize_) };
    }

    auto
    AggregateEvaluator::operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType
    {
        using vstat::univariate::accumulate;
        auto f = evaluator_.get()(rng, ind, buf);
        switch(aggtype_) {
            case AggregateType::Min: {
                return { *std::min_element(f.begin(), f.end()) };
            }
            case AggregateType::Max: {
                return { *std::max_element(f.begin(), f.end()) };
            }
            case AggregateType::Median: {
                auto const sz { std::ssize(f) };
                auto const a = f.begin() + sz / 2;
                std::nth_element(f.begin(), a, f.end());
                if (sz % 2 == 0) {
                    auto const b = std::max_element(f.begin(), a);
                    return { (*a + *b) / 2 };
                }
                return { *a };
            }
            case AggregateType::Mean: {
                return { static_cast<Operon::Scalar>(accumulate<Operon::Scalar>(f.begin(), f.end()).mean) };
            }
            case AggregateType::HarmonicMean: {
                auto stats = accumulate<Operon::Scalar>(f.begin(), f.end(), [](auto x) { return 1/x; });
                return { static_cast<Operon::Scalar>(stats.count / stats.sum) };
            }
            case AggregateType::Sum: {
                return { static_cast<Operon::Scalar>(vstat::univariate::accumulate<Operon::Scalar>(f.begin(), f.end()).sum) };
            }
            default: {
                throw std::runtime_error("Unknown AggregateType");
            }
        }
    }

    template<> auto OPERON_EXPORT
    BayesianInformationCriterionEvaluator<DefaultDispatch>::operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType {
        auto const& tree = ind.Genotype;
        auto p = static_cast<Operon::Scalar>(std::ranges::count_if(tree.Nodes(), &Operon::Node::Optimize));
        auto n = static_cast<Operon::Scalar>(Evaluator::GetProblem().TrainingRange().Size());
        auto mse = Evaluator::operator()(rng, ind, buf).front();
        auto bic = n * std::log(mse) + p * std::log(n);
        if (!std::isfinite(bic)) { bic = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(bic) };
    }

    template<> auto OPERON_EXPORT
    AkaikeInformationCriterionEvaluator<DefaultDispatch>::operator()(Operon::RandomGenerator& rng, Individual& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType {
        auto mse = Evaluator::operator()(rng, ind, buf).front();
        auto n = static_cast<Operon::Scalar>(Evaluator::GetProblem().TrainingRange().Size());
        auto aik = n/2 * (std::log(Operon::Math::Tau) + std::log(mse) + 1);
        if (!std::isfinite(aik)) { aik = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(aik) };
    }
} // namespace Operon
