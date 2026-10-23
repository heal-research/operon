// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/evaluator.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/distance.hpp"
#include "operon/operators/linear_scaling.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/operators/shape_constrained_evaluator.hpp"
#include "operon/optimizer/optimizer.hpp"
#include "operon/random/random.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <operon/operon_export.hpp>
#include <random>
#include <stdexcept>
#include <type_traits>
#include <vstat/vstat.hpp>

namespace Operon {
namespace {
    template <typename T>
    auto FitLeastSquaresForward(Operon::Span<T const> estimated, Operon::Span<T const> target,
        Operon::Span<T const> weights = {}) -> std::pair<double, double>
    {
        if constexpr (std::is_same_v<T, Operon::Scalar>) {
            auto const scaling = FitLinearScaling(estimated, target, weights);
            return { scaling.Scale, scaling.Offset };
        } else {
            auto stats = weights.empty()
                ? vstat::bivariate::accumulate<T>(estimated.data(), estimated.data() + estimated.size(), target.data())
                : vstat::bivariate::accumulate<T>(
                      estimated.data(), estimated.data() + estimated.size(), target.data(), weights.data());
            auto a = stats.covariance / stats.variance_x; // scale
            if (!std::isfinite(a)) {
                a = 1;
            }
            auto b = stats.mean_y - (a * stats.mean_x); // offset
            return { a, b };
        }
    }

    // Outlined so the hot default path (skipNonFinite_ == false) keeps `Evaluate` inlinable into
    // its caller. `[[gnu::noinline]]` is GCC/Clang-specific and silently ignored on compilers that
    // don't recognize it (e.g. MSVC); it only costs the opt-in caller here regardless.
    template <typename T>
    [[gnu::noinline]] auto SkipNonFiniteScore(ErrorMetric const& error, Operon::Span<T> estimated,
        Operon::Span<T const> target, Operon::Span<T const> weights, std::optional<LinearScaling> const& scaling,
        double penaltyWeight) -> Operon::Scalar
    {
        if (scaling) {
            scaling->ApplyInPlace(estimated);
        }
        auto [value, nonFiniteCount]
            = weights.empty() ? error.FiniteSubset(estimated, target) : error.FiniteSubset(estimated, target, weights);
        if (nonFiniteCount == estimated.size()) {
            return EvaluatorBase::ErrMax;
        }
        auto const fraction
            = nonFiniteCount != 0 ? static_cast<double>(nonFiniteCount) / static_cast<double>(estimated.size()) : 0.0;
        // Per-metric scale keeps one penaltyWeight meaningful across metric units: NMSE, R2, and C2 are
        // dimensionless (scale = 1); MSE/SSE are squared-error units (variance, and variance * finite-count
        // since SSE is a sum); RMSE/MAE are linear-error units (stddev = sqrt(variance)).
        double sumWeights = 0.0;
        double mean = 0.0;
        double m2 = 0.0;
        for (auto i = std::size_t { 0 }; i < target.size(); ++i) {
            auto const y = static_cast<double>(target[i]);
            auto const w = weights.empty() ? 1.0 : static_cast<double>(weights[i]);
            if (!std::isfinite(y) || !std::isfinite(w) || w == 0.0) {
                continue;
            }
            auto const nextSumWeights = sumWeights + w;
            auto const delta = y - mean;
            auto const r = delta * w / nextSumWeights;
            mean += r;
            m2 += sumWeights * delta * r;
            sumWeights = nextSumWeights;
        }
        auto const variance = sumWeights > 0.0 ? m2 / sumWeights : 0.0;
        double scale {};
        switch (error.Type()) {
        case ErrorType::NMSE:
        case ErrorType::R2:
        case ErrorType::C2:
            scale = 1.0;
            break;
        case ErrorType::MSE:
            scale = variance;
            break;
        case ErrorType::RMSE:
        case ErrorType::MAE:
            scale = std::sqrt(variance);
            break;
        case ErrorType::SSE:
            scale = variance * static_cast<double>(estimated.size() - nonFiniteCount);
            break;
        default:
            throw std::runtime_error("unknown error metric");
        }
        return static_cast<Operon::Scalar>(value + penaltyWeight * scale * fraction);
    }
} // namespace

auto FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target) noexcept
    -> std::pair<double, double>
{
    return FitLeastSquaresForward<float>(estimated, target);
}

auto FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target) noexcept
    -> std::pair<double, double>
{
    return FitLeastSquaresForward<double>(estimated, target);
}

auto FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target,
    Operon::Span<float const> weights) noexcept -> std::pair<double, double>
{
    return FitLeastSquaresForward<float>(estimated, target, weights);
}

auto FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target,
    Operon::Span<double const> weights) noexcept -> std::pair<double, double>
{
    return FitLeastSquaresForward<double>(estimated, target, weights);
}

TreePropertyEvaluator::TreePropertyEvaluator(
    gsl::not_null<Operon::Problem const*> problem, Property property, Operon::Scalar normalizer)
    : UserDefinedEvaluator(problem,
          [property = std::move(property), normalizer](
              Operon::RandomGenerator& /*unused*/, Operon::Individual const& ind) {
              return EvaluatorBase::ReturnType { property(ind.Genotype) / normalizer };
          })
{
    if (normalizer == Operon::Scalar { 0 }) {
        throw std::invalid_argument("TreePropertyEvaluator normalizer must be non-zero");
    }
}

template <>
auto OPERON_EXPORT Evaluator<ScalarDispatch>::Predict(
    Operon::Individual const& ind, Operon::Span<Operon::Scalar> values) const -> tl::expected<void, InterpreterError>
{
    auto const* problem = GetProblem();
    auto const& tree = ind.Genotype;
    TInterpreter const interpreter { GetDispatchTable(), problem->GetDataset(), &tree };
    auto coeff = tree.GetCoefficients();
    if (auto const evaluated = interpreter.Evaluate(coeff, problem->TrainingRange(), values); !evaluated) {
        return tl::unexpected(evaluated.error());
    }
    return {};
}

template <>
auto OPERON_EXPORT Evaluator<ScalarDispatch>::Evaluate(Operon::Individual const& ind,
    Operon::Span<Operon::Scalar> buf) const -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError>
{
    auto const* problem = GetProblem();
    auto const trainingRange = problem->TrainingRange();

    ENSURE(buf.size() >= trainingRange.Size());
    // buf.size() may exceed trainingRange.Size() (caller-owned scratch reused across calls), but
    // Interpreter::Evaluate and the target/weight spans require an exact-sized view -- slice once, up
    // front, for every downstream use.
    auto estimatedValues = buf.subspan(0, trainingRange.Size());
    ++ResidualEvaluations;
    if (auto const predicted = Predict(ind, estimatedValues); !predicted) {
        return tl::unexpected(predicted.error());
    }
    auto const scaling = detail::FitScalingIfEnabled(*problem, estimatedValues);
    return std::optional<EvaluatedBuffer> { MarkEvaluated(ind, estimatedValues, scaling) };
}

template <>
auto OPERON_EXPORT Evaluator<ScalarDispatch>::Score(Operon::RandomGenerator& /*rng*/, Operon::Individual const& ind,
    Operon::Span<Operon::Scalar> buf, std::optional<EvaluatedBuffer> evaluated) const ->
    typename EvaluatorBase::ReturnType
{
    auto const* problem = GetProblem();
    auto const trainingRange = problem->TrainingRange();
    ENSURE(evaluated.has_value());
    auto estimatedValues = evaluated->Values(ind, buf);
    ENSURE(estimatedValues.size() == trainingRange.Size());
    auto const targetValues = problem->TargetValues(trainingRange);
    auto const weights = problem->Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const> {});

    auto const& scaling = evaluated->Scaling();

    Operon::Scalar fit {};
    if (skipNonFinite_) [[unlikely]] {
        fit = SkipNonFiniteScore<Operon::Scalar>(
            error_, estimatedValues, targetValues, weights, scaling, nonFinitePenaltyWeight_);
    } else {
        if (scaling) {
            scaling->ApplyInPlace(estimatedValues);
        }
        fit = static_cast<Operon::Scalar>(
            weights.empty() ? error_(estimatedValues, targetValues) : error_(estimatedValues, targetValues, weights));
    }

    if (!std::isfinite(fit)) {
        fit = EvaluatorBase::ErrMax;
    }
    return typename EvaluatorBase::ReturnType { fit };
}
auto DiversityEvaluator::Prepare(Operon::Span<Operon::Individual const> pop) const -> void
{
    divmap_.clear();
    for (auto const& individual : pop) {
        auto const& tree = individual.Genotype;
        auto const& nodes = tree.Nodes();
        (void)tree.Hash(hashmode_);
        Operon::Vector<Operon::Hash> hash(nodes.size());
        ;
        std::ranges::transform(nodes, hash.begin(), [](auto const& n) -> auto { return n.CalculatedHashValue; });
        std::ranges::stable_sort(hash);
        divmap_[tree.HashValue()] = std::move(hash);
    }
}

auto DiversityEvaluator::Score(Operon::RandomGenerator& rng, Operon::Individual const& ind,
    Operon::Span<Operon::Scalar> /*buf*/, std::optional<EvaluatedBuffer> /*evaluated*/) const ->
    typename EvaluatorBase::ReturnType
{
    (void)ind.Genotype.Hash(hashmode_);
    Operon::Vector<Operon::Hash> lhs(ind.Genotype.Length());
    auto const& nodes = ind.Genotype.Nodes();
    std::ranges::transform(nodes, lhs.begin(), [](auto const& n) -> auto { return n.CalculatedHashValue; });
    std::ranges::stable_sort(lhs);
    auto const& values = divmap_.values();

    Operon::Scalar distance { 0 };
    Operon::Vector<double> const distances(sampleSize_);
    for (auto i = 0UL; i < sampleSize_; ++i) {
        auto const& rhs = Operon::Random::Sample(rng, values.begin(), values.end())->second;
        distance += static_cast<Operon::Scalar>(Operon::Distance::Jaccard(lhs, rhs));
    }
    return EvaluatorBase::ReturnType { -distance / static_cast<Operon::Scalar>(sampleSize_) };
}

auto MultiEvaluator::Add(EvaluatorBase const* evaluator) -> void
{
    // The adapter owns the complete constrained objective vector: reject the unsupported composition
    // deterministically.
    if (dynamic_cast<detail::ShapeConstraintEnforcer const*>(evaluator) != nullptr) {
        throw std::invalid_argument(
            "MultiEvaluator: a ShapeConstrainedValueEvaluator cannot be a child. It already owns the complete "
            "constrained objective vector (structural objectives included); hand it to the algorithm directly");
    }
    evaluators_.emplace_back(evaluator);
}

auto MultiEvaluator::Score(Operon::RandomGenerator& rng, Operon::Individual const& ind,
    Operon::Span<Operon::Scalar> buf, std::optional<EvaluatedBuffer> /*evaluated*/) const ->
    typename EvaluatorBase::ReturnType
{
    using vstat::univariate::accumulate;

    EvaluatorBase::ReturnType fit;
    fit.reserve(SubEvaluatorObjectiveCount());

    // Every child runs through its own operator() -- the sole evaluator execution path -- in Add order.
    for (auto const& ev : evaluators_) {
        auto f = (*ev)(rng, ind, buf);
        std::copy(f.begin(), f.end(), std::back_inserter(fit));
    }

    if (!aggregateType_) {
        return fit;
    }

    switch (*aggregateType_) {
    case AggregateType::Min: {
        return { *std::ranges::min_element(fit) };
    }
    case AggregateType::Max: {
        return { *std::ranges::max_element(fit) };
    }
    case AggregateType::Median: {
        auto const sz { std::ssize(fit) };
        auto const a = fit.begin() + sz / 2;
        std::nth_element(fit.begin(), a, fit.end());
        if (sz % 2 == 0) {
            auto const b = std::max_element(fit.begin(), a);
            return { (*a + *b) / 2 };
        }
        return { *a };
    }
    case AggregateType::Mean: {
        return { static_cast<Operon::Scalar>(accumulate<Operon::Scalar>(fit.begin(), fit.end()).mean) };
    }
    case AggregateType::HarmonicMean: {
        auto stats = accumulate<Operon::Scalar>(fit.begin(), fit.end(), [](auto x) -> auto { return 1 / x; });
        return { static_cast<Operon::Scalar>(stats.count / stats.sum) };
    }
    case AggregateType::Sum: {
        return { static_cast<Operon::Scalar>(
            vstat::univariate::accumulate<Operon::Scalar>(fit.begin(), fit.end()).sum) };
    }
    default: {
        throw std::runtime_error("Unknown AggregateType");
    }
    }
}

template <>
auto OPERON_EXPORT BayesianInformationCriterionEvaluator<ScalarDispatch>::Score(Operon::RandomGenerator& rng,
    Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf, std::optional<EvaluatedBuffer> evaluated) const ->
    typename EvaluatorBase::ReturnType
{
    auto const& tree = ind.Genotype;
    auto p = static_cast<Operon::Scalar>(std::ranges::count_if(tree.Nodes(), &Operon::Node::Optimize));
    auto n = static_cast<Operon::Scalar>(Evaluator::GetProblem()->TrainingRange().Size());
    auto mse = Evaluator<ScalarDispatch>::Score(rng, ind, buf, std::move(evaluated)).front();
    auto bic = (n * std::log(mse)) + (p * std::log(n));
    if (!std::isfinite(bic)) {
        bic = EvaluatorBase::ErrMax;
    }
    return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(bic) };
}

template <>
auto OPERON_EXPORT AkaikeInformationCriterionEvaluator<ScalarDispatch>::Score(Operon::RandomGenerator& rng,
    Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf, std::optional<EvaluatedBuffer> evaluated) const ->
    typename EvaluatorBase::ReturnType
{
    auto mse = Evaluator<ScalarDispatch>::Score(rng, ind, buf, std::move(evaluated)).front();
    auto n = static_cast<Operon::Scalar>(Evaluator::GetProblem()->TrainingRange().Size());
    auto aik = n / 2 * (std::log(Operon::Math::Tau) + std::log(mse) + 1);
    if (!std::isfinite(aik)) {
        aik = EvaluatorBase::ErrMax;
    }
    return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(aik) };
}

auto LocalSearch(Operon::RandomGenerator& random, Operon::Individual& ind, Operon::EvaluatorBase const& evaluator,
    Operon::CoefficientOptimizer const* coeffOptimizer, double pLocal, double pLamarck)
    -> std::optional<std::vector<Operon::Scalar>>
{
    using BernoulliTrial = std::bernoulli_distribution;

    if (coeffOptimizer == nullptr || pLocal <= 0 || !BernoulliTrial { pLocal }(random)) {
        return std::nullopt;
    }

    auto c = ind.Genotype.GetCoefficients(); // save original coefficients
    auto t0 = std::chrono::steady_clock::now();
    auto [optimizedTree, outcome] = (*coeffOptimizer)(random, std::move(ind.Genotype));
    auto t1 = std::chrono::steady_clock::now();
    auto const& diag = Diagnostics(outcome);
    evaluator.ResidualEvaluations += diag.FunctionEvaluations;
    evaluator.JacobianEvaluations += diag.JacobianEvaluations;
    // After the totals: DetailedStats() reads these first and relies on optimizer <= total.
    evaluator.OptimizerResidualEvaluations += diag.FunctionEvaluations;
    evaluator.OptimizerJacobianEvaluations += diag.JacobianEvaluations;
    evaluator.CostFunctionTime += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    ind.Genotype = std::move(optimizedTree);

    return BernoulliTrial { pLamarck }(random) ? std::nullopt : std::make_optional(std::move(c));
}

auto RestoreNonLamarckianCoefficients(Operon::Individual& ind, Operon::EvaluatorBase const& evaluator,
    Operon::Span<Operon::Scalar const> stored, Operon::Span<Operon::Scalar> buf) -> void
{
    evaluator.FinalizeRestoredCoefficients(ind, stored, buf);
}

auto ScoreIndividual(Operon::RandomGenerator& random, Operon::Individual& ind, Operon::EvaluatorBase const& evaluator,
    Operon::CoefficientOptimizer const* coeffOptimizer, double pLocal, double pLamarck,
    Operon::Span<Operon::Scalar> buf) -> void
{
    auto originalCoeffs = LocalSearch(random, ind, evaluator, coeffOptimizer, pLocal, pLamarck);
    ind.Fitness = evaluator(random, ind, buf);
    if (originalCoeffs) {
        RestoreNonLamarckianCoefficients(ind, evaluator, *originalCoeffs, buf);
    }

    for (auto& v : ind.Fitness) {
        if (!std::isfinite(v)) {
            v = EvaluatorBase::ErrMax;
        }
    }
}
} // namespace Operon
