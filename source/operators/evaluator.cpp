// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/operators/evaluator.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/distance.hpp"
#include "operon/operators/linear_scaling.hpp"
#include "operon/operators/local_search.hpp"
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
        Operon::Span<T const> target, Operon::Span<T const> weights, bool scaling, double penaltyWeight)
        -> Operon::Scalar
    {
        if (scaling) {
            FitLinearScaling(estimated, target, weights, /*omitNonFinite=*/true).ApplyInPlace(estimated);
        }
        auto [value, nonFiniteCount]
            = weights.empty() ? error.FiniteSubset(estimated, target) : error.FiniteSubset(estimated, target, weights);
        if (nonFiniteCount == estimated.size()) {
            return EvaluatorBase::ErrMax;
        }
        auto const fraction
            = nonFiniteCount != 0 ? static_cast<double>(nonFiniteCount) / static_cast<double>(estimated.size()) : 0.0;
        // Per-metric scale so a single penaltyWeight means the same thing regardless of the active metric's
        // units: NMSE already normalizes by target variance (scale = 1); MSE/SSE are squared-error units
        // (variance, and variance * finite-count since SSE is a sum, not an average); RMSE/MAE are linear-error
        // units (stddev = sqrt(variance)).
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
            scale = variance;
            break; // unreachable: R2/C2 reject --skip-nonfinite in ParseEvaluator
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
auto OPERON_EXPORT Evaluator<ScalarDispatch>::Evaluate(Operon::Individual const& ind,
    Operon::Span<Operon::Scalar> buf) const -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError>
{
    auto const* problem = GetProblem();
    auto const trainingRange = problem->TrainingRange();
    auto const& tree = ind.Genotype;
    TInterpreter const interpreter { GetDispatchTable(), problem->GetDataset(), &tree };

    ENSURE(buf.size() >= trainingRange.Size());
    // buf.size() may exceed trainingRange.Size() (caller-owned scratch reused across calls), but
    // Interpreter::TryEvaluate and the target/weight spans require an exact-sized view -- slice once, up
    // front, for every downstream use.
    auto estimatedValues = buf.subspan(0, trainingRange.Size());
    auto coeff = tree.GetCoefficients();
    ++ResidualEvaluations;
    if (auto const evaluated = interpreter.Evaluate(coeff, trainingRange, estimatedValues); !evaluated) {
        return tl::unexpected(std::move(evaluated.error()));
    }
    return std::optional<EvaluatedBuffer> { MarkEvaluated(ind, estimatedValues) };
}

template <>
auto OPERON_EXPORT Evaluator<ScalarDispatch>::Score(
    ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType
{
    ++CallCount;

    auto const* problem = GetProblem();
    auto const trainingRange = problem->TrainingRange();
    ENSURE(evaluated.has_value());
    auto estimatedValues = evaluated->Values(ctx.Ind, ctx.Scratch);
    ENSURE(estimatedValues.size() == trainingRange.Size());
    auto const targetValues = problem->TargetValues(trainingRange);
    auto const weights = problem->Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const> {});

    Operon::Scalar fit {};
    if (skipNonFinite_) [[unlikely]] {
        fit = SkipNonFiniteScore<Operon::Scalar>(
            error_, estimatedValues, targetValues, weights, UsesLinearScaling(), nonFinitePenaltyWeight_);
    } else {
        if (UsesLinearScaling()) {
            FitLinearScaling(estimatedValues, targetValues, weights, /*omitNonFinite=*/false)
                .ApplyInPlace(estimatedValues);
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

auto DiversityEvaluator::Score(ScoreContext ctx, std::optional<EvaluatedBuffer> /*evaluated*/) const ->
    typename EvaluatorBase::ReturnType
{
    ++CallCount;
    (void)ctx.Ind.Genotype.Hash(hashmode_);
    Operon::Vector<Operon::Hash> lhs(ctx.Ind.Genotype.Length());
    auto const& nodes = ctx.Ind.Genotype.Nodes();
    std::ranges::transform(nodes, lhs.begin(), [](auto const& n) -> auto { return n.CalculatedHashValue; });
    std::ranges::stable_sort(lhs);
    auto const& values = divmap_.values();

    Operon::Scalar distance { 0 };
    Operon::Vector<double> const distances(sampleSize_);
    for (auto i = 0UL; i < sampleSize_; ++i) {
        auto const& rhs = Operon::Random::Sample(ctx.Rng, values.begin(), values.end())->second;
        distance += static_cast<Operon::Scalar>(Operon::Distance::Jaccard(lhs, rhs));
    }
    return EvaluatorBase::ReturnType { -distance / static_cast<Operon::Scalar>(sampleSize_) };
}

auto MultiEvaluator::Score(ScoreContext ctx, std::optional<EvaluatedBuffer> /*evaluated*/) const ->
    typename EvaluatorBase::ReturnType
{
    using vstat::univariate::accumulate;

    // Counts one increment per call at every composition depth, not just the leaf Evaluator<DTable>, so a
    // caller (e.g. OffspringSelectionGenerator::SelectionPressure) sees the same per-call semantics regardless
    // of nesting. Stats() below separately sums the inner evaluators' own counters for total sub-evaluator work.
    ++CallCount;

    EvaluatorBase::ReturnType fit;
    fit.reserve(SubEvaluatorObjectiveCount());

    for (auto const& ev : evaluators_) {
        auto f = (*ev)(ctx.Rng, ctx.Ind, ctx.Scratch);
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
auto OPERON_EXPORT BayesianInformationCriterionEvaluator<ScalarDispatch>::Score(
    ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType
{
    auto const& tree = ctx.Ind.Genotype;
    auto p = static_cast<Operon::Scalar>(std::ranges::count_if(tree.Nodes(), &Operon::Node::Optimize));
    auto n = static_cast<Operon::Scalar>(Evaluator::GetProblem()->TrainingRange().Size());
    auto mse = Evaluator<ScalarDispatch>::Score(ctx, std::move(evaluated)).front();
    auto bic = (n * std::log(mse)) + (p * std::log(n));
    if (!std::isfinite(bic)) {
        bic = EvaluatorBase::ErrMax;
    }
    return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(bic) };
}

template <>
auto OPERON_EXPORT AkaikeInformationCriterionEvaluator<ScalarDispatch>::Score(
    ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType
{
    auto mse = Evaluator<ScalarDispatch>::Score(ctx, std::move(evaluated)).front();
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
    evaluator.CostFunctionTime += std::chrono::duration_cast<std::chrono::microseconds>(t1 - t0).count();
    ind.Genotype = std::move(optimizedTree);

    return BernoulliTrial { pLamarck }(random) ? std::nullopt : std::make_optional(std::move(c));
}

auto ScoreIndividual(Operon::RandomGenerator& random, Operon::Individual& ind, Operon::EvaluatorBase const& evaluator,
    Operon::CoefficientOptimizer const* coeffOptimizer, double pLocal, double pLamarck,
    Operon::Span<Operon::Scalar> buf) -> void
{
    auto originalCoeffs = LocalSearch(random, ind, evaluator, coeffOptimizer, pLocal, pLamarck);
    ind.Fitness = evaluator(random, ind, buf);
    if (originalCoeffs) {
        ind.Genotype.SetCoefficients(*originalCoeffs);
    }

    for (auto& v : ind.Fitness) {
        if (!std::isfinite(v)) {
            v = EvaluatorBase::ErrMax;
        }
    }
}
} // namespace Operon
