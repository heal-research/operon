// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/core/distance.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/operators/evaluator.hpp"
#include "operon/random/random.hpp"

#include <algorithm>
#include <cmath>
#include <operon/operon_export.hpp>
#include <type_traits>

namespace Operon {
namespace {
    template<typename T>
    auto FitLeastSquaresImpl(Operon::Span<T const> estimated, Operon::Span<T const> target,
                             Operon::Span<T const> weights = {}) -> std::pair<double, double>
    requires std::is_arithmetic_v<T>
    {
        auto stats = weights.empty()
            ? vstat::bivariate::accumulate<T>(estimated.data(), estimated.data() + estimated.size(), target.data())
            : vstat::bivariate::accumulate<T>(estimated.data(), estimated.data() + estimated.size(), target.data(), weights.data());
        auto a = stats.covariance / stats.variance_x; // scale
        if (!std::isfinite(a)) {
            a = 1;
        }
        auto b = stats.mean_y - (a * stats.mean_x); // offset
        return {a, b};
    }

    // Finite-aware variant: computes scale/offset from the finite subset
    // only, returning the count of skipped (non-finite) pairs. A NaN/Inf row
    // no longer disables scaling for every finite row via NaN-poisoned stats.
    // NaN/Inf rows preserve their non-finiteness through `a*x + b`
    // (NaN->NaN; Inf->+/-Inf when a != 0; Inf->NaN when a == 0 since
    // 0*Inf==NaN before adding b), so the downstream FiniteSubset metric
    // still detects and skips them after the in-place transform regardless
    // of which non-finite value the scaling produces -- both NaN and Inf
    // are excluded by the shared finiteness mask. (The symmetric "could a
    // *previously finite* row become non-finite after scaling?" overflow
    // case is a pre-existing risk of any linear scaling, not introduced
    // or worsened by finite-aware scaling.) See `NormalizedMeanSquaredErrorFinite`
    // / `MeanSquaredErrorFinite` for the mask this composes through.
    template<typename T>
    auto FitLeastSquaresFiniteImpl(Operon::Span<T const> estimated, Operon::Span<T const> target,
                                   Operon::Span<T const> weights = {}) -> std::tuple<double, double, std::size_t>
    requires std::is_arithmetic_v<T>
    {
        auto [stats, skipped] = weights.empty()
            ? vstat::bivariate::accumulate<T, vstat::nan_policy::omit>(estimated.data(), estimated.data() + estimated.size(), target.data())
            : vstat::bivariate::accumulate<T, vstat::nan_policy::omit>(estimated.data(), estimated.data() + estimated.size(), target.data(), weights.data());
        auto a = stats.covariance / stats.variance_x; // scale
        if (!std::isfinite(a)) {
            a = 1;
        }
        auto b = stats.mean_y - (a * stats.mean_x); // offset
        return {a, b, skipped};
    }

    // Outlined skip-mode body. Keeping this out of `Evaluate`'s inline path
    // keeps the default (skipNonFinite_ == false) hot path small enough that
    // the compiler still inlines `Evaluate` into its caller -- the inlining
    // heuristic that regressed when the skip branching was first added was
    // the dominant source of the ~7-9% end-to-end overhead measured in the
    // performance handoff. `noinline` is harmless to the opt-in user since
    // they have already accepted a modest per-call cost.
    template<typename T>
    [[gnu::noinline]] auto
    SkipNonFiniteScore(ErrorMetric const& error, Operon::Span<T> estimated, Operon::Span<T const> target,
                       Operon::Span<T const> weights, bool scaling, double penaltyWeight) -> Operon::Scalar
    {
        if (scaling) {
            auto [a, b, s] = weights.empty()
                ? FitLeastSquaresFiniteImpl<T>(estimated, target)
                : FitLeastSquaresFiniteImpl<T>(estimated, target, weights);
            (void)s;
            std::ranges::transform(estimated, estimated.begin(), [a=a,b=b](auto x) -> auto { return (a * x) + b; });
        }
        auto [value, nonFiniteCount] = weights.empty()
            ? error.FiniteSubset(estimated, target)
            : error.FiniteSubset(estimated, target, weights);
        auto const fraction = nonFiniteCount != 0
            ? static_cast<double>(nonFiniteCount) / static_cast<double>(estimated.size())
            : 0.0;
        // NMSE already normalizes by target variance, so its penalty needs no
        // extra scale. The other metrics are unit-dependent on the target and
        // each other, so the scale has to match each metric's own units, not
        // just SSE/MSE's (squared-error) units, or the same penaltyWeight
        // would over/under-shoot depending on which metric is active:
        //   MSE  is in squared-error units  -> variance
        //   RMSE/MAE are in linear-error units -> stddev (sqrt(variance))
        //   SSE is a *sum*, not an average, of squared errors, so a
        //   per-point variance-scale term alone would be ~N times too small
        //   -> variance * (finite point count)
        auto const variance = weights.empty()
            ? vstat::univariate::accumulate<T>(target.begin(), target.end()).variance
            : vstat::univariate::accumulate<T>(target.begin(), target.end(), weights.begin()).variance;
        double scale{};
        switch (error.Type()) {
        case ErrorType::NMSE: scale = 1.0; break;
        case ErrorType::MSE:  scale = variance; break;
        case ErrorType::RMSE:
        case ErrorType::MAE:  scale = std::sqrt(variance); break;
        case ErrorType::SSE:  scale = variance * static_cast<double>(estimated.size() - nonFiniteCount); break;
        default:              scale = variance; break; // unreachable: R2/C2 reject --skip-nonfinite in ParseEvaluator
        }
        return static_cast<Operon::Scalar>(value + penaltyWeight * scale * fraction);
    }
} // namespace

    auto FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target) noexcept -> std::pair<double, double> {
        return FitLeastSquaresImpl<float>(estimated, target);
    }

    auto FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target) noexcept -> std::pair<double, double> {
        return FitLeastSquaresImpl<double>(estimated, target);
    }

    auto FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target, Operon::Span<float const> weights) noexcept -> std::pair<double, double> {
        return FitLeastSquaresImpl<float>(estimated, target, weights);
    }

    auto FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target, Operon::Span<double const> weights) noexcept -> std::pair<double, double> {
        return FitLeastSquaresImpl<double>(estimated, target, weights);
    }

    template<> auto OPERON_EXPORT
    Evaluator<ScalarDispatch>::Evaluate(Operon::RandomGenerator& /*rng*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType
    {
        ++CallCount;
        auto const* problem = GetProblem();
        auto const* dataset = problem->GetDataset();

        auto const trainingRange = problem->TrainingRange();
        auto const targetValues  = problem->TargetValues(trainingRange);
        auto const weightsOpt    = problem->Weights(trainingRange);
        auto const weights       = weightsOpt.value_or(Operon::Span<Operon::Scalar const>{});

        auto const& tree = ind.Genotype;
        auto const* dtable = GetDispatchTable();
        TInterpreter const interpreter{dtable, dataset, &tree};

        ++ResidualEvaluations;
        ENSURE(buf.size() >= trainingRange.Size());
        // EvaluatorBase::Evaluate's contract permits buf.size() >
        // trainingRange.Size() (a caller-owned scratch buffer sized for
        // reuse across calls), but Interpreter::Evaluate only writes into
        // its result span when it's sized exactly to the range (silently
        // leaving an oversized buffer's tail untouched), and targetValues/
        // weights are always sized to exactly trainingRange.Size(). Slice
        // once, up front, so scaling and the error metric both operate on
        // the same exactly-sized view as the interpreter writes into -
        // same pattern as MinimumDescriptionLengthEvaluator/
        // FractionalBayesFactorEvaluator/LikelihoodEvaluator in evaluator.hpp.
        auto estimatedValues = buf.subspan(0, trainingRange.Size());
        auto coeff = tree.GetCoefficients();
        interpreter.Evaluate(coeff, trainingRange, estimatedValues);

        Operon::Scalar fit{};
        if (skipNonFinite_) [[unlikely]] {
            fit = SkipNonFiniteScore<Operon::Scalar>(error_, estimatedValues, targetValues, weights, scaling_, nonFinitePenaltyWeight_);
        } else {
            if (scaling_) {
                auto [a, b] = weights.empty()
                    ? FitLeastSquaresImpl<Operon::Scalar>(estimatedValues, targetValues)
                    : FitLeastSquaresImpl<Operon::Scalar>(estimatedValues, targetValues, weights);
                std::ranges::transform(estimatedValues, estimatedValues.begin(), [a=a,b=b](auto x) -> auto { return (a * x) + b; });
            }
            fit = static_cast<Operon::Scalar>(weights.empty() ? error_(estimatedValues, targetValues) : error_(estimatedValues, targetValues, weights));
        }

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
            Operon::Vector<Operon::Hash> hash(nodes.size());;
            std::ranges::transform(nodes, hash.begin(), [](auto const& n) -> auto { return n.CalculatedHashValue; });
            std::ranges::stable_sort(hash);
            divmap_[tree.HashValue()] = std::move(hash);
        }
    }

    auto
    DiversityEvaluator::Evaluate(Operon::RandomGenerator& random, Individual const& ind, Operon::Span<Operon::Scalar>  /*buf*/) const -> typename EvaluatorBase::ReturnType
    {
        (void)ind.Genotype.Hash(hashmode_);
        Operon::Vector<Operon::Hash> lhs(ind.Genotype.Length());
        auto const& nodes = ind.Genotype.Nodes();
        std::ranges::transform(nodes, lhs.begin(), [](auto const& n) -> auto { return n.CalculatedHashValue; });
        std::ranges::stable_sort(lhs);
        auto const& values = divmap_.values();

        Operon::Scalar distance{0};
        Operon::Vector<double> const distances(sampleSize_);
        for (auto i = 0UL; i < sampleSize_; ++i) {
            auto const& rhs = Operon::Random::Sample(random, values.begin(), values.end())->second;
            distance += static_cast<Operon::Scalar>(Operon::Distance::Jaccard(lhs, rhs));
        }
        return EvaluatorBase::ReturnType { -distance / static_cast<Operon::Scalar>(sampleSize_) };
    }

    auto
    AggregateEvaluator::Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType
    {
        using vstat::univariate::accumulate;
        auto f = (*evaluator_)(rng, ind, buf);
        switch(aggtype_) {
            case AggregateType::Min: {
                return { *std::ranges::min_element(f) };
            }
            case AggregateType::Max: {
                return { *std::ranges::max_element(f) };
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
                auto stats = accumulate<Operon::Scalar>(f.begin(), f.end(), [](auto x) -> auto { return 1/x; });
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
    BayesianInformationCriterionEvaluator<ScalarDispatch>::Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType {
        auto const& tree = ind.Genotype;
        auto p = static_cast<Operon::Scalar>(std::ranges::count_if(tree.Nodes(), &Operon::Node::Optimize));
        auto n = static_cast<Operon::Scalar>(Evaluator::GetProblem()->TrainingRange().Size());
        auto mse = Evaluator::Evaluate(rng, ind, buf).front();
        auto bic = (n * std::log(mse)) + (p * std::log(n));
        if (!std::isfinite(bic)) { bic = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(bic) };
    }

    template<> auto OPERON_EXPORT
    AkaikeInformationCriterionEvaluator<ScalarDispatch>::Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType {
        auto mse = Evaluator::Evaluate(rng, ind, buf).front();
        auto n = static_cast<Operon::Scalar>(Evaluator::GetProblem()->TrainingRange().Size());
        auto aik = n/2 * (std::log(Operon::Math::Tau) + std::log(mse) + 1);
        if (!std::isfinite(aik)) { aik = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(aik) };
    }
} // namespace Operon
