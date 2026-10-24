// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_EVALUATOR_HPP
#define OPERON_EVALUATOR_HPP

#include <algorithm>
#include <array>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <functional>
#include <optional>
#include <stdexcept>
#include <string>
#include <tl/expected.hpp>
#include <utility>
#include <vector>

#include "operon/collections/projection.hpp"
#include "operon/core/concepts.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/operator.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/types.hpp"
#include "operon/information_criteria/fractional_bayes_factor.hpp"
#include "operon/information_criteria/minimum_description_length.hpp"
#include "operon/information_criteria/weighted_complexity.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operators/linear_scaling.hpp"
#include "operon/operon_export.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include "operon/optimizer/likelihood/statistical_concepts.hpp"

namespace Operon {

class CoefficientOptimizer; // operators/local_search.hpp

enum class ErrorType : std::uint8_t { SSE, MSE, NMSE, RMSE, MAE, R2, C2 };

struct OPERON_EXPORT ErrorMetric {
    using Iterator = Operon::Scalar const*;
    using ProjIterator = ProjectionIterator<Iterator>;

    explicit ErrorMetric(ErrorType type) : type_(type) {}

    [[nodiscard]] auto Type() const noexcept -> ErrorType { return type_; }

    auto operator()(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y) const -> double;
    auto operator()(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y,
        Operon::Span<Operon::Scalar const> w) const -> double;
    auto operator()(Iterator beg1, Iterator end1, Iterator beg2) const -> double;
    auto operator()(Iterator beg1, Iterator end1, Iterator beg2, Iterator beg3) const -> double;

    // Metric over the finite subset of (x, y) pairs, plus the count of
    // skipped (non-finite) pairs.
    [[nodiscard]] auto FiniteSubset(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y) const
        -> std::pair<double, std::size_t>;
    [[nodiscard]] auto FiniteSubset(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y,
        Operon::Span<Operon::Scalar const> w) const -> std::pair<double, std::size_t>;

private:
    ErrorType type_;
};

struct OPERON_EXPORT SSE : public ErrorMetric {
    SSE() : ErrorMetric(ErrorType::SSE) {}
};

struct OPERON_EXPORT MSE : public ErrorMetric {
    MSE() : ErrorMetric(ErrorType::MSE) {}
};

struct OPERON_EXPORT NMSE : public ErrorMetric {
    NMSE() : ErrorMetric(ErrorType::NMSE) {}
};

struct OPERON_EXPORT RMSE : public ErrorMetric {
    RMSE() : ErrorMetric(ErrorType::RMSE) {}
};

struct OPERON_EXPORT MAE : public ErrorMetric {
    MAE() : ErrorMetric(ErrorType::MAE) {}
};

struct OPERON_EXPORT R2 : public ErrorMetric {
    R2() : ErrorMetric(ErrorType::R2) {}
};

struct OPERON_EXPORT C2 : public ErrorMetric {
    C2() : ErrorMetric(ErrorType::C2) {}
};

auto OPERON_EXPORT FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target) noexcept
    -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target) noexcept
    -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target,
    Operon::Span<float const> weights) noexcept -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target,
    Operon::Span<double const> weights) noexcept -> std::pair<double, double>;

struct EvaluatorBase;

// Lifecycle step for a non-Lamarckian local search: restores `stored` as `ind`'s genotype coefficients after the
// optimized coefficients were evaluated into `ind.Fitness`. The default only calls SetCoefficients; the restored
// coefficients are not re-certified. `buf` is the caller-owned scratch buffer used by operator() calls. This is a
// lifecycle hook, not an evaluator execution path: it never re-scores the error.
OPERON_EXPORT auto RestoreNonLamarckianCoefficients(Operon::Individual& ind, Operon::EvaluatorBase const& evaluator,
    Operon::Span<Operon::Scalar const> stored, Operon::Span<Operon::Scalar> buf) -> void;

// Move-only, single-use proof that a caller-owned `buf` span holds a specific Individual's valid per-row raw
// prediction. It also carries the exact transform the producing evaluator will apply to those predictions
// (`nullopt`: linear scaling is disabled). It is handed from `Evaluate` to `Score` inside one operator() call and
// never crosses a generic evaluator boundary.
class EvaluatedBuffer {
public:
    EvaluatedBuffer(EvaluatedBuffer const&) = delete;
    EvaluatedBuffer(EvaluatedBuffer&& other) noexcept
        : individual_(other.individual_)
        , span_(other.span_)
        , scaling_(other.scaling_)
    {
        other.individual_ = nullptr;
        other.span_ = {};
    }
    auto operator=(EvaluatedBuffer const&) -> EvaluatedBuffer& = delete;
    auto operator=(EvaluatedBuffer&& other) noexcept -> EvaluatedBuffer&
    {
        if (this != &other) {
            individual_ = other.individual_;
            span_ = other.span_;
            scaling_ = other.scaling_;
            other.individual_ = nullptr;
            other.span_ = {};
        }
        return *this;
    }
    ~EvaluatedBuffer() = default;

    [[nodiscard]] auto Values(Operon::Individual const& individual, Operon::Span<Operon::Scalar> scratch) const noexcept
        -> Operon::Span<Operon::Scalar>
    {
        ENSURE(ValidFor(individual, scratch));
        return span_;
    }
    // The exact transform Score will apply; `nullopt` means scaling is disabled.
    [[nodiscard]] auto Scaling() const noexcept -> std::optional<LinearScaling> const& { return scaling_; }

private:
    friend struct EvaluatorBase;

    explicit EvaluatedBuffer(
        Operon::Individual const& individual, Operon::Span<Operon::Scalar> span, std::optional<LinearScaling> scaling)
        : individual_(&individual)
        , span_(span)
        , scaling_(scaling)
    {
    }

    [[nodiscard]] auto ValidFor(
        Operon::Individual const& individual, Operon::Span<Operon::Scalar> scratch) const noexcept -> bool
    {
        return individual_ == &individual && span_.data() == scratch.data() && span_.size() <= scratch.size();
    }

    Operon::Individual const* individual_ {};
    Operon::Span<Operon::Scalar> span_;
    std::optional<LinearScaling> scaling_;
};

// Evaluation counters of an evaluator tree, split by who spent them. `Residual*`/`Jacobian*` count the same units
// that the budget is made of (EvaluatorBase::TotalEvaluations): `*Optimizer` is spent by coefficient optimization
// (LocalSearch), `*Scoring` by everything else (the prediction pass of Evaluate, a Jacobian taken by a likelihood
// scorer, ...). Budget consumption is `ResidualTotal() + JacobianTotal()`. Calls is the number of operator() calls
// (candidate scorings) and OptimizerTimeUs the wall time spent in the optimizer, summed over threads.
// The cache counters are not known to evaluators: they stay `nullopt` here and are filled by
// GeneticAlgorithmBase::DetailedStats() when a transposition cache is configured (null means "no cache", not zero).
struct EvaluationStats {
    std::size_t Calls {};
    std::size_t ResidualScoring {};
    std::size_t ResidualOptimizer {};
    std::size_t JacobianScoring {};
    std::size_t JacobianOptimizer {};
    std::size_t OptimizerTimeUs {};
    std::optional<std::size_t> CacheHits {};
    std::optional<std::size_t> CacheLookups {};

    [[nodiscard]] constexpr auto ResidualTotal() const -> std::size_t { return ResidualScoring + ResidualOptimizer; }
    [[nodiscard]] constexpr auto JacobianTotal() const -> std::size_t { return JacobianScoring + JacobianOptimizer; }
    [[nodiscard]] constexpr auto Consumed() const -> std::size_t { return ResidualTotal() + JacobianTotal(); }

    // Adds the evaluation counters of `other` (the cache counters belong to the algorithm and are left alone).
    constexpr auto operator+=(EvaluationStats const& other) -> EvaluationStats&
    {
        Calls += other.Calls;
        ResidualScoring += other.ResidualScoring;
        ResidualOptimizer += other.ResidualOptimizer;
        JacobianScoring += other.JacobianScoring;
        JacobianOptimizer += other.JacobianOptimizer;
        OptimizerTimeUs += other.OptimizerTimeUs;
        return *this;
    }

    [[nodiscard]] constexpr auto operator==(EvaluationStats const&) const -> bool = default;
};

// An evaluator is an OperatorBase: `operator()` is its sole public execution path. It is `final` and performs, in
// order, `Evaluate` (prediction phase), failure handling, and `Score` (objective phase). Both phases are protected
// implementation hooks: a derived evaluator calls its base's phases on itself (ShapeConstrainedValueEvaluator),
// while composition (MultiEvaluator, structural post-objectives) goes through `operator()`. `operator()` owns the
// CallCount increment (exactly one per call, including failure) and the ErrMax failure output, which always has
// `ObjectiveCount()` elements.
struct EvaluatorBase
    : public OperatorBase<Operon::Vector<Operon::Scalar>, Operon::Individual const&, Operon::Span<Operon::Scalar>> {
    using Base = OperatorBase<Operon::Vector<Operon::Scalar>, Operon::Individual const&, Operon::Span<Operon::Scalar>>;
    using ReturnType = Base::ReturnType;

    mutable std::atomic_ulong ResidualEvaluations { 0 }; // NOLINT
    mutable std::atomic_ulong JacobianEvaluations { 0 }; // NOLINT
    mutable std::atomic_ulong CallCount { 0 }; // NOLINT
    mutable std::atomic_ulong CostFunctionTime { 0 }; // NOLINT
    // The part of ResidualEvaluations/JacobianEvaluations that was spent by coefficient optimization (LocalSearch).
    // Always incremented after the totals above, so that a reader loading these first never sees more optimizer
    // than total evaluations (see DetailedStats).
    mutable std::atomic_ulong OptimizerResidualEvaluations { 0 }; // NOLINT
    mutable std::atomic_ulong OptimizerJacobianEvaluations { 0 }; // NOLINT

    static constexpr size_t DefaultEvaluationBudget = 100'000;

    static auto constexpr ErrMax { std::numeric_limits<Operon::Scalar>::max() };

    EvaluatorBase(EvaluatorBase const&) = delete;
    auto operator=(EvaluatorBase const&) -> EvaluatorBase& = delete;
    EvaluatorBase(EvaluatorBase&&) = delete;
    auto operator=(EvaluatorBase&&) -> EvaluatorBase& = delete;

    ~EvaluatorBase() override = default;

    explicit EvaluatorBase(gsl::not_null<Problem const*> problem) : problem_(problem) {}

    auto operator()(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const
        -> ReturnType final
    {
        ++CallCount;
        auto evaluated = Evaluate(ind, buf);
        if (!evaluated) {
            return ReturnType(ObjectiveCount(), EvaluatorBase::ErrMax);
        }
        return Score(rng, ind, buf, std::move(*evaluated));
    }

    // Non-virtual deducing-this 2-arg facade: allocates a TrainingRange()-sized scratch
    // buffer and forwards to the 3-arg operator() above.
    template <typename Self>
    auto operator()(this Self const& self, Operon::RandomGenerator& rng, Operon::Individual const& ind) -> ReturnType
    {
        std::vector<Operon::Scalar> buf(self.GetProblem()->TrainingRange().Size());
        return self(rng, ind, buf);
    }

    virtual void Prepare(Operon::Span<Individual const> /*pop*/) const {}

    virtual auto ObjectiveCount() const -> std::size_t { return 1UL; }

    // True iff a fitness computed for one genotype may be reused for any structurally identical genotype (the
    // Zobrist transposition cache ignores coefficient values). A shape-constrained evaluator's result depends on
    // the stored coefficients, so it bypasses structural fitness reuse.
    [[nodiscard]] virtual auto StructuralFitnessReusable() const -> bool { return true; }

    auto TotalEvaluations() const -> size_t { return ResidualEvaluations + JacobianEvaluations; }

    void SetBudget(size_t value) { budget_ = value; }
    auto Budget() const -> size_t { return budget_; }

    // virtual because more complex evaluators (e.g. MultiEvaluator) might need to calculate it differently
    virtual auto BudgetExhausted() const -> bool { return TotalEvaluations() >= Budget(); }

    virtual auto Stats() const -> std::tuple<std::size_t, std::size_t, std::size_t, std::size_t>
    {
        return std::tuple { ResidualEvaluations.load(), JacobianEvaluations.load(), CallCount.load(),
            CostFunctionTime.load() };
    }

    // Like Stats(), with the optimizer/scoring split of the residual and Jacobian evaluations (the totals are the
    // same as Stats()). Composite evaluators (MultiEvaluator, the shape-constrained adapter) include their children.
    // Safe to call while the evaluator is in use; the result is then a consistent-enough snapshot (every optimizer
    // evaluation is also counted in its total, never the other way round), exact once evaluation has stopped.
    virtual auto DetailedStats() const -> EvaluationStats
    {
        // Optimizer counters are read before the totals; LocalSearch bumps the total first, then the optimizer part.
        auto const optimizerResidual = OptimizerResidualEvaluations.load();
        auto const optimizerJacobian = OptimizerJacobianEvaluations.load();
        auto const residual = ResidualEvaluations.load();
        auto const jacobian = JacobianEvaluations.load();
        auto const clampedResidual = std::min<std::size_t>(optimizerResidual, residual);
        auto const clampedJacobian = std::min<std::size_t>(optimizerJacobian, jacobian);
        return EvaluationStats {
            .Calls = CallCount.load(),
            .ResidualScoring = residual - clampedResidual,
            .ResidualOptimizer = clampedResidual,
            .JacobianScoring = jacobian - clampedJacobian,
            .JacobianOptimizer = clampedJacobian,
            .OptimizerTimeUs = CostFunctionTime.load(),
        };
    }

    auto Population() const -> Operon::Span<Individual const> { return population_; }
    auto SetPopulation(Operon::Span<Operon::Individual const> pop) const { population_ = pop; }
    auto GetProblem() const -> Problem const* { return problem_; }
    auto SetProblem(gsl::not_null<Problem const*> problem) { problem_ = problem; }

    // Zeroes every counter of this evaluator, then calls ResetChildren() so that composite evaluators also clear
    // the evaluators they hold and a re-run in the same process starts from a clean budget. Deliberately not
    // virtual: a derived `Reset()` of its own hides this one instead of replacing it, so the base counters are
    // always cleared when called through an `EvaluatorBase`. Extend it through ResetChildren().
    void Reset() const
    {
        ResidualEvaluations = 0;
        JacobianEvaluations = 0;
        CallCount = 0;
        CostFunctionTime = 0;
        OptimizerResidualEvaluations = 0;
        OptimizerJacobianEvaluations = 0;
        ResetChildren();
    }

protected:
    // Extension point of Reset(): reset whatever counters this evaluator owns beyond the base ones (child
    // evaluators, structural counters). Called after the base counters were cleared; must not call Reset() on
    // `this`.
    virtual void ResetChildren() const {}

    // Prediction phase. Returns `nullopt` for an objective-only evaluator, or the prediction proof for a value
    // evaluator; an error is turned into the ErrMax vector by `operator()`.
    [[nodiscard]] virtual auto Evaluate(Operon::Individual const& /*ind*/, Operon::Span<Operon::Scalar> /*buf*/) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError>
    {
        return std::nullopt;
    }

    // Objective phase. `evaluated` is the invocation-local artifact produced by this evaluator's own `Evaluate`.
    virtual auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> ReturnType
        = 0;

    // See RestoreNonLamarckianCoefficients. The default only restores the coefficients.
    virtual void FinalizeRestoredCoefficients(
        Operon::Individual& ind, Operon::Span<Operon::Scalar const> stored, Operon::Span<Operon::Scalar> /*buf*/) const
    {
        ind.Genotype.SetCoefficients(stored);
    }

    [[nodiscard]] static auto MarkEvaluated(Operon::Individual const& individual, Operon::Span<Operon::Scalar> values,
        std::optional<LinearScaling> scaling) -> EvaluatedBuffer
    {
        return EvaluatedBuffer { individual, values, scaling };
    }

private:
    // The lifecycle helper reaches the restoration hook; nothing else can run an evaluator phase from outside the
    // evaluator hierarchy.
    friend auto RestoreNonLamarckianCoefficients(Operon::Individual&, Operon::EvaluatorBase const&,
        Operon::Span<Operon::Scalar const>, Operon::Span<Operon::Scalar>) -> void;

    mutable Operon::Span<Operon::Individual const> population_;
    gsl::not_null<Problem const*> problem_;
    size_t budget_ = DefaultEvaluationBudget;
};

// Optionally runs local search (coefficient optimization) on `ind`'s genotype with
// probability `pLocal`; does not score it. Non-Lamarckian updates return the original
// coefficients so the caller can evaluate the optimized genotype, then restore them.
OPERON_EXPORT auto LocalSearch(Operon::RandomGenerator& random, Operon::Individual& ind,
    Operon::EvaluatorBase const& evaluator, Operon::CoefficientOptimizer const* coeffOptimizer, double pLocal,
    double pLamarck) -> std::optional<std::vector<Operon::Scalar>>;

// Runs LocalSearch then scores via `evaluator`; non-finite fitness clamps to ErrMax. A non-Lamarckian update is
// restored through RestoreNonLamarckianCoefficients after the optimized coefficients are scored.
OPERON_EXPORT auto ScoreIndividual(Operon::RandomGenerator& random, Operon::Individual& ind,
    Operon::EvaluatorBase const& evaluator, Operon::CoefficientOptimizer const* coeffOptimizer, double pLocal,
    double pLamarck, Operon::Span<Operon::Scalar> buf) -> void;

class OPERON_EXPORT UserDefinedEvaluator : public EvaluatorBase {
public:
    UserDefinedEvaluator(gsl::not_null<Problem const*> problem,
        std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual const&)> func)
        : EvaluatorBase(problem)
        , fref_(std::move(func))
    {
    }

    // the func signature taking a pointer to the rng is a workaround for pybind11, since the random generator is
    // non-copyable we have to pass a pointer
    UserDefinedEvaluator(gsl::not_null<Problem const*> problem,
        std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual const&)> func)
        : EvaluatorBase(problem)
        , fptr_(std::move(func))
    {
    }

protected:
    auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> /*buf*/,
        std::optional<EvaluatedBuffer> /*evaluated*/) const -> typename EvaluatorBase::ReturnType override
    {
        return fptr_ ? fptr_(&rng, ind) : fref_(rng, ind);
    }

private:
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual const&)> fref_;
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual const&)>
        fptr_; // workaround for pybind11
};

namespace detail {
    // Fits (scale, offset) of `values` against `problem`'s training target under the problem's declared
    // non-finite fitting policy (Problem::LinearScalingOmitsNonFinite), the single policy shared by every
    // evaluator, backend, shape certification and exporter; nullopt iff linear scaling is disabled on the problem.
    inline auto FitScalingIfEnabled(Operon::Problem const& problem, Operon::Span<Operon::Scalar const> values)
        -> std::optional<LinearScaling>
    {
        if (!problem.LinearScalingEnabled()) {
            return std::nullopt;
        }
        auto const range = problem.TrainingRange();
        return Operon::FitLinearScaling(values, problem.TargetValues(range),
            problem.Weights(range).value_or(Operon::Span<Operon::Scalar const> {}),
            problem.LinearScalingOmitsNonFinite());
    }

} // namespace detail

// Interpreter-backed value evaluator. Its prediction phase fits the linear scaling under the problem's declared
// policy (Problem::LinearScalingOmitsNonFinite); `skipNonFinite` is a scoring-only option and never changes how
// the scaling is fitted, so it is accepted only when the problem's fitting policy already omits non-finite rows
// (or linear scaling is disabled). Backends (JIT) override `Predict` only and share this class's scoring.
template <typename DTable = ScalarDispatch> class OPERON_EXPORT Evaluator : public EvaluatorBase {
public:
    using TDispatch = DTable;
    using TInterpreter = Operon::Interpreter<Operon::Scalar, DTable>;

    explicit Evaluator(gsl::not_null<Problem const*> problem, gsl::not_null<DTable const*> dtable,
        ErrorMetric error = MSE {}, bool skipNonFinite = false, double nonFinitePenaltyWeight = 1.0)
        : EvaluatorBase(problem)
        , dtable_(dtable)
        , error_(error)
        , skipNonFinite_(skipNonFinite)
        , nonFinitePenaltyWeight_(nonFinitePenaltyWeight)
    {
        if (!std::isfinite(nonFinitePenaltyWeight_) || nonFinitePenaltyWeight_ < 0.0) {
            throw std::invalid_argument("non-finite penalty weight must be finite and non-negative");
        }
        if (skipNonFinite_ && problem->LinearScalingEnabled() && !problem->LinearScalingOmitsNonFinite()) {
            throw std::invalid_argument("skipNonFinite requires Problem::SetLinearScalingOmitsNonFinite(true) while "
                                        "linear scaling is enabled: the scaling fit policy is declared by the problem");
        }
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

protected:
    // Fills `values` (exactly TrainingRange().Size() rows) with the genotype's raw prediction. Backends override
    // only this step; counting, scaling fit and scoring are shared.
    [[nodiscard]] virtual auto Predict(Operon::Individual const& ind, Operon::Span<Operon::Scalar> values) const
        -> tl::expected<void, InterpreterError>;

    // One prediction evaluation: Predict, then the fitted-scaling transform that Score will apply.
    auto Evaluate(Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> override;

    // Skip-nonfinite scoring or the evaluated transform's application, then the error metric over `evaluated`'s
    // values.
    auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override;

private:
    gsl::not_null<DTable const*> dtable_;
    ErrorMetric error_;
    // Opt-in: when true, non-finite rows are excluded (ErrorMetric::FiniteSubset) and a
    // metric-scaled penalty is added instead (see SkipNonFiniteScore). Default clamps
    // a non-finite metric result to ErrMax.
    bool skipNonFinite_ { false };
    double nonFinitePenaltyWeight_ { 1.0 };
};

class OPERON_EXPORT MultiEvaluator : public EvaluatorBase {
public:
    // When AggregateType is set, per-evaluator results combine into one scalar instead
    // of concatenating into a multi-objective vector.
    enum class AggregateType : int { Min, Max, Median, Mean, HarmonicMean, Sum };

    explicit MultiEvaluator(Problem const* problem) : EvaluatorBase(problem) {}

    // Shared children: Add() does not reject a child that is already registered (directly or through a nested
    // composite). Stats(), DetailedStats() and BudgetExhausted() all sum the children slot by slot, and a child runs
    // once per slot (its own counters grow once per run), so a child occupying k slots contributes k times its own
    // counters. The three always agree and the budget consumes exactly what they report; Reset() reaching a shared
    // child more than once is harmless.
    // Registers a child that runs through its own operator(). Throws std::invalid_argument for a
    // ShapeConstrainedValueEvaluator: it owns the whole constrained objective vector (including its structural
    // objectives) and its non-Lamarckian restoration step (RestoreNonLamarckianCoefficients) is only reachable when
    // it is the evaluator handed to the algorithm, so inside a composition that step would be silently skipped.
    auto Add(EvaluatorBase const* evaluator) -> void;

    auto Prepare(Operon::Span<Operon::Individual const> pop) const -> void override
    {
        for (auto const& e : evaluators_) {
            e->Prepare(pop);
        }
    }

    auto SetAggregateType(std::optional<AggregateType> type) { aggregateType_ = type; }
    auto ClearAggregateType() { aggregateType_ = std::nullopt; }
    auto GetAggregateType() const -> std::optional<AggregateType> { return aggregateType_; }

    auto ObjectiveCount() const -> std::size_t override { return aggregateType_ ? 1UL : SubEvaluatorObjectiveCount(); }

    // Reusable only if every child's fitness is reusable across structurally identical genotypes.
    [[nodiscard]] auto StructuralFitnessReusable() const -> bool override
    {
        return std::ranges::all_of(evaluators_, [](auto const& ev) { return ev->StructuralFitnessReusable(); });
    }

    auto Stats() const -> std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> final
    {
        auto resEval { 0UL };
        auto jacEval { 0UL };
        auto cfTime { 0UL };

        for (auto const& ev : evaluators_) {
            auto [re, je, cc, ct] = ev->Stats();
            resEval += re;
            jacEval += je;
            cfTime += ct;
        }

        return std::tuple { resEval + ResidualEvaluations.load(), jacEval + JacobianEvaluations.load(),
            CallCount.load(), cfTime + CostFunctionTime.load() };
    }

    auto BudgetExhausted() const -> bool final
    {
        auto [re, je, cc, ct] = Stats();
        return re + je >= Budget();
    }

    // Own counters plus those of every child (Calls is the composite's own, like Stats()).
    auto DetailedStats() const -> EvaluationStats override
    {
        auto stats = EvaluatorBase::DetailedStats();
        auto const calls = stats.Calls;
        for (auto const& ev : evaluators_) {
            stats += ev->DetailedStats();
        }
        stats.Calls = calls;
        return stats;
    }

    auto Evaluators() const { return evaluators_; }

protected:
    void ResetChildren() const override
    {
        for (auto const& ev : evaluators_) {
            ev->Reset();
        }
    }

    // Children run exclusively through their operator().
    auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override;

private:
    auto SubEvaluatorObjectiveCount() const -> std::size_t
    {
        return std::transform_reduce(evaluators_.begin(), evaluators_.end(), 0UL, std::plus {},
            [](auto const eval) { return eval->ObjectiveCount(); });
    }

    std::vector<gsl::not_null<EvaluatorBase const*>> evaluators_;
    std::optional<AggregateType> aggregateType_;
};

// A genotype-only objective (e.g. tree length). It never produces predictions -- its `Evaluate` phase is sealed --
// which makes it the one evaluator kind a shape-constrained evaluator accepts as a structural post-objective.
class OPERON_EXPORT TreePropertyEvaluator : public UserDefinedEvaluator {
public:
    using Property = std::function<Operon::Scalar(Operon::Tree const&)>;

    explicit TreePropertyEvaluator(
        gsl::not_null<Operon::Problem const*> problem, Property property, Operon::Scalar normalizer = 1);

protected:
    auto Evaluate(Operon::Individual const& /*ind*/, Operon::Span<Operon::Scalar> /*buf*/) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> final
    {
        return std::nullopt;
    }
};

class OPERON_EXPORT DiversityEvaluator : public EvaluatorBase {
public:
    explicit DiversityEvaluator(Operon::Problem const* problem, Operon::HashMode hashmode = Operon::HashMode::Strict,
        std::size_t sampleSize = 100)
        : EvaluatorBase(problem)
        , hashmode_(hashmode)
        , sampleSize_(sampleSize)
    {
    }

    auto Prepare(Operon::Span<Operon::Individual const> pop) const -> void override;

protected:
    auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override;

private:
    mutable Operon::Map<Operon::Hash, Operon::Vector<Operon::Hash>> divmap_;
    Operon::HashMode hashmode_ { Operon::HashMode::Strict };
    std::size_t sampleSize_ {};
};

// See core/concepts.hpp for why these are asserted here rather than constraining a template.
// TreePropertyEvaluator inherits UserDefinedEvaluator's Score override without
// overriding it, so UserDefinedEvaluator's assert below already covers it.
static_assert(Concepts::EvaluatorCallable<UserDefinedEvaluator>);
static_assert(Concepts::EvaluatorCallable<Evaluator<ScalarDispatch>>);
static_assert(Concepts::EvaluatorCallable<MultiEvaluator>);
static_assert(Concepts::EvaluatorCallable<DiversityEvaluator>);

namespace detail {
    // Profile MLE sigma-hat = sqrt(SSR/n), clamped away from zero. Used when the caller
    // hasn't supplied its own sigma.
    inline auto ProfileSigma(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target)
        -> Operon::Scalar
    {
        // Bounded by the shorter span: callers only guarantee estimated.size() >= target.size().
        auto const count = std::min(estimated.size(), target.size());
        auto const n = static_cast<double>(count);
        auto ssr = 0.0;
        for (std::size_t i = 0; i < count; ++i) {
            auto const e = static_cast<double>(estimated[i]) - static_cast<double>(target[i]);
            ssr += e * e;
        }
        return std::max(
            static_cast<Operon::Scalar>(std::sqrt(ssr / n)), std::numeric_limits<Operon::Scalar>::epsilon());
    }

    // Predicted/target/weight spans over `evaluated`'s values, with the
    // producer's claimed scaling applied in place.
    struct ScaledValues {
        Operon::Range TrainingRange;
        Operon::Span<Operon::Scalar> YPred; // evaluated->Values(), scaled in place if Scaling is set
        Operon::Span<Operon::Scalar const> YTrue;
        Operon::Span<Operon::Scalar const> Weights;
        std::optional<LinearScaling> Scaling;
    };

    inline auto ValidateSigma(Operon::Span<Operon::Scalar const> sigma, bool required = false) -> void
    {
        if (required && sigma.empty()) {
            throw std::invalid_argument("sigma must not be empty");
        }
        for (std::size_t i = 0; i < sigma.size(); ++i) {
            if (!std::isfinite(static_cast<double>(sigma[i])) || sigma[i] <= Operon::Scalar {}) {
                throw std::invalid_argument("sigma[" + std::to_string(i) + "] must be finite and greater than zero");
            }
        }
    }

    inline auto PrepareScaledValues(Operon::Problem const& problem, Operon::Individual const& ind,
        Operon::Span<Operon::Scalar> buf, std::optional<EvaluatedBuffer>& evaluated) -> ScaledValues
    {
        auto const trainingRange = problem.TrainingRange();
        ENSURE(evaluated.has_value());
        auto yPred = evaluated->Values(ind, buf);
        auto yTrue = problem.TargetValues(trainingRange);
        auto const weights = problem.Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const> {});
        auto const& scaling = evaluated->Scaling();
        if (scaling) {
            scaling->ApplyInPlace(yPred);
        }
        return {
            .TrainingRange = trainingRange, .YPred = yPred, .YTrue = yTrue, .Weights = weights, .Scaling = scaling
        };
    }
} // namespace detail

template <typename DTable, Concepts::Likelihood Lik>
    requires Concepts::HasFisherDiagonal<Lik>
class OPERON_EXPORT MinimumDescriptionLengthEvaluator : public Evaluator<DTable> {
    // Scores the same fitted linear-scaled model as pareto_front.cpp export and shape certification,
    // closing the previous in-search/exported MDL divergence for the same individual.
    using Base = Evaluator<DTable>;

public:
    explicit MinimumDescriptionLengthEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, SSE {})
    {
    }

    auto Sigma() const { return std::span<Operon::Scalar const> { sigma_ }; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void
    {
        if constexpr (Lik::UsesSigma) {
            detail::ValidateSigma(sigma);
        }
        sigma_ = std::move(sigma);
    }

protected:
    auto Score(Operon::RandomGenerator& /*rng*/, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override
    {
        auto const* dtable = Base::GetDispatchTable();
        auto const* problem = Base::GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const& tree = ind.Genotype;
        auto parameters = tree.GetCoefficients();

        auto [trainingRange, yPred, yTrue, weights, scaling]
            = detail::PrepareScaledValues(*problem, ind, buf, evaluated);

        Operon::Scalar profiledSigma {};
        if (sigma_.empty() && Lik::UsesSigma) {
            profiledSigma = detail::ProfileSigma(yPred, yTrue);
        }

        auto const effectiveSigma = (sigma_.empty() && Lik::UsesSigma)
            ? std::span<Operon::Scalar const> { &profiledSigma, 1 } // profiled
            : std::span<Operon::Scalar const> { sigma_ }; // fixed scalar, per-sample, or empty (Poisson unweighted)

        ++Base::JacobianEvaluations;
        auto const columns = parameters.size();
        auto jacobianStorage = std::vector<Operon::Scalar>(trainingRange.Size() * columns);
        using Extents = std::dextents<MemoryIndex, 2>;
        using Mapping = std::layout_stride::mapping<Extents>;
        // Column stride must stay nonzero for an empty training range (extent 0
        // addresses no element, but a zero stride violates layout_stride preconditions).
        auto const columnStride = std::max<MemoryIndex>(trainingRange.Size(), MemoryIndex { 1 });
        auto jacobian = ScalarMatrixView { jacobianStorage.data(),
            Mapping { Extents { trainingRange.Size(), columns }, std::array<MemoryIndex, 2> { 1, columnStride } } };
        Operon::Interpreter<Operon::Scalar, DTable> const interpreter { dtable, dataset, &tree };
        if (auto result = interpreter.JacRev(parameters, trainingRange, jacobianStorage); !result) {
            return typename EvaluatorBase::ReturnType { EvaluatorBase::ErrMax };
        }
        if (scaling) {
            for (std::size_t row = 0; row < trainingRange.Size(); ++row) {
                for (std::size_t column = 0; column < columns; ++column) {
                    At(jacobian, row, column) *= static_cast<Operon::Scalar>(scaling->Scale);
                }
            }
        }
        auto fisherDiagonal = std::vector<Operon::Scalar>(columns);
        if (auto result = Lik::ComputeFisherDiagonal(yPred, jacobian, effectiveSigma, fisherDiagonal); !result) {
            // Candidate-local numerical failures score ErrMax. A length
            // mismatch (InvalidShape, e.g. a wrong-length SetSigma) is a
            // configuration error and must not silently rank every
            // individual at ErrMax; MakeMdlScorer and WriteParetoFront throw too.
            if (result.error().Code == FisherErrorCode::NonFiniteResult
                || (sigma_.empty() && Lik::UsesSigma && result.error().Code == FisherErrorCode::InvalidSigma)) {
                return typename EvaluatorBase::ReturnType { EvaluatorBase::ErrMax };
            }
            throw std::runtime_error("failed to compute Fisher diagonal");
        }

        auto cLikelihood = Lik::ComputeLikelihood(yPred, yTrue, effectiveSigma);
        auto mdl = Operon::MinimumDescriptionLength(tree, parameters, fisherDiagonal, static_cast<double>(cLikelihood));
        if (!std::isfinite(mdl)) {
            mdl = EvaluatorBase::ErrMax;
        }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(mdl) };
    }

private:
    mutable std::vector<Operon::Scalar> sigma_;
};

template <typename DTable, Concepts::Likelihood Lik>
class OPERON_EXPORT FractionalBayesFactorEvaluator : public Evaluator<DTable> {
    // Scores the same fitted linear-scaled model as pareto_front.cpp export and shape certification,
    // closing the previous in-search/exported FBF divergence for the same individual.
    using Base = Evaluator<DTable>;

public:
    explicit FractionalBayesFactorEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, SSE {})
    {
    }

    auto Sigma() const { return std::span<Operon::Scalar const> { sigma_ }; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void
    {
        if constexpr (Lik::UsesSigma) {
            detail::ValidateSigma(sigma);
        }
        sigma_ = std::move(sigma);
    }

protected:
    auto Score(Operon::RandomGenerator& /*rng*/, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override
    {
        auto const* problem = Base::GetProblem();
        auto const& tree = ind.Genotype;

        auto [trainingRange, estimatedValues, targetValues, weights, scaling]
            = detail::PrepareScaledValues(*problem, ind, buf, evaluated);
        auto const n { static_cast<double>(trainingRange.Size()) };

        double mlNLL {};
        Operon::Scalar profiledSigma {};
        if (sigma_.empty() && Lik::UsesSigma) { // NLL = 0.5*n*(log(2*pi*sigma^2)+1), clamped to avoid log(0)
            profiledSigma = detail::ProfileSigma(estimatedValues, targetValues);
            auto const s = static_cast<double>(profiledSigma);
            mlNLL = 0.5 * n * (std::log(Operon::Math::Tau * s * s) + 1.0);
        }
        auto const effectiveSigma = (sigma_.empty() && Lik::UsesSigma)
            ? std::span<Operon::Scalar const> { &profiledSigma, 1 } // profiled
            : std::span<Operon::Scalar const> { sigma_ }; // fixed scalar, per-sample, or empty (Poisson unweighted)

        auto const nll = (sigma_.empty() && Lik::UsesSigma)
            ? mlNLL
            : static_cast<double>(Lik::ComputeLikelihood(estimatedValues, targetValues, effectiveSigma));

        auto fbf = Operon::FractionalBayesFactor(tree, n, nll);
        if (!std::isfinite(fbf)) {
            fbf = EvaluatorBase::ErrMax;
        }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(fbf) };
    }

private:
    mutable std::vector<Operon::Scalar> sigma_;
};

template <typename DTable> class OPERON_EXPORT BayesianInformationCriterionEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit BayesianInformationCriterionEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, MSE {})
    {
    }

protected:
    auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override;
};

template <typename DTable> class OPERON_EXPORT AkaikeInformationCriterionEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit AkaikeInformationCriterionEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, MSE {})
    {
    }

protected:
    auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override;
};

template <typename DTable, Concepts::Likelihood Likelihood = GaussianLikelihood<Operon::Scalar>>
    requires(DTable::template SupportsType<typename Likelihood::Scalar>)
class OPERON_EXPORT LikelihoodEvaluator : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit LikelihoodEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable)
        , sigma_(1, 0.001)
    {
    }

    auto Sigma() const { return std::span<Operon::Scalar const> { sigma_ }; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void
    {
        if constexpr (Likelihood::UsesSigma) {
            detail::ValidateSigma(sigma, true);
        }
        sigma_ = std::move(sigma);
    }

protected:
    auto Score(Operon::RandomGenerator& /*rng*/, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override
    {
        auto const* problem = Base::Evaluator::GetProblem();
        auto scaled = detail::PrepareScaledValues(*problem, ind, buf, evaluated);
        auto estimatedValues = scaled.YPred;
        auto targetValues = scaled.YTrue;

        auto lik = Likelihood::ComputeLikelihood(estimatedValues, targetValues, sigma_);
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(lik) };
    }

private:
    mutable std::vector<Operon::Scalar> sigma_;
};

template <typename DTable>
using GaussianLikelihoodEvaluator = LikelihoodEvaluator<DTable, GaussianLikelihood<Operon::Scalar>>;

template <typename DTable>
using PoissonLikelihoodEvaluator = LikelihoodEvaluator<DTable, PoissonLikelihood<Operon::Scalar>>;

static_assert(Concepts::EvaluatorCallable<BayesianInformationCriterionEvaluator<ScalarDispatch>>);
static_assert(Concepts::EvaluatorCallable<AkaikeInformationCriterionEvaluator<ScalarDispatch>>);
static_assert(Concepts::EvaluatorCallable<LikelihoodEvaluator<ScalarDispatch>>);

// The closed set of value evaluators that can be the base of a shape-constrained adapter. BIC and AIC stay sealed:
// no factory builds a constrained BIC/AIC evaluator.
static_assert(Concepts::ValueEvaluator<Evaluator<ScalarDispatch>>);
static_assert(
    Concepts::ValueEvaluator<MinimumDescriptionLengthEvaluator<ScalarDispatch, GaussianLikelihood<Operon::Scalar>>>);
static_assert(
    Concepts::ValueEvaluator<MinimumDescriptionLengthEvaluator<ScalarDispatch, PoissonLikelihood<Operon::Scalar>>>);
static_assert(
    Concepts::ValueEvaluator<FractionalBayesFactorEvaluator<ScalarDispatch, GaussianLikelihood<Operon::Scalar>>>);
static_assert(Concepts::ValueEvaluator<LikelihoodEvaluator<ScalarDispatch>>);
static_assert(!Concepts::ValueEvaluator<MultiEvaluator>);
static_assert(!Concepts::ValueEvaluator<TreePropertyEvaluator>);
static_assert(!Concepts::ValueEvaluator<UserDefinedEvaluator>);

} // namespace Operon
#endif
