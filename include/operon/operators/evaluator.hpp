// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_EVALUATOR_HPP
#define OPERON_EVALUATOR_HPP

#include <atomic>
#include <cmath>
#include <functional>
#include <optional>
#include <stdexcept>
#include <utility>
#include <vector>
#include <tl/expected.hpp>

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
#include "operon/optimizer/likelihood/likelihood_base.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"
#include <tl/expected.hpp>

namespace Operon {

class CoefficientOptimizer; // operators/local_search.hpp

enum class ErrorType : int { SSE, MSE, NMSE, RMSE, MAE, R2, C2 };

struct OPERON_EXPORT ErrorMetric {
    using Iterator = Operon::Scalar const*;
    using ProjIterator = ProjectionIterator<Iterator>;

    explicit ErrorMetric(ErrorType type)
        : type_(type)
    {
    }

    [[nodiscard]] auto Type() const noexcept -> ErrorType { return type_; }

    auto operator()(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y) const -> double;
    auto operator()(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y,
        Operon::Span<Operon::Scalar const> w) const -> double;
    auto operator()(Iterator beg1, Iterator end1, Iterator beg2) const -> double;
    auto operator()(Iterator beg1, Iterator end1, Iterator beg2, Iterator beg3) const -> double;

    // Metric over the finite subset of (x, y) pairs, plus the count of
    // skipped (non-finite) pairs. SSE, MSE, NMSE, RMSE and MAE; throws otherwise.
    auto FiniteSubset(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y) const
        -> std::pair<double, std::size_t>;
    auto FiniteSubset(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y,
        Operon::Span<Operon::Scalar const> w) const -> std::pair<double, std::size_t>;

private:
    ErrorType type_;
};

struct OPERON_EXPORT SSE : public ErrorMetric {
    SSE()
        : ErrorMetric(ErrorType::SSE)
    {
    }
};

struct OPERON_EXPORT MSE : public ErrorMetric {
    MSE()
        : ErrorMetric(ErrorType::MSE)
    {
    }
};

struct OPERON_EXPORT NMSE : public ErrorMetric {
    NMSE()
        : ErrorMetric(ErrorType::NMSE)
    {
    }
};

struct OPERON_EXPORT RMSE : public ErrorMetric {
    RMSE()
        : ErrorMetric(ErrorType::RMSE)
    {
    }
};

struct OPERON_EXPORT MAE : public ErrorMetric {
    MAE()
        : ErrorMetric(ErrorType::MAE)
    {
    }
};

struct OPERON_EXPORT R2 : public ErrorMetric {
    R2()
        : ErrorMetric(ErrorType::R2)
    {
    }
};

struct OPERON_EXPORT C2 : public ErrorMetric {
    C2()
        : ErrorMetric(ErrorType::C2)
    {
    }
};

auto OPERON_EXPORT FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target) noexcept
    -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target) noexcept
    -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target,
    Operon::Span<float const> weights) noexcept -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target,
    Operon::Span<double const> weights) noexcept -> std::pair<double, double>;

// Move-only, single-use proof that a caller-owned `buf` span holds a specific Individual's
// valid per-row output. EvaluatorBase alone can mint one after a successful Evaluate call.
class EvaluatedBuffer {
public:
    EvaluatedBuffer(EvaluatedBuffer const&) = delete;
    EvaluatedBuffer(EvaluatedBuffer&& other) noexcept
        : individual_(other.individual_)
        , span_(other.span_)
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
            other.individual_ = nullptr;
            other.span_ = {};
        }
        return *this;
    }
    ~EvaluatedBuffer() = default;

    [[nodiscard]] auto Values(Operon::Individual const& individual, Operon::Span<Operon::Scalar> scratch) const
        noexcept -> Operon::Span<Operon::Scalar>
    {
        ENSURE(Matches(individual, scratch));
        return span_;
    }

private:
    friend struct EvaluatorBase;

    explicit EvaluatedBuffer(Operon::Individual const& individual, Operon::Span<Operon::Scalar> span)
        : individual_(&individual)
        , span_(span)
    {
    }

    [[nodiscard]] auto Matches(Operon::Individual const& individual, Operon::Span<Operon::Scalar> scratch) const
        noexcept -> bool
    {
        return individual_ == &individual && span_.data() == scratch.data() && span_.size() <= scratch.size();
    }

    Operon::Individual const* individual_ {};
    Operon::Span<Operon::Scalar> span_;
};

// Bundles Score's pass-through parameters (needed only by composite/forwarding
// evaluators to delegate to another Score/operator() call) so a leaf evaluator
// that only reads `evaluated` names one unused parameter instead of three.
// Transient: built on the stack immediately before each Score() call and never
// stored, so the reference members below never outlive their referents.
struct ScoreContext {
    Operon::RandomGenerator& Rng; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    Operon::Individual const& Ind; // NOLINT(cppcoreguidelines-avoid-const-or-ref-data-members)
    Operon::Span<Operon::Scalar> Scratch;
};

// `Evaluate(ind, buf)` and `Score(ctx)` are the two hooks subclasses override.
// `operator()` is `final`: it composes them (Evaluate, then Score) and is the
// only place CallCount/ErrMax fallback on Evaluate failure is applied.
struct EvaluatorBase
    : public OperatorBase<Operon::Vector<Operon::Scalar>, Operon::Individual const&, Operon::Span<Operon::Scalar>> {
    using Base = OperatorBase<Operon::Vector<Operon::Scalar>, Operon::Individual const&, Operon::Span<Operon::Scalar>>;
    using ReturnType = Base::ReturnType;

    mutable std::atomic_ulong ResidualEvaluations { 0 }; // NOLINT
    mutable std::atomic_ulong JacobianEvaluations { 0 }; // NOLINT
    mutable std::atomic_ulong CallCount { 0 }; // NOLINT
    mutable std::atomic_ulong CostFunctionTime { 0 }; // NOLINT

    static constexpr size_t DefaultEvaluationBudget = 100'000;

    static auto constexpr ErrMax { std::numeric_limits<Operon::Scalar>::max() };

    // EvaluatorBase has mutable atomic counters and a gsl::not_null member, so
    // the copy/move/default special members are implicitly deleted; it's held
    // by pointer throughout (EvaluatorBase const*) and never copied or moved.
    ~EvaluatorBase() override = default;

    explicit EvaluatorBase(gsl::not_null<Problem const*> problem)
        : problem_(problem)
    {
    }

    // Closes out OperatorBase's pure-virtual 3-arg operator() by composing the
    // two phase hooks below. `final` so no subclass can re-declare operator()
    // and reintroduce the name-hiding hazard this design avoids.
    auto operator()(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const
        -> ReturnType final
    {
        auto evaluated = Evaluate(ind, buf);
        if (!evaluated) {
            ++CallCount;
            return ReturnType { EvaluatorBase::ErrMax };
        }
        return Score({ .Rng = rng, .Ind = ind, .Scratch = buf }, std::move(*evaluated));
    }

    // Fills `buf` (size >= TrainingRange().Size()) with the genotype's raw output and
    // returns it wrapped, or nullopt if this objective isn't value-based (default), or
    // unexpected on evaluation failure.
    [[nodiscard]] virtual auto Evaluate(Operon::Individual const& /*ind*/, Operon::Span<Operon::Scalar> /*buf*/) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError>
    {
        return std::nullopt;
    }

    // `evaluated`, when present, proves `ctx.Scratch` holds `ctx.Ind`'s valid output; a
    // value-based override must ENSURE(evaluated.has_value()) and read
    // evaluated->Values(ctx.Ind, ctx.Scratch), not `ctx.Scratch`. Each override increments CallCount exactly once.
    virtual auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const -> ReturnType = 0;
protected:
    [[nodiscard]] static auto MarkEvaluated(Operon::Individual const& individual, Operon::Span<Operon::Scalar> values)
        -> EvaluatedBuffer
    {
        return EvaluatedBuffer { individual, values };
    }

public:

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

    auto Population() const -> Operon::Span<Individual const> { return population_; }
    auto SetPopulation(Operon::Span<Operon::Individual const> pop) const { population_ = pop; }
    auto GetProblem() const -> Problem const* { return problem_; }
    auto SetProblem(gsl::not_null<Problem const*> problem) { problem_ = problem; }

    void Reset() const
    {
        ResidualEvaluations = 0;
        JacobianEvaluations = 0;
        CallCount = 0;
        CostFunctionTime = 0;
    }

private:
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

// Runs LocalSearch then scores via `evaluator`; non-finite fitness clamps to ErrMax.
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

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> /*evaluated*/) const ->
        typename EvaluatorBase::ReturnType override
    {
        ++this->CallCount;
        return fptr_ ? fptr_(&ctx.Rng, ctx.Ind) : fref_(ctx.Rng, ctx.Ind);
    }

private:
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual const&)> fref_;
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual const&)>
        fptr_; // workaround for pybind11
};

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
        if (skipNonFinite_ && (error_.Type() == ErrorType::R2 || error_.Type() == ErrorType::C2)) {
            throw std::invalid_argument("--skip-nonfinite is only supported for sse, mse, nmse, rmse, and mae");
        }
        if (!std::isfinite(nonFinitePenaltyWeight_) || nonFinitePenaltyWeight_ < 0.0) {
            throw std::invalid_argument("non-finite penalty weight must be finite and non-negative");
        }
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    // Interpreter pass filling `buf` with the genotype's raw TrainingRange() output.
    auto Evaluate(Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> override;

    // Skip-nonfinite scoring or linear-scaling fit-and-apply, then the error metric, over
    // `evaluated`'s values (ctx is otherwise unused here).
    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const ->
        typename EvaluatorBase::ReturnType override;

protected:
    [[nodiscard]] auto UsesLinearScaling() const -> bool { return GetProblem()->LinearScalingEnabled(); }

private:
    gsl::not_null<DTable const*> dtable_;
    ErrorMetric error_;
    // Opt-in: when true, non-finite rows are excluded (ErrorMetric::FiniteSubset) and a
    // variance-scaled penalty is added instead (see SkipNonFiniteScore). Default clamps
    // a non-finite metric result to ErrMax.
    bool skipNonFinite_ { false };
    double nonFinitePenaltyWeight_ { 1.0 };
};

class OPERON_EXPORT MultiEvaluator : public EvaluatorBase {
public:
    // When AggregateType is set, per-evaluator results combine into one scalar instead
    // of concatenating into a multi-objective vector.
    enum class AggregateType : int { Min, Max, Median, Mean, HarmonicMean, Sum };

    explicit MultiEvaluator(Problem const* problem)
        : EvaluatorBase(problem)
    {
    }

    auto Add(EvaluatorBase const* evaluator) { evaluators_.emplace_back(evaluator); }

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

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> /*evaluated*/) const ->
        typename EvaluatorBase::ReturnType override;

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

    auto Evaluators() const { return evaluators_; }

private:
    auto SubEvaluatorObjectiveCount() const -> std::size_t
    {
        return std::transform_reduce(evaluators_.begin(), evaluators_.end(), 0UL, std::plus {},
            [](auto const eval) { return eval->ObjectiveCount(); });
    }

    std::vector<gsl::not_null<EvaluatorBase const*>> evaluators_;
    std::optional<AggregateType> aggregateType_;
};

class OPERON_EXPORT TreePropertyEvaluator : public UserDefinedEvaluator {
public:
    using Property = std::function<Operon::Scalar(Operon::Tree const&)>;

    explicit TreePropertyEvaluator(
        gsl::not_null<Operon::Problem const*> problem, Property property, Operon::Scalar normalizer = 1);
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

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> /*evaluated*/) const ->
        typename EvaluatorBase::ReturnType override;

    auto Prepare(Operon::Span<Operon::Individual const> pop) const -> void override;

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

    // Predicted/target/weight spans over `evaluated`'s values, with linear scaling fit-and-applied
    // in place when the problem enables it. Shared prologue for MDL/FBF/LikelihoodEvaluator::Score,
    // which otherwise duplicate this setup identically before diverging into their own statistic.
    struct ScaledValues {
        Operon::Range TrainingRange;
        Operon::Span<Operon::Scalar> YPred; // evaluated->Values(), scaled in place if Scaling is set
        Operon::Span<Operon::Scalar const> YTrue;
        Operon::Span<Operon::Scalar const> Weights;
        std::optional<LinearScaling> Scaling;
    };

    inline auto PrepareScaledValues(
        Operon::Problem const& problem, ScoreContext ctx, std::optional<EvaluatedBuffer>& evaluated) -> ScaledValues
    {
        auto const trainingRange = problem.TrainingRange();
        ENSURE(evaluated.has_value());
        auto yPred = evaluated->Values(ctx.Ind, ctx.Scratch);
        auto yTrue = problem.TargetValues(trainingRange);
        auto const weights = problem.Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const> {});
        std::optional<LinearScaling> scaling {};
        if (problem.LinearScalingEnabled()) {
            scaling = Operon::FitLinearScaling(yPred, yTrue, weights, problem.LinearScalingOmitsNonFinite());
            scaling->ApplyInPlace(yPred);
        }
        return { .TrainingRange = trainingRange, .YPred = yPred, .YTrue = yTrue, .Weights = weights,
            .Scaling = scaling };
    }
} // namespace detail

template <typename DTable, Concepts::Likelihood Lik>
    requires Concepts::HasFisherMatrix<Lik>
class OPERON_EXPORT MinimumDescriptionLengthEvaluator final : public Evaluator<DTable> {
    // Scores the same fitted linear-scaled model as pareto_front.cpp export and shape certification,
    // closing the previous in-search/exported MDL divergence for the same individual.
    using Base = Evaluator<DTable>;

public:
    explicit MinimumDescriptionLengthEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, SSE {})
    {
    }

    auto Sigma() const { return std::span<Operon::Scalar const> { sigma_ }; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void { sigma_ = std::move(sigma); }

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const ->
        typename EvaluatorBase::ReturnType override
    {
        ++Base::CallCount;

        auto const* dtable = Base::GetDispatchTable();
        auto const* problem = Base::GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const& tree = ctx.Ind.Genotype;
        auto parameters = tree.GetCoefficients();

        auto const p { static_cast<double>(parameters.size()) };

        auto [trainingRange, yPred, yTrue, weights, scaling] = detail::PrepareScaledValues(*problem, ctx, evaluated);

        Operon::Scalar profiledSigma {};
        if (sigma_.empty() && Lik::UsesSigma) {
            profiledSigma = detail::ProfileSigma(yPred, yTrue);
        }

        auto const effectiveSigma = (sigma_.empty() && Lik::UsesSigma)
            ? std::span<Operon::Scalar const> { &profiledSigma, 1 } // profiled
            : std::span<Operon::Scalar const> { sigma_ }; // fixed scalar, per-sample, or empty (Poisson unweighted)

        ++Base::JacobianEvaluations;
        Operon::Interpreter<Operon::Scalar, DTable> const interpreter { dtable, dataset, &tree };
        Eigen::Matrix<Operon::Scalar, -1, -1> jac = interpreter.JacRev(parameters, trainingRange); // jacobian
        if (scaling) {
            jac *= static_cast<Operon::Scalar>(scaling->Scale); // d(a*tree)/d(coeffs) = a * d(tree)/d(coeffs)
        }
        auto fisherMatrix
            = Lik::ComputeFisherMatrix(yPred, { jac.data(), static_cast<std::size_t>(jac.size()) }, effectiveSigma);
        auto fisherDiag = fisherMatrix.diagonal().array();
        ENSURE(fisherDiag.size() == p);

        auto cLikelihood = Lik::ComputeLikelihood(yPred, yTrue, effectiveSigma);
        auto mdl = Operon::MinimumDescriptionLength(tree, parameters, fisherDiag, static_cast<double>(cLikelihood));
        if (!std::isfinite(mdl)) {
            mdl = EvaluatorBase::ErrMax;
        }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(mdl) };
    }

private:
    mutable std::vector<Operon::Scalar> sigma_;
};

template <typename DTable, Concepts::Likelihood Lik>
    requires Concepts::HasFisherMatrix<Lik>
class OPERON_EXPORT FractionalBayesFactorEvaluator final : public Evaluator<DTable> {
    // Scores the same fitted linear-scaled model as pareto_front.cpp export and shape certification,
    // closing the previous in-search/exported FBF divergence for the same individual.
    using Base = Evaluator<DTable>;

public:
    explicit FractionalBayesFactorEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, SSE {})
    {
    }

    auto Sigma() const { return std::span<Operon::Scalar const> { sigma_ }; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void { sigma_ = std::move(sigma); }

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const ->
        typename EvaluatorBase::ReturnType override
    {
        ++Base::CallCount;

        auto const* problem = Base::GetProblem();
        auto const& tree = ctx.Ind.Genotype;

        auto [trainingRange, estimatedValues, targetValues, weights, scaling]
            = detail::PrepareScaledValues(*problem, ctx, evaluated);
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

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const ->
        typename EvaluatorBase::ReturnType override;
};

template <typename DTable> class OPERON_EXPORT AkaikeInformationCriterionEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit AkaikeInformationCriterionEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, MSE {})
    {
    }

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const ->
        typename EvaluatorBase::ReturnType override;
};

template <typename DTable, Concepts::Likelihood Likelihood = GaussianLikelihood<Operon::Scalar>>
    requires(DTable::template SupportsType<typename Likelihood::Scalar>)
class OPERON_EXPORT LikelihoodEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit LikelihoodEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable)
        , sigma_(1, 0.001)
    {
    }

    auto Score(ScoreContext ctx, std::optional<EvaluatedBuffer> evaluated) const ->
        typename EvaluatorBase::ReturnType override
    {
        ++Base::CallCount;

        auto const* problem = Base::Evaluator::GetProblem();
        auto scaled = detail::PrepareScaledValues(*problem, ctx, evaluated);
        auto estimatedValues = scaled.YPred;
        auto targetValues = scaled.YTrue;

        auto lik = Likelihood::ComputeLikelihood(estimatedValues, targetValues, sigma_);
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(lik) };
    }

    auto Sigma() const { return std::span<Operon::Scalar const> { sigma_ }; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void { sigma_ = std::move(sigma); }

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

} // namespace Operon
#endif
