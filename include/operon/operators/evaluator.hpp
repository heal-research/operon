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

// Move-only, single-use proof that a caller-owned scratch span (an Evaluate/Score
// `buf` parameter) holds a specific Individual's valid per-row model output. The
// only way to obtain one is a successful EvaluatorBase::Evaluate call.
// Score's own `buf` parameter is unaffected and
// stays an ordinary, unprotected scratch span (see MultiEvaluator/ShapeViolationEvaluator,
// which use it purely as reusable memory and never read `evaluated`).
class EvaluatedBuffer {
public:
    EvaluatedBuffer(EvaluatedBuffer const&) = delete;
    EvaluatedBuffer(EvaluatedBuffer&& other) noexcept
        : span_(other.span_)
    {
        other.span_ = {};
    }
    auto operator=(EvaluatedBuffer const&) -> EvaluatedBuffer& = delete;
    auto operator=(EvaluatedBuffer&& other) noexcept -> EvaluatedBuffer&
    {
        if (this != &other) {
            span_ = other.span_;
            other.span_ = {};
        }
        return *this;
    }
    ~EvaluatedBuffer() = default;

    [[nodiscard]] auto Values() const noexcept -> Operon::Span<Operon::Scalar> { return span_; }
private:
    friend auto MarkEvaluated(Operon::Span<Operon::Scalar>) -> EvaluatedBuffer;
    explicit EvaluatedBuffer(Operon::Span<Operon::Scalar> span)
        : span_(span)
    {
    }
    Operon::Span<Operon::Scalar> span_;
};

// The only way to construct an EvaluatedBuffer: an explicit, visible, auditable call
// an Evaluate() override makes once it has actually filled `values` with this
// Individual's per-row model output. Re-tags the SAME memory -- no copy, no
// reallocation.
[[nodiscard]] inline auto MarkEvaluated(Operon::Span<Operon::Scalar> values) -> EvaluatedBuffer
{
    return EvaluatedBuffer { values };
}

// EvaluatorBase inherits OperatorBase once, like every other operator family
// (CreatorBase, MutatorBase, CrossoverBase, ...) - the buffered 3-arg shape is
// the canonical one. The previous design instead inherited OperatorBase TWICE
// (E1 for the unbuffered call, E2 for the buffered call) to get two
// `operator()` overloads directly from the base; any subclass overriding just
// one of them (the common case) hid the other via C++ name hiding, forcing
// `using Base::operator();` in three subclasses plus a redundant 2-arg
// `operator() override { return Evaluate(rng, ind); }` boilerplate in every
// class. The fix keeps the single-inheritance shape uniform with every other
// family and splits the two roles `operator()` was playing:
//   - `Evaluate(ind, buf)` and `Score(rng, ind, buf)` below are the TWO hooks
//     subclasses override. They are NOT named `operator()`, so a subclass
//     overriding them never declares an `operator()` of its own and therefore
//     can never trigger name hiding.
//   - EvaluatorBase itself closes out OperatorBase's pure-virtual
//     `operator()(rng, ind, buf)` with a `final` override composing the two
//     phases (no subclass can re-override it, so hiding never has a chance to
//     recur below EvaluatorBase), and adds a non-virtual deducing-this
//     `operator()(rng, ind)` facade that allocates a scratch buffer and
//     forwards too. Both call forms the codebase and pyoperon use
//     (`eval(rng, ind, buf)` and `eval(rng, ind)`) keep working unchanged.
//
// Verb usage follows the nomenclature taxonomy: `evaluate` = compute the
// genotype's per-row model outputs; `score` = assign fitness. A wrapper that
// needs the outputs for its own purposes (e.g. ShapeConstrainedEvaluator's
// linear-scaling fit) calls the phases separately instead of `operator()`,
// so the outputs are computed once and shared.
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
        return Score(rng, ind, buf, std::move(*evaluated));
    }

    // Phase 1: evaluate the genotype's raw (pre-scaling) TrainingRange() output into `buf`
    // (a caller-owned scratch buffer of size >= TrainingRange().Size(); only the first
    // TrainingRange().Size() entries are written). Returns:
    //   EvaluatedBuffer wrapping `buf`'s valid prefix -> value-based, `buf` holds the values
    //   nullopt        -> this objective is not value-based (`buf` untouched) - default
    //   unexpected     -> evaluation attempted and failed (missing variable/primitive)
    // Value-based scorers MUST leave `buf` exactly as their own scoring pass would
    // produce it, so a caller that evaluates once and scores separately gets the
    // same numbers as a caller that goes through operator().
    [[nodiscard]] virtual auto Evaluate(Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError>
    {
        return std::nullopt;
    }

    // Phase 2: score. `buf` is the same ordinary caller-owned scratch span Evaluate
    // received -- no claims, safe to reuse for any purpose (MultiEvaluator forwards it
    // to each sub-evaluator's own operator(); ShapeViolationEvaluator uses it as
    // disposable FitLinearScaling scratch). `evaluated`, when present, additionally
    // PROVES `buf`'s content is this Individual's valid per-row output -- a value-based
    // override MUST assert `evaluated.has_value()` before reading it (ENSURE; a caller
    // that skipped or misordered Evaluate gets an immediate, loud failure instead of a
    // silently wrong fitness) and read `evaluated->Values()`, not `buf`, for the values.
    // Each implementation increments CallCount exactly once - via its own line
    // or via a base-class Score it delegates to. (operator() increments it on Evaluate failure).
    virtual auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> ReturnType
        = 0;

    // 2-arg convenience: non-virtual deducing-this facade (can't be virtual -
    // explicit-object members can't be) that allocates a scratch buffer of
    // TrainingRange().Size() and forwards to the 3-arg operator() above.
    // Self deduces to the static type at the call site (including when the
    // call comes through an `EvaluatorBase&`), so this works polymorphically
    // without itself needing to be virtual. Buffer-size contract is on each
    // concrete phase override that actually reads/writes the buffer (they
    // each carry their own ENSURE), not here: UserDefinedEvaluator and
    // DiversityEvaluator legitimately ignore `buf` and accept any size,
    // including the empty span pyoperon passes for UserDefinedEvaluator.
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

// Optionally applies local search (coefficient optimization) to `ind`'s
// genotype with probability `pLocal`. If local search ran and the update is
// non-Lamarckian, returns the original coefficients so the caller can evaluate
// the optimized genotype first, then restore inherited coefficients. Does not
// evaluate `ind`'s fitness - split out from ScoreIndividual so a caller that
// needs to run local search over a whole population before any of it is
// scored (e.g. so Prepare() on an evaluator that snapshots the population,
// such as DiversityEvaluator, sees post-optimization genotypes) can do so
// without duplicating this logic.
OPERON_EXPORT auto LocalSearch(Operon::RandomGenerator& random, Operon::Individual& ind,
    Operon::EvaluatorBase const& evaluator, Operon::CoefficientOptimizer const* coeffOptimizer, double pLocal,
    double pLamarck) -> std::optional<std::vector<Operon::Scalar>>;

// Optionally applies local search (coefficient optimization) to `ind`'s
// genotype with probability `pLocal`, then scores it via `evaluator`. Non-
// finite fitness values are clamped to EvaluatorBase::ErrMax either way.
//
// Shared by offspring generation (OffspringGeneratorBase::Generate) and
// initial-population scoring (GeneticProgrammingAlgorithm::Run,
// NSGA2::Run) so both receive identical local-search treatment - passing
// pLocal=0 (or a null coeffOptimizer) degenerates to a plain evaluate.
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

    auto Score(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> /*buf*/,
        std::optional<EvaluatedBuffer> /*evaluated*/) const -> typename EvaluatorBase::ReturnType override
    {
        ++this->CallCount;
        return fptr_ ? fptr_(&rng, ind) : fref_(rng, ind);
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

    // Phase 1: interpreter pass filling `buf` with the genotype's raw TrainingRange() output.
    auto Evaluate(Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const
        -> tl::expected<std::optional<EvaluatedBuffer>, InterpreterError> override;

    // Phase 2: skip-nonfinite scoring or linear-scaling fit-and-apply, then the error
    // metric, over phase 1's values (`evaluated->Values()`). `evaluated` must be present
    // (a preceding successful Evaluate); its values are read, not `buf` (`buf` is only
    // this class's own scratch parameter and is unused here).
    auto Score(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override;

protected:
    [[nodiscard]] auto UsesLinearScaling() const -> bool { return GetProblem()->LinearScalingEnabled(); }

private:
    gsl::not_null<DTable const*> dtable_;
    ErrorMetric error_;
    // Opt-in. When true: non-finite rows excluded via ErrorMetric::FiniteSubset
    // (SSE/MSE/NMSE/RMSE/MAE). fit += nonFinitePenaltyWeight_ * nonfinite
    // fraction, scaled by target variance for the non-normalized metrics
    // (SSE/MSE/RMSE/MAE are unit-dependent; NMSE already divides by target
    // variance, so it isn't scaled again) -- see SkipNonFiniteScore. This
    // keeps a single default meaningful regardless of the metric or the
    // dataset's units: at nonFinitePenaltyWeight_ == 1.0, an individual that
    // is 100% non-finite is penalized by roughly one target-variance's worth
    // of error, the same order of magnitude as a naive constant-mean
    // predictor's MSE.
    // Default (false): non-finite metric result clamps fit to ErrMax.
    bool skipNonFinite_ { false };
    double nonFinitePenaltyWeight_ { 1.0 };
};

class OPERON_EXPORT MultiEvaluator : public EvaluatorBase {
public:
    // When AggregateType is set (see SetAggregateType), the per-evaluator
    // results are combined into a single scalar (e.g. so several objectives
    // can be optimized as one aggregate) instead of being concatenated into
    // a multi-objective vector.
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

    auto Score(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> /*evaluated*/) const -> typename EvaluatorBase::ReturnType override;

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

    auto Score(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> /*evaluated*/) const -> typename EvaluatorBase::ReturnType override;

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
    // Profile MLE sigma-hat = sqrt(SSR/n) from residuals (estimated - target),
    // clamped away from zero so a downstream log(sigma^2) or division by
    // sigma can't hit zero. Shared by evaluators that fall back to this
    // estimate when the caller hasn't supplied a sigma of their own (see the
    // `sigma_.empty() && Lik::UsesSigma` gating at each call site).
    inline auto ProfileSigma(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target)
        -> Operon::Scalar
    {
        // Bounded by the shorter of the two spans, not just estimated's -
        // callers only guarantee estimated.size() >= target.size() (e.g. a
        // reused scratch buffer sized to a training range but possibly
        // larger; see EvaluatorBase::Evaluate's ENSURE), not equality, so
        // indexing target[i] up to estimated.size() alone would read past
        // target's end whenever the buffer is oversized.
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

    auto Score(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> /*buf*/,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override
    {
        ++Base::CallCount;

        auto const* dtable = Base::GetDispatchTable();
        auto const* problem = Base::GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const& tree = ind.Genotype;
        auto parameters = tree.GetCoefficients();

        auto const p { static_cast<double>(parameters.size()) };

        auto const trainingRange = problem->TrainingRange();
        ENSURE(evaluated.has_value());
        auto estimatedValues = evaluated->Values();
        auto targetValues = problem->TargetValues(trainingRange);
        auto const weights = problem->Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const> {});
        auto const scaling = problem->LinearScalingEnabled()
            ? std::optional { Operon::FitLinearScaling(
                  estimatedValues, targetValues, weights, problem->LinearScalingOmitsNonFinite()) }
            : std::nullopt;
        if (scaling) {
            scaling->ApplyInPlace(estimatedValues);
        }

        Operon::Scalar profiledSigma {};
        if (sigma_.empty() && Lik::UsesSigma) {
            profiledSigma = detail::ProfileSigma(estimatedValues, targetValues);
        }
        auto const effectiveSigma = (sigma_.empty() && Lik::UsesSigma)
            ? std::span<Operon::Scalar const> { &profiledSigma, 1 } // profiled
            : std::span<Operon::Scalar const> { sigma_ }; // fixed scalar, per-sample, or empty (Poisson unweighted)

        ++Base::JacobianEvaluations;
        Operon::Interpreter<Operon::Scalar, DTable> const interpreter { dtable, dataset, &ind.Genotype };
        Eigen::Matrix<Operon::Scalar, -1, -1> jac = interpreter.JacRev(parameters, trainingRange); // jacobian
        if (scaling) {
            jac *= static_cast<Operon::Scalar>(scaling->Scale); // d(a*tree)/d(coeffs) = a * d(tree)/d(coeffs)
        }
        auto fisherMatrix = Lik::ComputeFisherMatrix(
            estimatedValues, { jac.data(), static_cast<std::size_t>(jac.size()) }, effectiveSigma);
        auto fisherDiag = fisherMatrix.diagonal().array();
        ENSURE(fisherDiag.size() == p);

        auto cLikelihood = Lik::ComputeLikelihood(estimatedValues, targetValues, effectiveSigma);
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

    auto Score(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> /*buf*/,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override
    {
        ++Base::CallCount;

        auto const* problem = Base::GetProblem();
        auto const& tree = ind.Genotype;

        auto const trainingRange = problem->TrainingRange();
        auto const n { static_cast<double>(trainingRange.Size()) };
        ENSURE(evaluated.has_value());
        auto estimatedValues = evaluated->Values();

        // Scaling refit from phase 1's values - see MinimumDescriptionLengthEvaluator::Score.
        auto targetValues = problem->TargetValues(trainingRange);
        auto const weights = problem->Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const> {});
        auto const scaling = problem->LinearScalingEnabled()
            ? std::optional { Operon::FitLinearScaling(
                  estimatedValues, targetValues, weights, problem->LinearScalingOmitsNonFinite()) }
            : std::nullopt;
        if (scaling) {
            scaling->ApplyInPlace(estimatedValues);
        }

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

    auto Score(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override;
};

template <typename DTable> class OPERON_EXPORT AkaikeInformationCriterionEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit AkaikeInformationCriterionEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, MSE {})
    {
    }

    auto Score(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override;
};

template <typename DTable, Concepts::Likelihood Likelihood = GaussianLikelihood<Operon::Scalar>>
    requires(DTable::template SupportsType<typename Likelihood::Scalar>)
class OPERON_EXPORT LikelihoodEvaluator final : public Evaluator<DTable> {
    // Scores the same fitted linear-scaled model as pareto_front.cpp export and shape certification,
    // closing the previous in-search/exported likelihood divergence for the same individual.
    using Base = Evaluator<DTable>;

public:
    explicit LikelihoodEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable)
        , sigma_(1, 0.001)
    {
    }

    auto Score(Operon::RandomGenerator& /*rng*/, Individual const& ind, Operon::Span<Operon::Scalar> /*buf*/,
        std::optional<EvaluatedBuffer> evaluated) const -> typename EvaluatorBase::ReturnType override
    {
        ++Base::CallCount;

        auto const* problem = Base::Evaluator::GetProblem();
        auto const* tree = &ind.Genotype;

        auto const trainingRange = problem->TrainingRange();
        ENSURE(evaluated.has_value());
        // Phase 1 (inherited Evaluator<DTable>::Evaluate) already filled the memory
        // `evaluated` proves is valid, exactly trainingRange.Size() rows.
        auto estimatedValues = evaluated->Values();

        // Scaling refit from phase 1's values - see MinimumDescriptionLengthEvaluator::Score.
        auto targetValues = problem->TargetValues(trainingRange);
        auto const weights = problem->Weights(trainingRange).value_or(Operon::Span<Operon::Scalar const> {});
        auto const scaling = problem->LinearScalingEnabled()
            ? std::optional { Operon::FitLinearScaling(
                  estimatedValues, targetValues, weights, problem->LinearScalingOmitsNonFinite()) }
            : std::nullopt;
        if (scaling) {
            scaling->ApplyInPlace(estimatedValues);
        }

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
