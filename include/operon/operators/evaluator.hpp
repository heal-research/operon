// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_EVALUATOR_HPP
#define OPERON_EVALUATOR_HPP

#include <atomic>
#include <functional>
#include <utility>

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
#include "operon/operon_export.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/likelihood_base.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"

namespace Operon {

enum class ErrorType : int { SSE, MSE, NMSE, RMSE, MAE, R2, C2 };

struct OPERON_EXPORT ErrorMetric {
    using Iterator = Operon::Scalar const*;
    using ProjIterator = ProjectionIterator<Iterator>;

    explicit ErrorMetric(ErrorType type) : type_(type) { }

    auto operator()(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y) const -> double;
    auto operator()(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> double;
    auto operator()(Iterator beg1, Iterator end1, Iterator beg2) const -> double;
    auto operator()(Iterator beg1, Iterator end1, Iterator beg2, Iterator beg3) const -> double;

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

auto OPERON_EXPORT FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target) noexcept -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target) noexcept -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<float const> estimated, Operon::Span<float const> target, Operon::Span<float const> weights) noexcept -> std::pair<double, double>;
auto OPERON_EXPORT FitLeastSquares(Operon::Span<double const> estimated, Operon::Span<double const> target, Operon::Span<double const> weights) noexcept -> std::pair<double, double>;

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
//   - `Evaluate(rng, ind, buf)` below is the ONE hook subclasses override to
//     score an individual. It is NOT named `operator()`, so a subclass
//     overriding it never declares an `operator()` of its own and therefore
//     can never trigger name hiding.
//   - EvaluatorBase itself closes out OperatorBase's pure-virtual
//     `operator()(rng, ind, buf)` with a `final` override that just forwards
//     to `Evaluate` (no subclass can re-override it, so hiding never has a
//     chance to recur below EvaluatorBase), and adds a non-virtual
//     deducing-this `operator()(rng, ind)` facade that allocates a scratch
//     buffer and forwards too. Both call forms the codebase and pyoperon use
//     (`eval(rng, ind, buf)` and `eval(rng, ind)`) keep working unchanged.
struct EvaluatorBase : public OperatorBase<Operon::Vector<Operon::Scalar>, Operon::Individual const&, Operon::Span<Operon::Scalar>>
{
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

    // Closes out OperatorBase's pure-virtual 3-arg operator() by forwarding
    // to the Evaluate hook below. `final` so no subclass can re-declare
    // operator() and reintroduce the name-hiding hazard this design avoids.
    auto operator()(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> ReturnType final {
        return Evaluate(rng, ind, buf);
    }

    // The single hook subclasses override. `buf` is a caller-owned scratch
    // buffer of size >= TrainingRange().Size().
    virtual auto Evaluate(Operon::RandomGenerator& rng, Operon::Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> ReturnType = 0;

    // 2-arg convenience: non-virtual deducing-this facade (can't be virtual -
    // explicit-object members can't be) that allocates a scratch buffer of
    // TrainingRange().Size() and forwards to the 3-arg operator() above.
    // Self deduces to the static type at the call site (including when the
    // call comes through an `EvaluatorBase&`), so this works polymorphically
    // without itself needing to be virtual. Buffer-size contract is on each
    // concrete Evaluate override that actually reads/writes the buffer (they
    // each carry their own ENSURE), not here: UserDefinedEvaluator and
    // DiversityEvaluator legitimately ignore `buf` and accept any size,
    // including the empty span pyoperon passes for UserDefinedEvaluator.
    template<typename Self>
    auto operator()(this Self const& self, Operon::RandomGenerator& rng, Operon::Individual const& ind) -> ReturnType {
        std::vector<Operon::Scalar> buf(self.GetProblem()->TrainingRange().Size());
        return self.Evaluate(rng, ind, buf);
    }

    virtual void Prepare(Operon::Span<Individual const> /*pop*/) const
    {
    }

    virtual auto ObjectiveCount() const -> std::size_t { return 1UL; }

    auto TotalEvaluations() const -> size_t { return ResidualEvaluations + JacobianEvaluations; }

    void SetBudget(size_t value) { budget_ = value; }
    auto Budget() const -> size_t { return budget_; }

    // virtual because more complex evaluators (e.g. MultiEvaluator) might need to calculate it differently
    virtual auto BudgetExhausted() const -> bool { return TotalEvaluations() >= Budget(); }

    virtual auto Stats() const -> std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> {
        return std::tuple{
            ResidualEvaluations.load(),
            JacobianEvaluations.load(),
            CallCount.load(),
            CostFunctionTime.load()
        };
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

class OPERON_EXPORT UserDefinedEvaluator : public EvaluatorBase {
public:
    UserDefinedEvaluator(gsl::not_null<Problem const*> problem, std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual const&)> func)
        : EvaluatorBase(problem)
        , fref_(std::move(func))
    {
    }

    // the func signature taking a pointer to the rng is a workaround for pybind11, since the random generator is non-copyable we have to pass a pointer
    UserDefinedEvaluator(gsl::not_null<Problem const*> problem, std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual const&)> func)
        : EvaluatorBase(problem)
        , fptr_(std::move(func))
    {
    }

    auto
    Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> /*buf*/) const -> typename EvaluatorBase::ReturnType override
    {
        ++this->CallCount;
        return fptr_ ? fptr_(&rng, ind) : fref_(rng, ind);
    }

private:
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual const&)> fref_;
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual const&)> fptr_; // workaround for pybind11
};

template <typename DTable = ScalarDispatch>
class OPERON_EXPORT Evaluator : public EvaluatorBase {
public:
    using TDispatch    = DTable;
    using TInterpreter = Operon::Interpreter<Operon::Scalar, DTable>;

    explicit Evaluator(gsl::not_null<Problem const*> problem, gsl::not_null<DTable const*> dtable, ErrorMetric error = MSE{}, bool linearScaling = true)
        : EvaluatorBase(problem)
        , dtable_(dtable)
        , error_(error)
        , scaling_(linearScaling)
    {
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    auto
    Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;

private:
    gsl::not_null<DTable const*> dtable_;
    ErrorMetric error_;
    bool scaling_{false};
};

class OPERON_EXPORT MultiEvaluator : public EvaluatorBase {
public:
    explicit MultiEvaluator(Problem const* problem)
        : EvaluatorBase(problem)
    {
    }

    auto Add(EvaluatorBase const* evaluator)
    {
        evaluators_.emplace_back(evaluator);
    }

    auto Prepare(Operon::Span<Operon::Individual const> pop) const -> void override
    {
        for (auto const& e : evaluators_) {
            e->Prepare(pop);
        }
    }

    auto ObjectiveCount() const -> std::size_t override
    {
        return std::transform_reduce(evaluators_.begin(), evaluators_.end(), 0UL, std::plus {}, [](auto const eval) { return eval->ObjectiveCount(); });
    }

    auto
    Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override
    {
        EvaluatorBase::ReturnType fit;
        fit.reserve(ind.Size());

        for (auto const& ev: evaluators_) {
            auto f = (*ev)(rng, ind, buf);
            std::copy(f.begin(), f.end(), std::back_inserter(fit));
        }

        return fit;
    }

    auto Stats() const -> std::tuple<std::size_t, std::size_t, std::size_t, std::size_t> final {
        auto resEval{0UL};
        auto jacEval{0UL};
        auto callCnt{0UL};
        auto cfTime{0UL};

        for (auto const& ev: evaluators_) {
            auto [re, je, cc, ct] = ev->Stats();
            resEval += re;
            jacEval += je;
            callCnt += cc;
            cfTime  += ct;
        }

        return std::tuple{resEval + ResidualEvaluations.load(),
            jacEval + JacobianEvaluations.load(),
            callCnt + CallCount.load(),
            cfTime + CostFunctionTime.load()};
    }

    auto BudgetExhausted() const -> bool final {
        auto [re, je, cc, ct] = Stats();
        return re + je >= Budget();
    }

    auto Evaluators() const { return evaluators_; }

private:
    std::vector<gsl::not_null<EvaluatorBase const*>> evaluators_;
};

class OPERON_EXPORT AggregateEvaluator final : public EvaluatorBase {
public:
    enum class AggregateType : int { Min,
        Max,
        Median,
        Mean,
        HarmonicMean,
        Sum };

    explicit AggregateEvaluator(gsl::not_null<EvaluatorBase const*> evaluator)
        : EvaluatorBase(evaluator->GetProblem())
        , evaluator_(evaluator)
    {
    }

    auto SetAggregateType(AggregateType type) { aggtype_ = type; }
    auto GetAggregateType() const { return aggtype_; }

    auto
    Evaluate(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;

private:
    gsl::not_null<EvaluatorBase const*> evaluator_;
    AggregateType aggtype_ { AggregateType::Mean };
};

// a couple of useful user-defined evaluators (mostly to avoid calling lambdas from python)
// TODO: think about a better design
class OPERON_EXPORT LengthEvaluator : public UserDefinedEvaluator {
public:
    explicit LengthEvaluator(gsl::not_null<Operon::Problem const*> problem, size_t maxlength = 1)
        : UserDefinedEvaluator(problem, [maxlength](Operon::RandomGenerator& /*unused*/, Operon::Individual const& ind) {
            return EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(ind.Genotype.Length()) / static_cast<Operon::Scalar>(maxlength) };
        })
    {
    }
};

class OPERON_EXPORT ShapeEvaluator : public UserDefinedEvaluator {
public:
    explicit ShapeEvaluator(Operon::Problem const* problem)
        : UserDefinedEvaluator(problem, [](Operon::RandomGenerator& /*unused*/, Operon::Individual const& ind) {
            return EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(ind.Genotype.VisitationLength()) };
        })
    {
    }
};

class OPERON_EXPORT DiversityEvaluator : public EvaluatorBase {
public:
    explicit DiversityEvaluator(Operon::Problem const* problem, Operon::HashMode hashmode = Operon::HashMode::Strict, std::size_t sampleSize = 100)
        : EvaluatorBase(problem)
        , hashmode_(hashmode)
        , sampleSize_(sampleSize)
    {
    }

    auto
    Evaluate(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;

    auto Prepare(Operon::Span<Operon::Individual const> pop) const -> void override;

private:
    mutable Operon::Map<Operon::Hash, Operon::Vector<Operon::Hash>> divmap_;
    Operon::HashMode hashmode_ { Operon::HashMode::Strict };
    std::size_t sampleSize_ {};
};

// See core/concepts.hpp for why these are asserted here rather than constraining a template.
// LengthEvaluator/ShapeEvaluator aren't asserted separately: they inherit
// UserDefinedEvaluator's Evaluate override without overriding it themselves,
// so UserDefinedEvaluator's assert below already covers them.
static_assert(Concepts::EvaluatorCallable<UserDefinedEvaluator>);
static_assert(Concepts::EvaluatorCallable<Evaluator<ScalarDispatch>>);
static_assert(Concepts::EvaluatorCallable<MultiEvaluator>);
static_assert(Concepts::EvaluatorCallable<AggregateEvaluator>);
static_assert(Concepts::EvaluatorCallable<DiversityEvaluator>);

namespace detail {
    // Profile MLE sigma-hat = sqrt(SSR/n) from residuals (estimated - target),
    // clamped away from zero so a downstream log(sigma^2) or division by
    // sigma can't hit zero. Shared by evaluators that fall back to this
    // estimate when the caller hasn't supplied a sigma of their own (see the
    // `sigma_.empty() && Lik::UsesSigma` gating at each call site).
    inline auto ProfileSigma(Operon::Span<Operon::Scalar const> estimated, Operon::Span<Operon::Scalar const> target) -> Operon::Scalar
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
        return std::max(static_cast<Operon::Scalar>(std::sqrt(ssr / n)),
                         std::numeric_limits<Operon::Scalar>::epsilon());
    }
} // namespace detail

template <typename DTable, Concepts::Likelihood Lik>
requires Concepts::HasFisherMatrix<Lik>
class OPERON_EXPORT MinimumDescriptionLengthEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit MinimumDescriptionLengthEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, SSE{})
    {
    }

    auto Sigma() const { return std::span<Operon::Scalar const>{sigma_}; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void { sigma_ = std::move(sigma); }

    auto Evaluate(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override {
        ++Base::CallCount;

        auto const* dtable = Base::GetDispatchTable();
        auto const* problem = Base::GetProblem();
        auto const* dataset = problem->GetDataset();

        // this call will optimize the tree coefficients and compute the SSE
        auto const& tree = ind.Genotype;
        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, &ind.Genotype};
        auto parameters = tree.GetCoefficients();

        auto const p { static_cast<double>(parameters.size()) };

        auto const trainingRange = problem->TrainingRange();
        ENSURE(buf.size() >= trainingRange.Size());
        // EvaluatorBase::Evaluate's contract permits buf.size() >
        // trainingRange.Size() (a caller-owned scratch buffer sized for
        // reuse across calls), but everything below - the interpreter's
        // fixed-size write, the Jacobian's row count, ComputeFisherMatrix's
        // row-count inference from estimatedValues.size() - is built
        // around exactly trainingRange.Size() rows. Slice once, up front,
        // so every downstream use (including Interpreter::Evaluate's own
        // result buffer, which it silently leaves untouched if given a
        // mismatched size) sees a span sized to match.
        auto estimatedValues = buf.subspan(0, trainingRange.Size());

        ++Base::ResidualEvaluations;
        interpreter.Evaluate(parameters, trainingRange, estimatedValues);

        auto targetValues = problem->TargetValues(trainingRange);
        Operon::Scalar profiledSigma{};
        if (sigma_.empty() && Lik::UsesSigma) {
            profiledSigma = detail::ProfileSigma(estimatedValues, targetValues);
        }
        auto const effectiveSigma = (sigma_.empty() && Lik::UsesSigma)
            ? std::span<Operon::Scalar const>{&profiledSigma, 1}  // profiled
            : std::span<Operon::Scalar const>{sigma_};             // fixed scalar, per-sample, or empty (Poisson unweighted)

        ++Base::JacobianEvaluations;
        Eigen::Matrix<Operon::Scalar, -1, -1> jac = interpreter.JacRev(parameters, trainingRange); // jacobian
        auto fisherMatrix = Lik::ComputeFisherMatrix(estimatedValues, {jac.data(), static_cast<std::size_t>(jac.size())}, effectiveSigma);
        auto fisherDiag   = fisherMatrix.diagonal().array();
        ENSURE(fisherDiag.size() == p);

        auto cLikelihood = Lik::ComputeLikelihood(estimatedValues, targetValues, effectiveSigma);
        auto mdl = Operon::MinimumDescriptionLength(tree, parameters, fisherDiag, static_cast<double>(cLikelihood));
        if (!std::isfinite(mdl)) { mdl = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(mdl) };
    }

private:
    mutable std::vector<Operon::Scalar> sigma_;
};

template <typename DTable, Concepts::Likelihood Lik>
requires Concepts::HasFisherMatrix<Lik>
class OPERON_EXPORT FractionalBayesFactorEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit FractionalBayesFactorEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, SSE{})
    {
    }

    auto Sigma() const { return std::span<Operon::Scalar const>{sigma_}; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void { sigma_ = std::move(sigma); }

    auto Evaluate(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override {
        ++Base::CallCount;

        auto const* dtable  = Base::GetDispatchTable();
        auto const* problem = Base::GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const& tree    = ind.Genotype;

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, &tree};
        auto parameters = tree.GetCoefficients();

        auto const trainingRange = problem->TrainingRange();
        auto const n { static_cast<double>(trainingRange.Size()) };
        ENSURE(buf.size() >= trainingRange.Size());
        // See MinimumDescriptionLengthEvaluator's Evaluate() for why this
        // slice is needed: everything downstream assumes exactly
        // trainingRange.Size() rows, but buf may legitimately be larger.
        auto estimatedValues = buf.subspan(0, trainingRange.Size());

        ++Base::ResidualEvaluations;
        interpreter.Evaluate(parameters, trainingRange, estimatedValues);

        auto targetValues = problem->TargetValues(trainingRange);
        double mlNLL{};
        Operon::Scalar profiledSigma{};
        if (sigma_.empty() && Lik::UsesSigma) { // NLL = 0.5*n*(log(2*pi*sigma^2)+1), clamped to avoid log(0)
            profiledSigma = detail::ProfileSigma(estimatedValues, targetValues);
            auto const s = static_cast<double>(profiledSigma);
            mlNLL = 0.5 * n * (std::log(Operon::Math::Tau * s * s) + 1.0);
        }
        auto const effectiveSigma = (sigma_.empty() && Lik::UsesSigma)
            ? std::span<Operon::Scalar const>{&profiledSigma, 1}  // profiled
            : std::span<Operon::Scalar const>{sigma_};             // fixed scalar, per-sample, or empty (Poisson unweighted)

        // Profiled case: use pre-computed NLL (avoids second O(n) pass inside ComputeLikelihood).
        // Fixed scalar or per-sample: delegate to ComputeLikelihood.
        auto const nll = (sigma_.empty() && Lik::UsesSigma)
            ? mlNLL
            : static_cast<double>(Lik::ComputeLikelihood(estimatedValues, targetValues, effectiveSigma));

        auto fbf = Operon::FractionalBayesFactor(tree, n, nll);
        if (!std::isfinite(fbf)) { fbf = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(fbf) };
    }

private:
    mutable std::vector<Operon::Scalar> sigma_;
};

template <typename DTable>
class OPERON_EXPORT BayesianInformationCriterionEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit BayesianInformationCriterionEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, MSE{})
    {
    }

    auto
    Evaluate(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;
};

template <typename DTable>
class OPERON_EXPORT AkaikeInformationCriterionEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit AkaikeInformationCriterionEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, MSE{})
    {
    }

    auto
    Evaluate(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;
};

template<typename DTable, Concepts::Likelihood Likelihood = GaussianLikelihood<Operon::Scalar>>
requires (DTable::template SupportsType<typename Likelihood::Scalar>)
class OPERON_EXPORT LikelihoodEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

    public:
    explicit LikelihoodEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable), sigma_(1, 0.001)
    {
    }

    auto
    Evaluate(Operon::RandomGenerator& /*rng*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override {
        ++Base::CallCount;

        auto const* dtable  = Base::Evaluator::GetDispatchTable();
        auto const* problem = Base::Evaluator::GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const* tree    = &ind.Genotype;

        // this call will optimize the tree coefficients and compute the SSE
        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, tree};
        auto parameters = tree->GetCoefficients();

        auto const trainingRange = problem->TrainingRange();
        ENSURE(buf.size() >= trainingRange.Size());
        // See MinimumDescriptionLengthEvaluator's Evaluate() for why this
        // slice is needed: everything downstream assumes exactly
        // trainingRange.Size() rows, but buf may legitimately be larger.
        auto estimatedValues = buf.subspan(0, trainingRange.Size());
        ++Base::ResidualEvaluations;
        interpreter.Evaluate(parameters, trainingRange, estimatedValues);

        auto targetValues = problem->TargetValues(trainingRange);

        auto lik = Likelihood::ComputeLikelihood(estimatedValues, targetValues, sigma_);
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(lik) };
    }

    auto Sigma() const { return std::span<Operon::Scalar const>{sigma_}; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const -> void { sigma_ = std::move(sigma); }

private:
    mutable std::vector<Operon::Scalar> sigma_;
};

template<typename DTable>
using GaussianLikelihoodEvaluator = LikelihoodEvaluator<DTable, GaussianLikelihood<Operon::Scalar>>;

template<typename DTable>
using PoissonLikelihoodEvaluator = LikelihoodEvaluator<DTable, PoissonLikelihood<Operon::Scalar>>;

// These three were previously at risk of not satisfying
// Concepts::EvaluatorCallable: each declared only the 3-arg buffered
// operator(), which hid Evaluator<DTable>'s 2-arg overload from unqualified
// lookup and forced a `using Base::operator();` in each. The deducing-this
// facade in EvaluatorBase removed that hazard (subclasses now override a
// differently-named `Evaluate` and never declare `operator()`), but the
// asserts stay as a permanent guard on the call forms the concept requires.
// Asserted here, after their definitions, since they're templates.
static_assert(Concepts::EvaluatorCallable<BayesianInformationCriterionEvaluator<ScalarDispatch>>);
static_assert(Concepts::EvaluatorCallable<AkaikeInformationCriterionEvaluator<ScalarDispatch>>);
static_assert(Concepts::EvaluatorCallable<LikelihoodEvaluator<ScalarDispatch>>);

} // namespace Operon
#endif
