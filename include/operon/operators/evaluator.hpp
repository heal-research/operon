// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_EVALUATOR_HPP
#define OPERON_EVALUATOR_HPP

#include <atomic>
#include <functional>
#include <utility>

#include "operon/collections/projection.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/operator.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/types.hpp"
#include "operon/interpreter/interpreter.hpp"
#include "operon/operon_export.hpp"
#include "operon/optimizer/likelihood/gaussian_likelihood.hpp"
#include "operon/optimizer/likelihood/likelihood_base.hpp"
#include "operon/optimizer/likelihood/poisson_likelihood.hpp"

namespace Operon {

enum class ErrorType : int { SSE, MSE, NMSE, RMSE, MAE, R2, C2 };

struct OPERON_EXPORT ErrorMetric {
    using Iterator = Operon::Span<Operon::Scalar const>::iterator;
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

using E1 = OperatorBase<Operon::Vector<Operon::Scalar>, Individual const&>;
using E2 = OperatorBase<Operon::Vector<Operon::Scalar>, Individual const&, Operon::Span<Operon::Scalar>>;

struct EvaluatorBase : public E1, E2
{
    using ReturnType = E1::ReturnType;

    mutable std::atomic_ulong ResidualEvaluations { 0 }; // NOLINT
    mutable std::atomic_ulong JacobianEvaluations { 0 }; // NOLINT
    mutable std::atomic_ulong CallCount { 0 }; // NOLINT
    mutable std::atomic_ulong CostFunctionTime { 0 }; // NOLINT

    static constexpr size_t DefaultEvaluationBudget = 100'000;

    static auto constexpr ErrMax { std::numeric_limits<Operon::Scalar>::max() };

    auto operator()(Operon::RandomGenerator& /*unused*/, Operon::Individual const& /*unused*/, Operon::Span<Operon::Scalar> /*unused*/) const -> ReturnType override = 0;

    auto operator()(Operon::RandomGenerator& rng, Operon::Individual const& ind) const -> ReturnType override = 0;

    explicit EvaluatorBase(gsl::not_null<Problem const*> problem)
        : problem_(problem)
    {
    }

    template<typename Derived>
    static auto Evaluate(Derived const* self, Operon::RandomGenerator& rng, Operon::Individual const& ind, std::span<Operon::Scalar> buf) {
        ENSURE(buf.size() >= self->GetProblem()->TrainingRange.Size());
        return std::invoke(*self, rng, ind, buf);
    }

    template<typename Derived>
    static auto Evaluate(Derived const* self, Operon::RandomGenerator& rng, Operon::Individual const& ind) {
        std::vector<Operon::Scalar> buf(self->GetProblem()->TrainingRange().Size());
        return std::invoke(*self, rng, ind, buf);
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
    operator()(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> /*args*/) const -> typename EvaluatorBase::ReturnType override
    {
        ++this->CallCount;
        return fptr_ ? fptr_(&rng, ind) : fref_(rng, ind);
    }

    auto operator()(Operon::RandomGenerator& rng, Individual const& ind) const -> typename EvaluatorBase::ReturnType override {
        return EvaluatorBase::Evaluate(this, rng, ind);
    }

private:
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator&, Operon::Individual const&)> fref_;
    std::function<typename EvaluatorBase::ReturnType(Operon::RandomGenerator*, Operon::Individual const&)> fptr_; // workaround for pybind11
};

template <typename DTable = DefaultDispatch>
class OPERON_EXPORT Evaluator : public EvaluatorBase {
public:
    using TDispatch    = DTable;
    using TInterpreter = Operon::Interpreter<Operon::Scalar, DTable>;

    explicit Evaluator(gsl::not_null<Problem const*> problem, gsl::not_null<DTable const*> dtable, ErrorMetric error = MSE{}, bool linearScaling = true, std::vector<Operon::Scalar> weights = {})
        : EvaluatorBase(problem)
        , dtable_(dtable)
        , error_(error)
        , scaling_(linearScaling)
        , weights_(std::move(weights))
    {
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    auto
    operator()(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;

    auto
    operator()(Operon::RandomGenerator& rng, Individual const& ind) const -> typename EvaluatorBase::ReturnType override;

private:
    gsl::not_null<DTable const*> dtable_;
    ErrorMetric error_;
    bool scaling_{false};
    mutable std::vector<Operon::Scalar> weights_;
};

class MultiEvaluator : public EvaluatorBase {
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
    operator()(Operon::RandomGenerator& rng, Individual const& ind) const -> typename EvaluatorBase::ReturnType override
    {
        return EvaluatorBase::Evaluate(this, rng, ind);
    }

    auto
    operator()(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override
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
    operator()(Operon::RandomGenerator& rng, Individual const& ind) const -> typename EvaluatorBase::ReturnType override;

    auto
    operator()(Operon::RandomGenerator& rng, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;

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
    operator()(Operon::RandomGenerator& /*random*/, Individual const& ind) const -> typename EvaluatorBase::ReturnType override;

    auto
    operator()(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;

    auto Prepare(Operon::Span<Operon::Individual const> pop) const -> void override;

private:
    mutable Operon::Map<Operon::Hash, Operon::Vector<Operon::Hash>> divmap_;
    Operon::HashMode hashmode_ { Operon::HashMode::Strict };
    std::size_t sampleSize_ {};
};

template <typename DTable, typename Lik>
class OPERON_EXPORT MinimumDescriptionLengthEvaluator final : public Evaluator<DTable> {
    using Base = Evaluator<DTable>;

public:
    explicit MinimumDescriptionLengthEvaluator(Operon::Problem const* problem, DTable const* dtable)
        : Base(problem, dtable, SSE{}), sigma_(1, 0.001)
    {
    }

    auto Sigma() const { return std::span<Operon::Scalar const>{sigma_}; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const { sigma_ = std::move(sigma); }

    auto operator()(Operon::RandomGenerator& rng, Operon::Individual const& ind) const -> typename EvaluatorBase::ReturnType override {
        return EvaluatorBase::Evaluate(this, rng, ind);
    }

    auto operator()(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override {
        ++Base::CallCount;

        auto const* dtable = Base::GetDispatchTable();
        auto const* problem = Base::GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const& nodes = ind.Genotype.Nodes();

        // this call will optimize the tree coefficients and compute the SSE
        auto const& tree = ind.Genotype;
        Operon::Interpreter<Operon::Scalar, DefaultDispatch> interpreter{dtable, dataset, &ind.Genotype};
        auto parameters = tree.GetCoefficients();

        auto const p { static_cast<double>(parameters.size()) };

        auto const trainingRange = problem->TrainingRange();
        ENSURE(buf.size() >= trainingRange.Size());

        ++Base::ResidualEvaluations;
        interpreter.Evaluate(parameters, trainingRange, buf);


        // codelength of the complexity
        // count number of unique functions
        // - count weight * variable as three nodes
        // - compute complexity c of the remaining numerical values
        //   (that are not part of the coefficients that are optimized)
        Operon::Set<Operon::Hash> uniqueFunctions; // to count the number of unique functions
        auto k{0.0}; // number of nodes
        auto cComplexity { 0.0 };

        // codelength of the parameters
        ++Base::JacobianEvaluations;
        Eigen::Matrix<Operon::Scalar, -1, -1> j = interpreter.JacRev(parameters, trainingRange); // jacobian
        auto estimatedValues = buf;
        auto fisherMatrix = Lik::ComputeFisherMatrix(estimatedValues, {j.data(), static_cast<std::size_t>(j.size())}, sigma_);
        auto fisherDiag   = fisherMatrix.diagonal().array();
        ENSURE(fisherDiag.size() == p);

        auto cParameters { 0.0 };
        auto constexpr eps = std::numeric_limits<Operon::Scalar>::epsilon(); // machine epsilon for zero comparison

        for (auto i = 0, j = 0; i < std::ssize(nodes); ++i) {
            auto const& n = nodes[i];

            // count the number of nodes and the number of unique operators
            k += n.IsVariable() ? 3 : 1;
            uniqueFunctions.insert(n.HashValue);

            if (n.Optimize) {
                // this branch computes the description length of the parameters to be optimized
                auto const di = std::sqrt(12 / fisherDiag(j));
                auto const ci = std::abs(parameters[j]);

                if (!(std::isfinite(ci) && std::isfinite(di)) || ci / di < 1) {
                    //ind.Genotype[i].Optimize = false;
                    //auto const v = ind.Genotype[i].Value;
                    //ind.Genotype[i].Value = 0;
                    //auto fit = (*this)(rng, ind, buf);
                    //ind.Genotype[i].Optimize = true;
                    //ind.Genotype[i].Value = v;
                    //return fit;
                } else {
                    cParameters += 0.5 * std::log(fisherDiag(j)) + std::log(ci);
                }
                ++j;
            } else {
                // this branch computes the description length of the remaining tree structure
                if (std::abs(n.Value) < eps) { continue; }
                cComplexity += std::log(std::abs(n.Value));
            }
        }

        auto q { static_cast<double>(uniqueFunctions.size()) };
        if (q > 0) { cComplexity += static_cast<double>(k) * std::log(q); }

        cParameters -= p/2 * std::log(3);

        auto targetValues = problem->TargetValues(trainingRange);
        auto cLikelihood  = Lik::ComputeLikelihood(estimatedValues, targetValues, sigma_);
        auto mdl = cComplexity + cParameters + cLikelihood;
        if (!std::isfinite(mdl)) { mdl = EvaluatorBase::ErrMax; }
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(mdl) };
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
    operator()(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;
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
    operator()(Operon::RandomGenerator& /*random*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override;
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
    operator()(Operon::RandomGenerator& /*rng*/, Individual const& ind, Operon::Span<Operon::Scalar> buf) const -> typename EvaluatorBase::ReturnType override {
        ++Base::CallCount;

        auto const* dtable  = Base::Evaluator::GetDispatchTable();
        auto const* problem = Base::Evaluator::GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const* tree    = &ind.Genotype;

        // this call will optimize the tree coefficients and compute the SSE
        Operon::Interpreter<Operon::Scalar, DefaultDispatch> interpreter{dtable, dataset, tree};
        auto parameters = tree->GetCoefficients();

        std::vector<Operon::Scalar> buffer;
        auto const trainingRange = problem->TrainingRange();
        if (buf.size() < trainingRange.Size()) {
            buffer.resize(trainingRange.Size());
            buf = Operon::Span<Operon::Scalar>(buffer);
        }
        ++Base::ResidualEvaluations;
        interpreter.Evaluate(parameters, trainingRange, buf);

        auto estimatedValues = buf;
        auto targetValues    = problem->TargetValues(trainingRange);

        auto lik = Likelihood::ComputeLikelihood(estimatedValues, targetValues, sigma_);
        return typename EvaluatorBase::ReturnType { static_cast<Operon::Scalar>(lik) };
    }

    auto Sigma() const { return std::span<Operon::Scalar const>{sigma_}; }
    auto SetSigma(std::vector<Operon::Scalar> sigma) const { sigma_ = std::move(sigma); }

private:
    mutable std::vector<Operon::Scalar> sigma_;
};

template<typename DTable>
using GaussianLikelihoodEvaluator = LikelihoodEvaluator<DTable, GaussianLikelihood<Operon::Scalar>>;

template<typename DTable>
using PoissonLikelihoodEvaluator = LikelihoodEvaluator<DTable, PoissonLikelihood<Operon::Scalar>>;

} // namespace Operon
#endif
