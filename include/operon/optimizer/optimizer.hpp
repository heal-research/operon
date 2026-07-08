// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_OPTIMIZER_HPP
#define OPERON_OPTIMIZER_HPP

#include <gsl/pointers>
#include <lbfgs/solver.hpp>
#include <functional>

#include "operon/error_metrics/sum_of_squared_errors.hpp"

#include "operon/ceres/tiny_solver.h"

#include <unsupported/Eigen/LevenbergMarquardt>

#include "likelihood/gaussian_likelihood.hpp"
#include "likelihood/poisson_likelihood.hpp"
// GaussianLoss / PoissonLoss are defined in the same headers above.
#include "lm_cost_function.hpp"
#include "operon/core/comparison.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "solvers/sgd.hpp"
#if defined(HAVE_ASMJIT)
#include "jit_lm_cost_function.hpp"
#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#endif

namespace Operon {

enum class OptimizerType : int { Tiny, Eigen };

struct OptimizerSummary {
    std::vector<Operon::Scalar> InitialParameters;
    std::vector<Operon::Scalar> FinalParameters;
    Operon::Scalar InitialCost{};
    Operon::Scalar FinalCost{};
    int Iterations{};
    int FunctionEvaluations{};
    int JacobianEvaluations{};
    bool Success{};
};

class OptimizerBase {
gsl::not_null<Problem const*> problem_;
// batch size for loss functions (default = 0 -> use entire data range)
mutable std::size_t batchSize_{0};
mutable std::size_t iterations_{100}; // NOLINT

public:
    explicit OptimizerBase(gsl::not_null<Problem const*> problem)
        : problem_ { problem }
    {
    }

    OptimizerBase(const OptimizerBase&) = default;
    OptimizerBase(OptimizerBase&&) = delete;
    auto operator=(const OptimizerBase&) -> OptimizerBase& = default;
    auto operator=(OptimizerBase&&) -> OptimizerBase& = delete;

    virtual ~OptimizerBase() = default;

    [[nodiscard]] auto GetProblem() const -> Problem const* { return problem_.get(); }
    [[nodiscard]] auto BatchSize() const -> std::size_t { return batchSize_; }
    [[nodiscard]] auto Iterations() const -> std::size_t { return iterations_; }

    auto SetBatchSize(std::size_t batchSize) const { batchSize_ = batchSize; }
    auto SetIterations(std::size_t iterations) const { iterations_ = iterations; }

    [[nodiscard]] virtual auto Optimize(Operon::RandomGenerator& rng, Tree const& tree) const -> OptimizerSummary = 0;
    [[nodiscard]] virtual auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar = 0;
    [[nodiscard]] virtual auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> = 0;
};

namespace detail {
    inline auto CheckSuccess(double initialCost, double finalCost) {
        constexpr auto CHECK_NAN{true};
        return Operon::Less<CHECK_NAN>{}(finalCost, initialCost);
    }
} // namespace detail

template <typename DTable, OptimizerType = OptimizerType::Tiny>
struct LevenbergMarquardtOptimizer : public OptimizerBase {
    explicit LevenbergMarquardtOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase{problem}, dtable_{dtable}
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*unused*/, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range  = problem->TrainingRange();
        auto target = problem->TargetValues();
        auto iterations = this->Iterations();

        auto weights = dataset->Weights().value_or(Operon::Span<Operon::Scalar const>{});

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, &tree};
        Operon::LMCostFunction cf{gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*>{&interpreter}, target, range, weights};
        ceres::TinySolver<decltype(cf)> solver;
        solver.options.max_num_iterations = static_cast<int>(iterations+1);

        auto x0 = tree.GetCoefficients();
        OptimizerSummary summary;
        summary.InitialParameters = x0;
        auto m0 = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size());
        if (!x0.empty()) {
            typename decltype(solver)::Parameters p = m0.cast<typename decltype(cf)::Scalar>();
            solver.Solve(cf, &p);
            m0 = p.template cast<Operon::Scalar>();
        }
        summary.FinalParameters = x0;
        summary.InitialCost = solver.summary.initial_cost;
        summary.FinalCost = solver.summary.final_cost;
        summary.Iterations = solver.summary.iterations;
        summary.FunctionEvaluations = cf.ResidualCalls();
        summary.JacobianEvaluations = cf.JacobianCalls();
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return summary;
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    [[nodiscard]] auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar final
    {
        return GaussianLikelihood<Operon::Scalar>::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final {
        return GaussianLikelihood<Operon::Scalar>::ComputeFisherMatrix(pred, jac, sigma);
    }

    private:
    gsl::not_null<DTable const*> dtable_;
};

template <typename DTable>
struct LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> final : public OptimizerBase {
    explicit LevenbergMarquardtOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase{problem}, dtable_{dtable}
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*unused*/, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range  = problem->TrainingRange();
        auto target = problem->TargetValues();
        auto iterations = this->Iterations();

        auto weights = dataset->Weights().value_or(Operon::Span<Operon::Scalar const>{});

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, &tree};
        Operon::LMCostFunction<Operon::Scalar> cf{&interpreter, target, range, weights};
        Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
        lm.setMaxfev(static_cast<int>(iterations+2));

        auto x0 = tree.GetCoefficients();
        OptimizerSummary summary;
        summary.InitialParameters = x0;
        if (!x0.empty()) {
            Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
            Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;

            // do the minimization loop manually because we want to extract the initial cost
            Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(m);
            summary.InitialCost = summary.FinalCost = lm.fnorm() * lm.fnorm() * 0.5; // get the initial cost after calling minimizeInit()
            if (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
                do {
                    status = lm.minimizeOneStep(m);
                } while (status == Eigen::LevenbergMarquardtSpace::Running);
            }
            m0 = m;
        }
        summary.FinalParameters = x0;
        summary.FinalCost = lm.fnorm() * lm.fnorm() * 0.5;
        summary.Iterations = static_cast<int>(lm.iterations());
        summary.FunctionEvaluations = static_cast<int>(cf.ResidualCalls());
        summary.JacobianEvaluations = static_cast<int>(cf.JacobianCalls());

        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return summary;
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    [[nodiscard]] auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar final
    {
        return GaussianLikelihood<Operon::Scalar>::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final {
        return GaussianLikelihood<Operon::Scalar>::ComputeFisherMatrix(pred, jac, sigma);
    }

    private:
    gsl::not_null<DTable const*> dtable_;
};

template<typename DTable, Concepts::OptimizerLoss LossFunction = GaussianLoss<Operon::Scalar>>
struct LBFGSOptimizer final : public OptimizerBase {
    LBFGSOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase{problem}, dtable_{dtable}
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& rng, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range  = problem->TrainingRange();
        auto target = problem->TargetValues(range);
        auto iterations = this->Iterations();
        auto batchSize = this->BatchSize();
        if (batchSize == 0) { batchSize = range.Size(); }
        auto weights = problem->Weights(range).value_or(Operon::Span<Operon::Scalar const>{});

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, &tree};
        // LossFunction batches internally (SelectBatch), so it needs the
        // whole-dataset target/weights columns (absolute, dataset-row-indexed
        // - the same indexing it hands the interpreter for any sub-range),
        // not the range-local `target`/`weights` above (which line up with
        // `pred` in the single-range `cost` lambda below).
        LossFunction loss{&rng, &interpreter, problem->TargetValues(), range, batchSize, dataset->Weights().value_or(Operon::Span<Operon::Scalar const>{})};

        auto cost = [&](auto const& coeff) {
            auto pred = interpreter.Evaluate(coeff, range);
            // Delegated to LossFunction::Cost (not computed unweighted here
            // directly) so this stays consistent with what operator() actually
            // optimizes. Do NOT assume this line is weighted just because
            // `weights` is passed in - each LossFunction decides for itself
            // whether to apply it (GaussianLoss::Cost: yes; PoissonLoss::Cost:
            // no, see its comment) - otherwise Success could be judged against
            // the wrong objective and CoefficientOptimizer (local_search.cpp)
            // would drop valid weighted gains.
            //
            // TODO: Cost is an SSE surrogate for every LossFunction, not each
            // one's true objective (Poisson::operator() actually optimizes
            // Poisson NLL) - a pre-existing mismatch, unrelated to weighting,
            // that should eventually report the real objective per loss type.
            return LossFunction::Cost(pred, target, weights);
        };

        auto coeff = tree.GetCoefficients();
        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));

        lbfgs::solver solver{loss};
        solver.max_iterations = iterations;
        solver.max_line_search_iterations = iterations;
        auto const f0 = cost(coeff);
        OptimizerSummary summary;
        summary.InitialParameters = coeff;
        summary.InitialCost = f0;

        if (auto res = solver.optimize(x0)) {
            auto xf = res.value();
            std::copy(xf.begin(), xf.end(), coeff.begin());
        }

        summary.FinalParameters = coeff;
        auto const f1 = cost(coeff);
        summary.FinalCost = f1;
        summary.Success = detail::CheckSuccess(f0, f1);
        auto const funEvals = loss.FunctionEvaluations();
        auto const jacEvals = loss.JacobianEvaluations();
        auto const rangeSize = range.Size();
        summary.FunctionEvaluations = static_cast<std::size_t>(static_cast<double>(funEvals + jacEvals) * batchSize / rangeSize);
        summary.JacobianEvaluations = summary.FunctionEvaluations;
        return summary;
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    [[nodiscard]] auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar override
    {
        return LossFunction::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final {
        return LossFunction::ComputeFisherMatrix(pred, jac, sigma);
    }

    private:
    gsl::not_null<DTable const*> dtable_;
};

template<typename DTable, Concepts::OptimizerLoss LossFunction = GaussianLoss<Operon::Scalar>>
struct SGDOptimizer final : public OptimizerBase {
    SGDOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase{problem}
        , dtable_{dtable}
        , update_{std::make_unique<UpdateRule::Constant<Operon::Scalar>>(Operon::Scalar{0.01})}
    { }

    SGDOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem, UpdateRule::LearningRateUpdateRule const& update)
        : OptimizerBase{problem}
        , dtable_{dtable}
        , update_{update.Clone(0)}
    { }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& rng, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range  = problem->TrainingRange();
        auto target = problem->TargetValues(range);
        auto iterations = this->Iterations();
        auto batchSize = this->BatchSize();
        if (batchSize == 0) { batchSize = range.Size(); }
        auto weights = problem->Weights(range).value_or(Operon::Span<Operon::Scalar const>{});

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, &tree};
        // LossFunction batches internally (SelectBatch), so it needs the
        // whole-dataset target/weights columns (absolute, dataset-row-indexed
        // - the same indexing it hands the interpreter for any sub-range),
        // not the range-local `target`/`weights` above (which line up with
        // `pred` in the single-range `cost` lambda below).
        LossFunction loss{&rng, &interpreter, problem->TargetValues(), range, batchSize, dataset->Weights().value_or(Operon::Span<Operon::Scalar const>{})};

        auto cost = [&](auto const& coeff) {
            auto pred = interpreter.Evaluate(coeff, range);
            // Delegated to LossFunction::Cost (not computed unweighted here
            // directly) so this stays consistent with what operator() actually
            // optimizes. Do NOT assume this line is weighted just because
            // `weights` is passed in - each LossFunction decides for itself
            // whether to apply it (GaussianLoss::Cost: yes; PoissonLoss::Cost:
            // no, see its comment) - otherwise Success could be judged against
            // the wrong objective and CoefficientOptimizer (local_search.cpp)
            // would drop valid weighted gains.
            //
            // TODO: Cost is an SSE surrogate for every LossFunction, not each
            // one's true objective (Poisson::operator() actually optimizes
            // Poisson NLL) - a pre-existing mismatch, unrelated to weighting,
            // that should eventually report the real objective per loss type.
            return LossFunction::Cost(pred, target, weights);
        };

        auto coeff = tree.GetCoefficients();
        auto const f0 = cost(coeff);
        OptimizerSummary summary;
        summary.InitialParameters = coeff;
        summary.InitialCost = f0;
        auto rule = update_->Clone(coeff.size());
        SGDSolver<LossFunction> solver(&loss, rule.get());

        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));
        auto x = solver.Optimize(x0, iterations);
        std::copy(x.begin(), x.end(), coeff.begin());
        auto const f1 = cost(coeff);

        summary.FinalParameters = coeff;
        summary.FinalCost = f1;
        summary.Success = detail::CheckSuccess(f0, f1);
        summary.Iterations = solver.Epochs();
        auto const funEvals = loss.FunctionEvaluations();
        auto const jacEvals = loss.JacobianEvaluations();
        auto const rangeSize = range.Size();
        summary.FunctionEvaluations = static_cast<std::size_t>(static_cast<double>(funEvals + jacEvals) * batchSize / rangeSize);
        summary.JacobianEvaluations = summary.FunctionEvaluations;
        return summary;
    }

    [[nodiscard]] auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar override
    {
        return LossFunction::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final {
        return LossFunction::ComputeFisherMatrix(pred, jac, sigma);
    }

    auto SetUpdateRule(std::unique_ptr<UpdateRule::LearningRateUpdateRule const> update) {
        update_ = std::move(update);
    }

    auto UpdateRule() const { return update_.get(); }

    private:
    gsl::not_null<DTable const*> dtable_;
    std::unique_ptr<UpdateRule::LearningRateUpdateRule const> update_{nullptr};
};
#if defined(HAVE_ASMJIT)
// LM optimizer backed by a JitEvaluator for compiled residuals and/or Jacobian.
//
// JacobianOnly=false (default): JIT-compiles both the forward pass (residuals)
//   and the Jacobian; falls back to interpreter when compilation fails.
// JacobianOnly=true: uses the interpreter for residuals; only the Jacobian is
//   JIT-compiled.  Useful when forward-pass compilation overhead exceeds savings.
//
// Pass a JitEvaluator constructed for the same GP run so the code cache is
// shared between fitness evaluation and coefficient optimisation.
template <typename DTable, OptimizerType Type = OptimizerType::Tiny, bool JacobianOnly = false>
struct JitLevenbergMarquardtOptimizer : public OptimizerBase {
    explicit JitLevenbergMarquardtOptimizer(gsl::not_null<DTable const*>           dtable,
                                            gsl::not_null<Problem const*>          problem,
                                            gsl::not_null<JIT::JitEvaluator const*> jitEvaluator)
        : OptimizerBase{problem}
        , dtable_{dtable}
        , jitEval_{jitEvaluator}
    {}

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*rng*/, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const* dtable  = dtable_.get();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const  range   = problem->TrainingRange();
        auto const  target  = problem->TargetValues();
        auto const  iters   = this->Iterations();
        auto const  weights = dataset->Weights().value_or(Operon::Span<Operon::Scalar const>{});

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, &tree};

        JIT::CompileMeta const* meta = jitEval_->GetOrCompileJacobian(tree);
        if (!JacobianOnly && (!meta || !meta->fn)) { meta = jitEval_->GetOrCompile(tree); }

        OptimizerSummary summary;
        auto x0 = tree.GetCoefficients();
        summary.InitialParameters = x0;

        bool const hasFn    = meta && meta->fn;
        bool const hasJacFn = meta && meta->jacFn;
        // In JacobianOnly mode only enter the JIT path when the Jacobian was actually compiled;
        // falling through to JitLMCostFunction with a null jacFn wastes allocation for nothing.
        bool const useJitCf = !x0.empty() && (hasFn || (JacobianOnly && hasJacFn));

        if (!useJitCf) {
            // Pure interpreter fallback — no JIT at all.
            Operon::LMCostFunction cf{
                gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*>{&interpreter},
                target, range, weights};
            Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
            lm.setMaxfev(static_cast<int>(iters + 2));
            if (!x0.empty()) {
                Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
                Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;
                Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(m);
                summary.InitialCost = summary.FinalCost = lm.fnorm() * lm.fnorm() * 0.5;
                if (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
                    do { status = lm.minimizeOneStep(m); }
                    while (status == Eigen::LevenbergMarquardtSpace::Running);
                }
                m0 = m;
            }
            summary.FinalParameters       = x0;
            summary.FinalCost             = lm.fnorm() * lm.fnorm() * 0.5;
            summary.Iterations            = static_cast<int>(lm.iterations());
            summary.FunctionEvaluations   = static_cast<int>(cf.ResidualCalls());
            summary.JacobianEvaluations   = static_cast<int>(cf.JacobianCalls());
            summary.Success               = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
            return summary;
        }

        // Column pointer arrays are rebuilt from the tree (VarOrder is re-derivable;
        // the fixed Zobrist hash makes it structurally unique per entry).
        // Both fn and jacFn use the same variable ordering, so one colPtrs suffices.
        auto const varOrder = JIT::VarOrder(tree);
        auto const start    = static_cast<std::ptrdiff_t>(range.Start());

        std::vector<float const*> colPtrs;
        JIT::EvalFn evalFn{};
        if (hasFn) {
            evalFn = meta->fn;
            colPtrs.resize(varOrder.size());
            for (std::size_t i = 0; i < varOrder.size(); ++i) {
                colPtrs[i] = dataset->GetPaddedValues(varOrder[i]) + start;
            }
        }

        std::vector<float const*> jacColPtrs;
        JIT::EvalJacFn jacFn{};
        if (meta && meta->jacFn) {
            jacFn = meta->jacFn;
            jacColPtrs.resize(varOrder.size());
            for (std::size_t i = 0; i < varOrder.size(); ++i) {
                jacColPtrs[i] = dataset->GetPaddedValues(varOrder[i]) + start;
            }
        }

        Operon::JitLMCostFunction cf{
            gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*>{&interpreter},
            evalFn,
            std::move(colPtrs),
            target, range,
            jacFn,
            std::move(jacColPtrs),
            meta->nVars,
            meta->nConsts,
            weights};

        Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
        lm.setMaxfev(static_cast<int>(iters + 2));

        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
        Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;

        Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(m);
        summary.InitialCost = summary.FinalCost = lm.fnorm() * lm.fnorm() * 0.5;
        if (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
            do { status = lm.minimizeOneStep(m); }
            while (status == Eigen::LevenbergMarquardtSpace::Running);
        }
        m0 = m;

        summary.FinalParameters     = x0;
        summary.FinalCost           = lm.fnorm() * lm.fnorm() * 0.5;
        summary.Iterations          = static_cast<int>(lm.iterations());
        summary.FunctionEvaluations = static_cast<int>(cf.ResidualCalls());
        summary.JacobianEvaluations = static_cast<int>(cf.JacobianCalls());
        summary.Success             = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return summary;
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    [[nodiscard]] auto ComputeLikelihood(Operon::Span<Operon::Scalar const> x, Operon::Span<Operon::Scalar const> y, Operon::Span<Operon::Scalar const> w) const -> Operon::Scalar final
    {
        return GaussianLikelihood<Operon::Scalar>::ComputeLikelihood(x, y, w);
    }

    [[nodiscard]] auto ComputeFisherMatrix(Operon::Span<Operon::Scalar const> pred, Operon::Span<Operon::Scalar const> jac, Operon::Span<Operon::Scalar const> sigma) const -> Eigen::Matrix<Operon::Scalar, -1, -1> final
    {
        return GaussianLikelihood<Operon::Scalar>::ComputeFisherMatrix(pred, jac, sigma);
    }

private:
    gsl::not_null<DTable const*>            dtable_;
    gsl::not_null<JIT::JitEvaluator const*> jitEval_;
};
#endif // HAVE_ASMJIT

} // namespace Operon
#endif
