// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_OPTIMIZER_HPP
#define OPERON_OPTIMIZER_HPP

#include <functional>
#include <gsl/pointers>
#include <lbfgs/solver.hpp>
#include <tl/expected.hpp>
#include <variant>

#include "operon/error_metrics/sum_of_squared_errors.hpp"

#include "operon/ceres/tiny_solver.h"

#include <unsupported/Eigen/LevenbergMarquardt>

#include "operon/optimizer/detail/gradient_solver_adapter.hpp"
#include "operon/optimizer/gaussian_gradient_cost.hpp"
#include "operon/optimizer/poisson_gradient_cost.hpp"
#include "operon/optimizer/interpreter_least_squares.hpp"
#include "operon/optimizer/least_squares_lm_adapter.hpp"
#include "operon/optimizer/lm_weights.hpp"
#include "operon/core/comparison.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "solvers/sgd.hpp"
#if defined(HAVE_ASMJIT)
#include "operon/optimizer/jit_least_squares.hpp"
#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#endif

namespace Operon {

enum class OptimizerType : int { Tiny,
    Eigen };

// Fields every Optimize() call always produces. FitResult, FitFailure,
// FitEvaluationError, and FitConfigurationError all carry them, so callers
// that need e.g. FunctionEvaluations regardless of outcome use Diagnostics().
struct FitDiagnostics {
    std::vector<Operon::Scalar> InitialParameters;
    std::vector<Operon::Scalar> FinalParameters;
    Operon::Scalar InitialCost {};
    Operon::Scalar FinalCost {};
    int Iterations {};
    int FunctionEvaluations {};
    int JacobianEvaluations {};
};

struct FitResult : FitDiagnostics {}; // FinalCost improved on InitialCost
struct FitFailure : FitDiagnostics {}; // valid fit that did not improve (incl. non-finite cost)
struct FitEvaluationError : FitDiagnostics {
    GradientError Error;
};
struct FitConfigurationError : FitDiagnostics {
    LMWeightError Error;
};

using FitError = std::variant<FitFailure, FitEvaluationError, FitConfigurationError>;
using FitOutcome = tl::expected<FitResult, FitError>;

[[nodiscard]] inline auto Diagnostics(FitOutcome const& outcome) -> FitDiagnostics const&
{
    if (outcome) {
        return *outcome;
    }
    return std::visit([](auto const& error) -> FitDiagnostics const& { return error; }, outcome.error());
}

[[nodiscard]] inline auto EvaluationError(FitOutcome const& outcome) -> FitEvaluationError const*
{
    return outcome ? nullptr : std::get_if<FitEvaluationError>(&outcome.error());
}

[[nodiscard]] inline auto ConfigurationError(FitOutcome const& outcome) -> FitConfigurationError const*
{
    return outcome ? nullptr : std::get_if<FitConfigurationError>(&outcome.error());
}

class OptimizerBase {
    gsl::not_null<Problem const*> problem_;
    // batch size for loss functions (default = 0 -> use entire data range)
    mutable std::size_t batchSize_ { 0 };
    mutable std::size_t iterations_ { 100 }; // NOLINT

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

    [[nodiscard]] virtual auto Optimize(Operon::RandomGenerator& rng, Tree const& tree) const -> FitOutcome = 0;
};

namespace detail {
    inline auto CheckSuccess(double initialCost, double finalCost)
    {
        constexpr auto CHECK_NAN { true };
        return Operon::Less<CHECK_NAN> {}(finalCost, initialCost);
    }

    // Replaces the near-identical summary-assembly tail block that used to
    // be repeated at the end of every Optimize() override: each override
    // builds one FitDiagnostics via aggregate init, then returns
    // MakeFitOutcome(std::move(diag)) as its last line.
    inline auto MakeFitOutcome(FitDiagnostics diag) -> FitOutcome
    {
        if (CheckSuccess(diag.InitialCost, diag.FinalCost)) {
            return FitResult { std::move(diag) };
        }
        return tl::unexpected(FitFailure { std::move(diag) });
    }
    inline auto MakeFitEvaluationError(GradientError error, FitDiagnostics diag) -> FitOutcome
    {
        FitEvaluationError failure;
        static_cast<FitDiagnostics&>(failure) = std::move(diag);
        failure.Error = std::move(error);
        return tl::unexpected(FitError { std::move(failure) });
    }

    // Convenience for the (still common) case of a raw interpreter failure
    // with no separate numerical cost wrapping it.
    inline auto MakeFitEvaluationError(InterpreterError error, FitDiagnostics diag) -> FitOutcome
    {
        return MakeFitEvaluationError(GradientError { .Code = GradientErrorCode::EvaluationFailure, .Cause = std::move(error) }, std::move(diag));
    }

    inline auto MakeFitConfigurationError(LMWeightError error, FitDiagnostics diag) -> FitOutcome
    {
        FitConfigurationError failure;
        static_cast<FitDiagnostics&>(failure) = std::move(diag);
        failure.Error = error;
        return tl::unexpected(FitError { std::move(failure) });
    }
} // namespace detail

template <typename DTable, OptimizerType = OptimizerType::Tiny>
struct LevenbergMarquardtOptimizer : public OptimizerBase {
    explicit LevenbergMarquardtOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase { problem }
        , dtable_ { dtable }
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*unused*/, Operon::Tree const& tree) const -> FitOutcome final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range = problem->TrainingRange();
        auto target = problem->TargetValues();
        auto iterations = this->Iterations();

        auto const localWeights = problem->Weights(range).value_or(Operon::Span<Operon::Scalar const> {});
        auto x0 = tree.GetCoefficients();
        FitDiagnostics diag;
        diag.InitialParameters = x0;
        auto validWeights = TryValidateLMWeights(localWeights, range.Size());
        if (!validWeights) {
            diag.FinalParameters = x0;
            return detail::MakeFitConfigurationError(validWeights.error(), std::move(diag));
        }

        Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, dataset, &tree };
        Operon::InterpreterLeastSquaresCostFunction costFn { gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*> { &interpreter }, target, range };
        Operon::LeastSquaresLMAdapter<> cf { &costFn, localWeights };
        ceres::TinySolver<decltype(cf)> solver;
        auto m0 = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size());
        if (!x0.empty()) {
            // max_num_accepted_steps counts accepted LM steps only, matching
            // Eigen::LevenbergMarquardt's iterations() semantics (see the
            // Eigen-backend LevenbergMarquardtOptimizer below) - unlike this
            // class's own max_num_iterations, which (unmodified) bounds total
            // attempts, accepted or rejected. max_num_iterations is still set,
            // as a MINPACK-convention-scaled safety net on rejected-retry
            // attempts, mirroring maxfev's role for the Eigen backend.
            solver.options.max_num_accepted_steps = static_cast<int>(iterations);
            solver.options.max_num_iterations = static_cast<int>(iterations) * (static_cast<int>(x0.size()) + 1);
            typename decltype(solver)::ParameterVector p = m0.cast<typename decltype(cf)::Scalar>();
            solver.Solve(cf, &p);
            m0 = p.template cast<Operon::Scalar>();
        }
        diag.FinalParameters = x0;
        diag.InitialCost = solver.summary.initial_cost;
        diag.FinalCost = solver.summary.final_cost;
        diag.Iterations = solver.summary.iterations;
        diag.FunctionEvaluations = cf.ResidualCalls();
        diag.JacobianEvaluations = cf.JacobianCalls();
        if (auto const& error = cf.Error(); error) {
            return detail::MakeFitEvaluationError(detail::ToGradientError(*error), std::move(diag));
        }
        return detail::MakeFitOutcome(std::move(diag));
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

private:
    gsl::not_null<DTable const*> dtable_;
};

template <typename DTable>
struct LevenbergMarquardtOptimizer<DTable, OptimizerType::Eigen> final : public OptimizerBase {
    explicit LevenbergMarquardtOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase { problem }
        , dtable_ { dtable }
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*unused*/, Operon::Tree const& tree) const -> FitOutcome final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range = problem->TrainingRange();
        auto target = problem->TargetValues();
        auto iterations = this->Iterations();

        auto const localWeights = problem->Weights(range).value_or(Operon::Span<Operon::Scalar const> {});
        auto x0 = tree.GetCoefficients();
        FitDiagnostics diag;
        diag.InitialParameters = x0;
        auto validWeights = TryValidateLMWeights(localWeights, range.Size());
        if (!validWeights) {
            diag.FinalParameters = x0;
            return detail::MakeFitConfigurationError(validWeights.error(), std::move(diag));
        }

        Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, dataset, &tree };
        Operon::InterpreterLeastSquaresCostFunction costFn { gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*> { &interpreter }, target, range };
        Operon::LeastSquaresLMAdapter<> cf { &costFn, localWeights };
        Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
        if (!x0.empty()) {
            // `iterations` counts accepted LM steps (lm.iterations()), matching the
            // Tiny/ceres variant's max_num_iterations - it is not itself a function-
            // evaluation budget. maxfev is still needed as a bound on rejected
            // trust-region retries within/across those steps (Eigen's own default,
            // 400 regardless of iterations, is enough per individual to exhaust the
            // CLI's overall --evaluations budget across a full GP run), scaled by
            // parameter count using MINPACK's own convention (100*(n+1) for its
            // "no fixed iteration count" default) so the ceiling grows with problem
            // size instead of being a fixed constant.
            auto const maxfev = static_cast<Eigen::Index>(iterations) * (static_cast<Eigen::Index>(x0.size()) + 1);
            lm.setMaxfev(std::max<Eigen::Index>(maxfev, 1));

            Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
            Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;

            // do the minimization loop manually because we want to extract the initial cost
            Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(m);
            diag.InitialCost = diag.FinalCost = lm.fnorm() * lm.fnorm() * Operon::Scalar{0.5}; // get the initial cost after calling minimizeInit()
            if (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
                do {
                    status = lm.minimizeOneStep(m);
                } while (status == Eigen::LevenbergMarquardtSpace::Running
                    && lm.iterations() < static_cast<Eigen::Index>(iterations));
            }
            m0 = m;
        }
        diag.FinalParameters = x0;
        diag.FinalCost = lm.fnorm() * lm.fnorm() * Operon::Scalar{0.5};
        diag.Iterations = static_cast<int>(lm.iterations());
        diag.FunctionEvaluations = static_cast<int>(cf.ResidualCalls());
        diag.JacobianEvaluations = static_cast<int>(cf.JacobianCalls());
        if (auto const& error = cf.Error(); error) {
            return detail::MakeFitEvaluationError(detail::ToGradientError(*error), std::move(diag));
        }
        return detail::MakeFitOutcome(std::move(diag));
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

private:
    gsl::not_null<DTable const*> dtable_;
};

namespace detail {
    // Ordinary dataset sample weights apply to Gaussian gradient costs as
    // numerical WLS weights. They are never forwarded as Poisson exposure --
    // a caller that genuinely intends exposure constructs
    // PoissonGradientCostFunction directly with an explicit exposure span.
    template <typename Cost>
    struct GradientCostSampleWeights {
        static auto Get(Operon::Dataset const* /*dataset*/) -> Operon::Span<Operon::Scalar const> { return {}; }
    };

    template <typename T>
    struct GradientCostSampleWeights<GaussianGradientCostFunction<T>> {
        static auto Get(Operon::Dataset const* dataset) -> Operon::Span<Operon::Scalar const>
        {
            return dataset->Weights().value_or(Operon::Span<Operon::Scalar const> {});
        }
    };

    [[nodiscard]] inline auto ScaleBatchEvaluations(std::size_t evaluations, std::size_t batchSize, std::size_t rangeSize) -> int
    {
        auto const effectiveBatchSize = batchSize == 0 ? rangeSize : batchSize;
        if (evaluations == 0 || effectiveBatchSize == 0 || rangeSize == 0) {
            return 0;
        }
        auto const scaled = static_cast<double>(evaluations) * static_cast<double>(effectiveBatchSize)
            / static_cast<double>(rangeSize);
        return std::max(1, static_cast<int>(scaled));
    }
} // namespace detail

template <typename DTable, Concepts::GradientCost Cost = GaussianGradientCostFunction<Operon::Scalar>>
struct LBFGSOptimizer final : public OptimizerBase {
    LBFGSOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase { problem }
        , dtable_ { dtable }
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& rng, Operon::Tree const& tree) const -> FitOutcome final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range = problem->TrainingRange();
        auto iterations = this->Iterations();
        auto batchSize = this->BatchSize();

        Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, dataset, &tree };
        // Cost batches internally (SelectBatch), so it needs the whole-dataset
        // target column (absolute, dataset-row-indexed), not a slice pre-cut
        // to range.
        Cost cost { &interpreter, problem->TargetValues(), range, &rng, batchSize, detail::GradientCostSampleWeights<Cost>::Get(dataset) };
        Operon::detail::GradientSolverAdapter<Cost> bridge { &cost };

        auto coeff = tree.GetCoefficients();
        FitDiagnostics diag;
        diag.InitialParameters = coeff;

        std::vector<Operon::Scalar> gradScratch(coeff.size());
        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> gradMap(gradScratch.data(), std::ssize(gradScratch));
        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));
        diag.InitialCost = bridge(x0, gradMap);
        if (auto const& error = bridge.Error(); error) {
            diag.FinalParameters = coeff;
            return detail::MakeFitEvaluationError(*error, std::move(diag));
        }

        lbfgs::solver solver { bridge };
        solver.max_iterations = iterations;
        solver.max_line_search_iterations = iterations;
        auto result = solver.optimize(x0);
        if (result) {
            auto xf = result.value();
            std::copy(xf.begin(), xf.end(), coeff.begin());
        }

        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const> xFinal(coeff.data(), std::ssize(coeff));
        diag.FinalCost = bridge(xFinal, gradMap);
        diag.FinalParameters = coeff;
        diag.FunctionEvaluations = detail::ScaleBatchEvaluations(cost.FunctionEvaluations(), batchSize, range.Size());
        diag.JacobianEvaluations = detail::ScaleBatchEvaluations(cost.JacobianEvaluations(), batchSize, range.Size());
        if (auto const& error = bridge.Error(); error) {
            return detail::MakeFitEvaluationError(*error, std::move(diag));
        }
        return detail::MakeFitOutcome(std::move(diag));
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

private:
    gsl::not_null<DTable const*> dtable_;
};

template <typename DTable, Concepts::GradientCost Cost = GaussianGradientCostFunction<Operon::Scalar>>
struct SGDOptimizer final : public OptimizerBase {
    SGDOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase { problem }
        , dtable_ { dtable }
        , update_ { std::make_unique<UpdateRule::Constant<Operon::Scalar>>(Operon::Scalar { 0.01 }) }
    {
    }

    SGDOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem, UpdateRule::LearningRateUpdateRule const& update)
        : OptimizerBase { problem }
        , dtable_ { dtable }
        , update_ { update.Clone(0) }
    {
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& rng, Operon::Tree const& tree) const -> FitOutcome final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range = problem->TrainingRange();
        auto iterations = this->Iterations();
        auto batchSize = this->BatchSize();

        Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, dataset, &tree };
        // Cost batches internally (SelectBatch), so it needs the whole-dataset
        // target column (absolute, dataset-row-indexed), not a slice pre-cut
        // to range.
        Cost cost { &interpreter, problem->TargetValues(), range, &rng, batchSize, detail::GradientCostSampleWeights<Cost>::Get(dataset) };
        Operon::detail::GradientSolverAdapter<Cost> bridge { &cost };

        auto coeff = tree.GetCoefficients();
        FitDiagnostics diag;
        diag.InitialParameters = coeff;

        Eigen::Array<Operon::Scalar, -1, 1> gradScratch(coeff.size());
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));
        diag.InitialCost = bridge(x0, gradScratch);
        if (auto const& error = bridge.Error(); error) {
            diag.FinalParameters = coeff;
            return detail::MakeFitEvaluationError(*error, std::move(diag));
        }

        auto rule = update_->Clone(coeff.size());
        SGDSolver<decltype(bridge)> solver(&bridge, rule.get());
        auto x = solver.Optimize(x0, iterations);
        std::copy(x.begin(), x.end(), coeff.begin());

        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> xFinal(coeff.data(), std::ssize(coeff));
        diag.FinalCost = bridge(xFinal, gradScratch);
        diag.FinalParameters = coeff;
        diag.Iterations = solver.Epochs();
        diag.FunctionEvaluations = detail::ScaleBatchEvaluations(cost.FunctionEvaluations(), batchSize, range.Size());
        diag.JacobianEvaluations = detail::ScaleBatchEvaluations(cost.JacobianEvaluations(), batchSize, range.Size());
        if (auto const& error = bridge.Error(); error) {
            return detail::MakeFitEvaluationError(*error, std::move(diag));
        }
        return detail::MakeFitOutcome(std::move(diag));
    }

    auto SetUpdateRule(std::unique_ptr<UpdateRule::LearningRateUpdateRule const> update)
    {
        update_ = std::move(update);
    }

    auto UpdateRule() const { return update_.get(); }

private:
    gsl::not_null<DTable const*> dtable_;
    std::unique_ptr<UpdateRule::LearningRateUpdateRule const> update_ { nullptr };
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
    explicit JitLevenbergMarquardtOptimizer(gsl::not_null<DTable const*> dtable,
        gsl::not_null<Problem const*> problem,
        gsl::not_null<JIT::JitEvaluator const*> jitEvaluator)
        : OptimizerBase { problem }
        , dtable_ { dtable }
        , jitEval_ { jitEvaluator }
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*rng*/, Operon::Tree const& tree) const -> FitOutcome final
    {
        auto const* dtable = dtable_.get();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto const range = problem->TrainingRange();
        auto const target = problem->TargetValues();
        auto const iters = this->Iterations();

        Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, dataset, &tree };
        FitDiagnostics diag;
        auto x0 = tree.GetCoefficients();
        diag.InitialParameters = x0;
        auto const localWeights = problem->Weights(range).value_or(Operon::Span<Operon::Scalar const> {});
        auto validWeights = TryValidateLMWeights(localWeights, range.Size());
        if (!validWeights) {
            diag.FinalParameters = x0;
            return detail::MakeFitConfigurationError(validWeights.error(), std::move(diag));
        }
        auto bound = interpreter.BindTree(range);
        if (!bound) {
            diag.FinalParameters = x0;
            return detail::MakeFitEvaluationError(std::move(bound.error()), std::move(diag));
        }

        JIT::CompileMeta const* meta = jitEval_->GetOrCompileJacobian(tree);
        if (!JacobianOnly && (!meta || !meta->fn)) {
            meta = jitEval_->GetOrCompile(tree);
        }

        bool const hasFn = meta && meta->fn;
        bool const hasJacFn = meta && meta->jacFn;
        // In JacobianOnly mode only enter the JIT path when the Jacobian was actually compiled;
        // falling through to JitLeastSquaresCostFunction with a null jacFn wastes allocation for nothing.
        bool const useJitCf = !x0.empty() && (hasFn || (JacobianOnly && hasJacFn));

        if (!useJitCf) {
            // Pure interpreter fallback — no JIT at all.
            Operon::InterpreterLeastSquaresCostFunction costFn { gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*> { &interpreter }, target, range };
            Operon::LeastSquaresLMAdapter<> cf { &costFn, localWeights };
            Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
            if (!x0.empty()) {
                lm.setMaxfev(std::max<Eigen::Index>(
                    static_cast<Eigen::Index>(iters) * (static_cast<Eigen::Index>(x0.size()) + 1), 1));
                Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
                Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;
                Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(m);
                diag.InitialCost = diag.FinalCost = lm.fnorm() * lm.fnorm() * 0.5;
                if (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
                    do {
                        status = lm.minimizeOneStep(m);
                    } while (status == Eigen::LevenbergMarquardtSpace::Running
                        && lm.iterations() < static_cast<Eigen::Index>(iters));
                }
                m0 = m;
            }
            diag.FinalParameters = x0;
            diag.FinalCost = lm.fnorm() * lm.fnorm() * 0.5;
            diag.Iterations = static_cast<int>(lm.iterations());
            diag.FunctionEvaluations = static_cast<int>(cf.ResidualCalls());
            diag.JacobianEvaluations = static_cast<int>(cf.JacobianCalls());
            if (auto const& error = cf.Error(); error) {
                return detail::MakeFitEvaluationError(detail::ToGradientError(*error), std::move(diag));
            }
            return detail::MakeFitOutcome(std::move(diag));
        }

        // Column pointer arrays are rebuilt from the tree (VarOrder is re-derivable;
        // the fixed Zobrist hash makes it structurally unique per entry).
        // Both fn and jacFn use the same variable ordering, so one colPtrs suffices.
        auto const varOrder = JIT::VarOrder(tree);
        auto const start = static_cast<std::ptrdiff_t>(range.Start());

        std::vector<float const*> colPtrs;
        JIT::EvalFn evalFn {};
        if (hasFn) {
            evalFn = meta->fn;
            colPtrs.resize(varOrder.size());
            for (std::size_t i = 0; i < varOrder.size(); ++i) {
                colPtrs[i] = dataset->GetPaddedValues(varOrder[i]) + start;
            }
        }

        std::vector<float const*> jacColPtrs;
        JIT::EvalJacFn jacFn {};
        if (meta && meta->jacFn) {
            jacFn = meta->jacFn;
            jacColPtrs.resize(varOrder.size());
            for (std::size_t i = 0; i < varOrder.size(); ++i) {
                jacColPtrs[i] = dataset->GetPaddedValues(varOrder[i]) + start;
            }
        }

        Operon::JitLeastSquaresCostFunction costFn {
            gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*> { &interpreter },
            evalFn,
            std::move(colPtrs),
            target, range,
            jacFn,
            std::move(jacColPtrs),
            meta->nVars,
            meta->nConsts
        };
        Operon::LeastSquaresLMAdapter<> cf { &costFn, localWeights };

        Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
        lm.setMaxfev(std::max<Eigen::Index>(
            static_cast<Eigen::Index>(iters) * (static_cast<Eigen::Index>(x0.size()) + 1), 1));

        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
        Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;

        Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(m);
        diag.InitialCost = diag.FinalCost = lm.fnorm() * lm.fnorm() * 0.5;
        if (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
            do {
                status = lm.minimizeOneStep(m);
            } while (status == Eigen::LevenbergMarquardtSpace::Running
                && lm.iterations() < static_cast<Eigen::Index>(iters));
        }
        m0 = m;

        diag.FinalParameters = x0;
        diag.FinalCost = lm.fnorm() * lm.fnorm() * 0.5;
        diag.Iterations = static_cast<int>(lm.iterations());
        diag.FunctionEvaluations = static_cast<int>(cf.ResidualCalls());
        diag.JacobianEvaluations = static_cast<int>(cf.JacobianCalls());
        if (auto const& error = cf.Error(); error) {
            return detail::MakeFitEvaluationError(detail::ToGradientError(*error), std::move(diag));
        }
        return detail::MakeFitOutcome(std::move(diag));
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

private:
    gsl::not_null<DTable const*> dtable_;
    gsl::not_null<JIT::JitEvaluator const*> jitEval_;
};
#endif // HAVE_ASMJIT

} // namespace Operon
#endif
