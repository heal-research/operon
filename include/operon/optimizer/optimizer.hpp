// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_OPTIMIZER_HPP
#define OPERON_OPTIMIZER_HPP

#include <algorithm>
#include <functional>
#include <gsl/pointers>
#include <lbfgs/solver.hpp>
#include <limits>
#include <tl/expected.hpp>
#include <variant>
#include <vector>

#include "operon/error_metrics/sum_of_squared_errors.hpp"

#include "operon/ceres/tiny_solver.h"

#include <unsupported/Eigen/LevenbergMarquardt>

#include "operon/core/comparison.hpp"
#include "operon/core/dispatch.hpp"
#include "operon/core/problem.hpp"
#include "operon/optimizer/detail/gradient_solver_adapter.hpp"
#include "operon/optimizer/fit_outcome.hpp"
#include "operon/optimizer/gaussian_gradient_cost.hpp"
#include "operon/optimizer/interpreter_gradient_cost.hpp"
#include "operon/optimizer/interpreter_least_squares.hpp"
#include "operon/optimizer/least_squares_fit.hpp"
#include "operon/optimizer/least_squares_lm_adapter.hpp"
#include "operon/optimizer/poisson_gradient_cost.hpp"
#include "solvers/sgd.hpp"
#if defined(HAVE_ASMJIT)
#include "operon/interpreter/backend/jit/jit_evaluator.hpp"
#include "operon/optimizer/jit_least_squares.hpp"
#endif

namespace Operon {

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

// Levenberg-Marquardt on the tree's coefficients over the problem's training
// range. Type selects the backend (Tiny by default; Eigen on request); both
// run through detail::RunLeastSquares, the driver behind Operon::FitLeastSquares.
// Optimize() builds all per-call state locally, so concurrent calls on a shared
// optimizer are safe as long as nothing calls SetIterations concurrently.
template <typename DTable, OptimizerType Type = OptimizerType::Tiny>
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
        FitDiagnostics diag;
        diag.InitialParameters = tree.GetCoefficients();
        auto validWeights = ValidateWeights(localWeights, range.Size());
        if (!validWeights) {
            diag.FinalParameters = diag.InitialParameters;
            return detail::MakeFitConfigurationError(validWeights.error(), std::move(diag));
        }

        Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, dataset, &tree };
        Operon::InterpreterLeastSquaresCostFunction costFn {
            gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*> { &interpreter }, target, range
        };
        Operon::LeastSquaresLMAdapter<> cf { &costFn, localWeights, true };
        return detail::RunLeastSquares<Type>(cf, iterations, std::move(diag));
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

private:
    gsl::not_null<DTable const*> dtable_;
};

namespace detail {
    // The dataset's sample weights reach a gradient cost only when the cost
    // declares Cost::UsesDatasetWeights (Gaussian: numerical WLS weights).
    // Otherwise the cost gets an empty span and the dataset weights are
    // neither validated nor forwarded -- notably never as Poisson exposure; a
    // caller that genuinely intends exposure constructs
    // PoissonGradientCostFunction directly with an explicit exposure span.
    template <Concepts::InterpreterGradientCost Cost>
    [[nodiscard]] auto CostDatasetWeights(Operon::Dataset const* dataset) -> Operon::Span<Operon::Scalar const>
    {
        if constexpr (Cost::UsesDatasetWeights) {
            return dataset->Weights().value_or(Operon::Span<Operon::Scalar const> {});
        } else {
            return {};
        }
    }

    // Validates the in-range slice of a whole-dataset-column sample-weight
    // span (rows outside the training range are never read and may hold
    // placeholder values). Row in the error is relative to the range start,
    // like the LM path's range-local weights.
    [[nodiscard]] inline auto ValidateRangeWeights(Operon::Span<Operon::Scalar const> column, Operon::Range range)
        -> tl::expected<void, WeightError>
    {
        if (column.empty()) {
            return {};
        }
        ENSURE(range.Start() + range.Size() <= column.size());
        return ValidateWeights(column.subspan(range.Start(), range.Size()), range.Size());
    }

    [[nodiscard]] inline auto ScaleBatchEvaluations(
        std::size_t evaluations, std::size_t batchSize, std::size_t rangeSize) -> int
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

template <typename DTable, Concepts::InterpreterGradientCost Cost = GaussianGradientCostFunction>
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

        auto const sampleWeights = detail::CostDatasetWeights<Cost>(dataset);
        if (auto validWeights = detail::ValidateRangeWeights(sampleWeights, range); !validWeights) {
            FitDiagnostics diag;
            diag.InitialParameters = tree.GetCoefficients();
            diag.FinalParameters = diag.InitialParameters;
            return detail::MakeFitConfigurationError(validWeights.error(), std::move(diag));
        }

        Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, dataset, &tree };
        // Cost batches internally (SelectBatch), so it needs the whole-dataset
        // target column (absolute, dataset-row-indexed), not a slice pre-cut
        // to range.
        Cost cost { &interpreter, problem->TargetValues(), range, &rng, batchSize, sampleWeights };
        Cost endpointCost { &interpreter, problem->TargetValues(), range, nullptr, 0, sampleWeights };
        Operon::detail::GradientSolverAdapter<Cost> bridge { &cost };
        Operon::detail::GradientSolverAdapter<Cost> endpointBridge { &endpointCost };

        auto coeff = tree.GetCoefficients();
        FitDiagnostics diag;
        diag.InitialParameters = coeff;

        std::vector<Operon::Scalar> gradScratch(coeff.size());
        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> gradMap(gradScratch.data(), std::ssize(gradScratch));
        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));
        diag.InitialCost = endpointBridge(x0, gradMap);
        if (auto const& error = endpointBridge.Error(); error) {
            diag.FinalParameters = coeff;
            return detail::MakeFitEvaluationError(*error, std::move(diag));
        }
        if (coeff.empty()) {
            auto const cost = diag.InitialCost;
            return detail::MakeFitOutcome(detail::ZeroParameterDiagnostics(std::move(diag), cost, 1));
        }

        lbfgs::solver solver { bridge };
        solver.max_iterations = detail::SaturatingCast<int>(iterations);
        solver.max_line_search_iterations = detail::SaturatingCast<int>(iterations);
        // lbfgs::solver reverts to the last accepted iterate and still
        // succeeds when a line-search trial is non-finite or fails, so a
        // solver-facing bridge error alone is advisory. optimize() returns an
        // error only when the solve itself fails.
        auto result = solver.optimize(x0);
        if (result) {
            auto xf = result.value();
            std::copy(xf.begin(), xf.end(), coeff.begin());
        }
        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const> xFinal(coeff.data(), std::ssize(coeff));
        diag.FinalCost = endpointBridge(xFinal, gradMap);
        diag.FinalParameters = coeff;
        // lbfgs::solver does not report an iteration count (solver_status::iterations
        // is never populated), so Iterations stays 0 for this optimizer.
        diag.FunctionEvaluations = detail::ScaleBatchEvaluations(cost.FunctionEvaluations(), batchSize, range.Size());
        diag.JacobianEvaluations = detail::ScaleBatchEvaluations(cost.JacobianEvaluations(), batchSize, range.Size());
        // The endpoint error describes the returned point and takes precedence.
        if (auto const& error = endpointBridge.Error(); error) {
            return detail::MakeFitEvaluationError(*error, std::move(diag));
        }
        if (auto const& error = bridge.Error(); !result && error) {
            return detail::MakeFitEvaluationError(*error, std::move(diag));
        }
        return detail::MakeFitOutcome(std::move(diag));
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

private:
    gsl::not_null<DTable const*> dtable_;
};

template <typename DTable, Concepts::InterpreterGradientCost Cost = GaussianGradientCostFunction>
struct SGDOptimizer final : public OptimizerBase {
    SGDOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : OptimizerBase { problem }
        , dtable_ { dtable }
        , update_ { std::make_unique<UpdateRule::Constant<Operon::Scalar>>(Operon::Scalar { 0.01 }) }
    {
    }

    SGDOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem,
        UpdateRule::LearningRateUpdateRule const& update)
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

        auto const sampleWeights = detail::CostDatasetWeights<Cost>(dataset);
        if (auto validWeights = detail::ValidateRangeWeights(sampleWeights, range); !validWeights) {
            FitDiagnostics diag;
            diag.InitialParameters = tree.GetCoefficients();
            diag.FinalParameters = diag.InitialParameters;
            return detail::MakeFitConfigurationError(validWeights.error(), std::move(diag));
        }

        Operon::Interpreter<Operon::Scalar, DTable> interpreter { dtable, dataset, &tree };
        // Cost batches internally (SelectBatch), so it needs the whole-dataset
        // target column (absolute, dataset-row-indexed), not a slice pre-cut
        // to range.
        Cost cost { &interpreter, problem->TargetValues(), range, &rng, batchSize, sampleWeights };
        Cost endpointCost { &interpreter, problem->TargetValues(), range, nullptr, 0, sampleWeights };
        Operon::detail::GradientSolverAdapter<Cost> bridge { &cost };
        Operon::detail::GradientSolverAdapter<Cost> endpointBridge { &endpointCost };

        auto coeff = tree.GetCoefficients();
        FitDiagnostics diag;
        diag.InitialParameters = coeff;

        Eigen::Array<Operon::Scalar, -1, 1> gradScratch(coeff.size());
        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> x0(coeff.data(), std::ssize(coeff));
        diag.InitialCost = endpointBridge(x0, gradScratch);
        if (auto const& error = endpointBridge.Error(); error) {
            diag.FinalParameters = coeff;
            return detail::MakeFitEvaluationError(*error, std::move(diag));
        }
        if (coeff.empty()) {
            auto const cost = diag.InitialCost;
            return detail::MakeFitOutcome(detail::ZeroParameterDiagnostics(std::move(diag), cost, 1));
        }

        auto rule = update_->Clone(coeff.size());
        SGDSolver<decltype(bridge)> solver(&bridge, rule.get());
        // The solver stops at the first failed or non-finite evaluation and
        // returns the last finite iterate, so coeff never receives a NaN
        // update; the endpoint evaluation below judges that iterate.
        auto x = solver.Optimize(x0, detail::SaturatingCast<int>(iterations));
        std::copy(x.begin(), x.end(), coeff.begin());

        Eigen::Map<Eigen::Array<Operon::Scalar, -1, 1> const> xFinal(coeff.data(), std::ssize(coeff));
        diag.FinalCost = endpointBridge(xFinal, gradScratch);
        diag.FinalParameters = coeff;
        diag.Iterations = solver.Epochs();
        diag.FunctionEvaluations = detail::ScaleBatchEvaluations(cost.FunctionEvaluations(), batchSize, range.Size());
        diag.JacobianEvaluations = detail::ScaleBatchEvaluations(cost.JacobianEvaluations(), batchSize, range.Size());
        if (auto const& error = endpointBridge.Error(); error) {
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
// It always solves with the Eigen LM backend (through detail::RunLeastSquares);
// there is no Tiny JIT variant.
//
// JacobianOnly=false (default): JIT-compiles both the forward pass (residuals)
//   and the Jacobian; falls back to interpreter when compilation fails.
// JacobianOnly=true: uses the interpreter for residuals; only the Jacobian is
//   JIT-compiled.  Useful when forward-pass compilation overhead exceeds savings.
//
// Pass a JitEvaluator constructed for the same GP run so the code cache is
// shared between fitness evaluation and coefficient optimisation.
template <typename DTable, bool JacobianOnly = false> struct JitLevenbergMarquardtOptimizer : public OptimizerBase {
    explicit JitLevenbergMarquardtOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem,
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
        diag.InitialParameters = tree.GetCoefficients();
        auto const localWeights = problem->Weights(range).value_or(Operon::Span<Operon::Scalar const> {});
        auto validWeights = ValidateWeights(localWeights, range.Size());
        if (!validWeights) {
            diag.FinalParameters = diag.InitialParameters;
            return detail::MakeFitConfigurationError(validWeights.error(), std::move(diag));
        }
        auto bound = interpreter.BindTree(range);
        if (!bound) {
            return detail::MakeUnevaluatedFitEvaluationError(
                GradientError { .Code = GradientErrorCode::EvaluationFailure, .Cause = std::move(bound.error()) },
                std::move(diag));
        }

        JIT::CompileMeta const* meta = jitEval_->GetOrCompileJacobian(tree);
        if (!JacobianOnly && (!meta || !meta->fn)) {
            meta = jitEval_->GetOrCompile(tree);
        }

        bool const hasFn = meta && meta->fn;
        bool const hasJacFn = meta && meta->jacFn;
        // In JacobianOnly mode only enter the JIT path when the Jacobian was actually compiled;
        // falling through to JitLeastSquaresCostFunction with a null jacFn wastes allocation for nothing.
        bool const useJitCf = !diag.InitialParameters.empty() && (hasFn || (JacobianOnly && hasJacFn));

        if (!useJitCf) {
            // Pure interpreter fallback — no JIT at all.
            Operon::InterpreterLeastSquaresCostFunction costFn {
                gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*> { &interpreter }, target, range
            };
            Operon::LeastSquaresLMAdapter<> cf { &costFn, localWeights, true };
            return detail::RunLeastSquares<OptimizerType::Eigen>(cf, iters, std::move(diag));
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

        Operon::JitLeastSquaresCostFunction costFn { gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*> {
                                                         &interpreter },
            evalFn, std::move(colPtrs), target, range, jacFn, std::move(jacColPtrs), meta->nVars, meta->nConsts };
        Operon::LeastSquaresLMAdapter<> cf { &costFn, localWeights, true };
        return detail::RunLeastSquares<OptimizerType::Eigen>(cf, iters, std::move(diag));
    }

    auto GetDispatchTable() const -> DTable const* { return dtable_.get(); }

private:
    gsl::not_null<DTable const*> dtable_;
    gsl::not_null<JIT::JitEvaluator const*> jitEval_;
};
#endif // HAVE_ASMJIT

} // namespace Operon
#endif
