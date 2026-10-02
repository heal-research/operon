// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_LEAST_SQUARES_FIT_HPP
#define OPERON_LEAST_SQUARES_FIT_HPP

#include <algorithm>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include <Eigen/Core>
#include <unsupported/Eigen/LevenbergMarquardt>

#include "operon/ceres/tiny_solver.h"
#include "operon/core/contracts.hpp"
#include "operon/core/memory_view.hpp"
#include "operon/core/types.hpp"
#include "operon/optimizer/fit_outcome.hpp"
#include "operon/optimizer/least_squares.hpp"
#include "operon/optimizer/least_squares_lm_adapter.hpp"

namespace Operon {

/** Levenberg-Marquardt implementation used to solve a least-squares problem. */
enum class OptimizerType : int { Tiny,
    Eigen };

namespace detail {
    // LM backends: evaluates the weighted half sum of squares (the cost
    // convention of both Tiny and Eigen summaries) through the adapter once.
    template <typename Adapter>
    inline auto FitZeroParameterLeastSquares(Adapter const& cf, FitDiagnostics diag) -> FitOutcome
    {
        std::vector<Operon::Scalar> residuals(static_cast<std::size_t>(cf.NumResiduals()));
        auto const ok = cf.Evaluate(diag.InitialParameters.data(), residuals.data(), nullptr);
        auto const functionEvaluations = static_cast<int>(cf.ResidualCalls());
        if (!ok) {
            auto failed = ZeroParameterDiagnostics(std::move(diag), std::numeric_limits<Operon::Scalar>::quiet_NaN(), functionEvaluations);
            if (auto const& error = cf.Error(); error) {
                return MakeFitEvaluationError(ToGradientError(*error), std::move(failed));
            }
            return MakeFitOutcome(std::move(failed));
        }
        Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1> const> r(residuals.data(), std::ssize(residuals));
        return MakeFitOutcome(ZeroParameterDiagnostics(std::move(diag), Operon::Scalar { 0.5 } * r.squaredNorm(), functionEvaluations));
    }

    // The single least-squares driver shared by LevenbergMarquardtOptimizer
    // (Tiny and Eigen), JitLevenbergMarquardtOptimizer (Eigen), and the public
    // FitLeastSquares. `cf` is a LeastSquaresLMAdapter (or any type with the
    // same Evaluate/Error/ResidualCalls/JacobianCalls surface) that the caller
    // owns and that is not shared with another thread; diag.InitialParameters
    // holds the starting point and is the only diag field read. The adapter's
    // first recorded error, if any, is classified as FitEvaluationError;
    // otherwise success is FinalCost < InitialCost (non-finite costs fail).
    //
    // `iterations` is an accepted-LM-step budget on both backends:
    //  - Tiny: caps accepted steps (max_num_accepted_steps); max_num_iterations,
    //    which bounds total attempts accepted or rejected, is
    //    iterations * (n + 1).
    //  - Eigen: caps accepted steps (lm.iterations() - 1, because Eigen counts
    //    from 1); maxfev, which bounds function evaluations including rejected
    //    trial steps, is max(iterations * (n + 1), 1).
    // iterations == 0 performs no step on either backend. Oversized budgets
    // saturate at the backend's integer limits instead of wrapping.
    // Diagnostics().Iterations is the accepted-step count; FunctionEvaluations
    // and JacobianEvaluations are the adapter's call counters.
    //
    // Eigen requires NumResiduals() >= NumParameters(): fewer residuals than
    // parameters is rejected before the solver or the cost runs, as a
    // FitEvaluationError with GradientErrorCode::InvalidShape
    // (Expected == NumParameters(), Actual == NumResiduals()), NaN costs,
    // FinalParameters == InitialParameters, and zero counters.
    //
    // If the initial evaluation fails there is no valid initial cost: both
    // costs are NaN and Iterations is 0.
    //
    // A parameterless problem skips the solver (see ZeroParameterDiagnostics).
    template <OptimizerType Backend, typename Adapter>
    [[nodiscard]] inline auto RunLeastSquares(Adapter& cf, std::size_t iterations, FitDiagnostics diag) -> FitOutcome
    {
        if (diag.InitialParameters.empty()) {
            return FitZeroParameterLeastSquares(cf, std::move(diag));
        }

        if constexpr (Backend == OptimizerType::Eigen) {
            if (cf.NumResiduals() < cf.NumParameters()) {
                return MakeUnevaluatedFitEvaluationError(
                    GradientError { .Code = GradientErrorCode::InvalidShape, .Expected = static_cast<std::size_t>(cf.NumParameters()), .Actual = static_cast<std::size_t>(cf.NumResiduals()) },
                    std::move(diag));
            }
        }

        auto x0 = diag.InitialParameters;
        auto const attempts = SaturatingMultiply(iterations, x0.size() + 1);
        Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>> m0(x0.data(), std::ssize(x0));
        if constexpr (Backend == OptimizerType::Tiny) {
            ceres::TinySolver<Adapter> solver;
            // max_num_accepted_steps counts accepted LM steps only, matching
            // the Eigen branch below - unlike max_num_iterations, which
            // (unmodified) bounds total attempts, accepted or rejected.
            // max_num_iterations is still set, as a MINPACK-convention-scaled
            // safety net on rejected-retry attempts, mirroring maxfev's role
            // for the Eigen backend.
            solver.options.max_num_accepted_steps = SaturatingCast<int>(iterations);
            solver.options.max_num_iterations = SaturatingCast<int>(attempts);
            typename decltype(solver)::ParameterVector p = m0.template cast<typename Adapter::Scalar>();
            solver.Solve(cf, &p);
            m0 = p.template cast<Operon::Scalar>();
            diag.FinalParameters = std::move(x0);
            diag.InitialCost = solver.summary.initial_cost;
            diag.FinalCost = solver.summary.final_cost;
            diag.Iterations = solver.summary.iterations;
        } else {
            Eigen::LevenbergMarquardt<Adapter> lm(cf);
            // `iterations` counts accepted LM steps, matching Tiny's
            // max_num_accepted_steps - it is not itself a function-evaluation
            // budget. maxfev is still needed as a bound on rejected
            // trust-region retries within/across those steps (Eigen's own
            // default, 400 regardless of iterations, is enough per individual to
            // exhaust the CLI's overall --evaluations budget across a full GP
            // run), scaled by parameter count using MINPACK's own convention
            // (100*(n+1) for its "no fixed iteration count" default) so the
            // ceiling grows with problem size instead of being a fixed constant.
            auto const budget = SaturatingCast<Eigen::Index>(iterations);
            lm.setMaxfev(std::max<Eigen::Index>(SaturatingCast<Eigen::Index>(attempts), 1));

            Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;

            // do the minimization loop manually because we want to extract the initial cost
            auto status = lm.minimizeInit(m);
            if (status == Eigen::LevenbergMarquardtSpace::NotStarted) {
                diag.InitialCost = lm.fnorm() * lm.fnorm() * Operon::Scalar { 0.5 }; // initial cost after minimizeInit()
                // lm.iterations() is 1 after minimizeInit() and gains one per
                // accepted step, so accepted steps = lm.iterations() - 1.
                while ((status == Eigen::LevenbergMarquardtSpace::NotStarted || status == Eigen::LevenbergMarquardtSpace::Running)
                    && lm.iterations() - 1 < budget) {
                    status = lm.minimizeOneStep(m);
                }
                diag.FinalCost = lm.fnorm() * lm.fnorm() * Operon::Scalar { 0.5 };
                diag.Iterations = SaturatingCast<int>(static_cast<std::size_t>(lm.iterations() - 1));
            } else {
                // The initial evaluation failed (UserAsked) or the input was
                // rejected: fnorm() and iterations() were never set, so there
                // is no valid cost to report.
                diag.InitialCost = diag.FinalCost = std::numeric_limits<Operon::Scalar>::quiet_NaN();
                diag.Iterations = 0;
            }
            m0 = m;
            diag.FinalParameters = std::move(x0);
        }
        diag.FunctionEvaluations = static_cast<int>(cf.ResidualCalls());
        diag.JacobianEvaluations = static_cast<int>(cf.JacobianCalls());
        if (auto const& error = cf.Error(); error) {
            return MakeFitEvaluationError(ToGradientError(*error), std::move(diag));
        }
        return MakeFitOutcome(std::move(diag));
    }
} // namespace detail

/**
 * Configuration of FitLeastSquares.
 *
 * Backend  Tiny (default) or Eigen; must be an enumerator.
 * Iterations  Accepted Levenberg-Marquardt step budget, with the per-backend
 *     details documented on FitLeastSquares. Default 100 (the OptimizerBase
 *     default).
 * Weights  Numerical WLS weights following ValidateWeights: empty
 *     (unweighted), one value (uniform), or cost.NumResiduals() values
 *     (per-row); finite and nonnegative. Borrowed for the duration of the
 *     call only. They scale each residual and Jacobian row by sqrt(w_i); they
 *     are never interpreted as statistical sigma. A violation is reported as a
 *     FitConfigurationError whose WeightError::Row is the index into
 *     options.Weights itself (0 for a size mismatch).
 * RecoverNonFinite  false (default): a non-finite residual or Jacobian entry
 *     from an otherwise successful cost evaluation is a typed
 *     NonFiniteEvaluation FitEvaluationError. true: it is handed back to the
 *     solver as a rejected trial step (damping is increased); this is what the
 *     interpreter and JIT optimizers use.
 */
struct LeastSquaresFitOptions {
    OptimizerType Backend { OptimizerType::Tiny };
    std::size_t Iterations { 100 }; // NOLINT
    ConstScalarSpan Weights {};
    bool RecoverNonFinite { false };
};

/**
 * Minimizes 0.5 * sum_i w_i * r_i(x)^2 over x, starting from
 * initialParameters, for any LeastSquaresCostFunction, using the same
 * adapter and solver driver as LevenbergMarquardtOptimizer. Callers need no
 * solver loop, backend adapter, or detail:: type of their own.
 *
 * Ownership and lifetime: `cost`, `initialParameters`, and options.Weights are
 * borrowed for the duration of the call only; nothing is retained after the
 * function returns. The returned FitOutcome owns its parameter vectors and
 * holds no reference to the inputs. `cost` is only called through const
 * Evaluate(), always from the calling thread.
 *
 * Thread safety: the function holds no global or static state and builds all
 * solver and adapter state locally, so concurrent calls on distinct cost
 * objects are safe. Concurrent calls sharing one cost object are safe only if
 * that cost documents its const Evaluate() as thread-safe.
 *
 * Parameters: initialParameters.size() must equal cost.NumParameters().
 *
 * Outcome (the same FitOutcome model as OptimizerBase::Optimize; use
 * Diagnostics(outcome) for the fields common to every case):
 *  - FitResult: FinalCost < InitialCost. FinalParameters holds the solution.
 *  - FitFailure: a valid fit that did not improve the cost, including a
 *    non-finite final cost. FinalParameters is the solver's last iterate.
 *  - FitConfigurationError (WeightError): options.Weights violates
 *    ValidateWeights for cost.NumResiduals(). The cost is not evaluated;
 *    FinalParameters equals InitialParameters and counters are zero.
 *  - FitEvaluationError (GradientError): the cost failed or, with
 *    RecoverNonFinite == false, produced a non-finite output; the first error
 *    is reported with its Code/Row/Column/Cause preserved from the cost's
 *    LeastSquaresError. Counters and FinalParameters describe the point
 *    reached when the error stopped the solve. If the very first evaluation
 *    fails there is no valid initial cost: InitialCost and FinalCost are NaN
 *    and Iterations is 0. Two shape errors are reported as
 *    GradientErrorCode::InvalidShape before the cost is evaluated (NaN costs,
 *    FinalParameters == InitialParameters, zero counters):
 *      - a parameter-count mismatch, with Expected == cost.NumParameters()
 *        and Actual == initialParameters.size();
 *      - with the Eigen backend, an underdetermined problem
 *        (cost.NumResiduals() < cost.NumParameters()), with Expected ==
 *        cost.NumParameters() (the minimum residual count) and Actual ==
 *        cost.NumResiduals(). Tiny has no such restriction.
 *
 * Costs: InitialCost and FinalCost use the 0.5 * sum(w_i * r_i^2)
 * convention (never gradient-norm or statistical likelihood values).
 * FunctionEvaluations and JacobianEvaluations count cost Evaluate() calls
 * that requested residuals and a Jacobian respectively.
 *
 * Iterations is an accepted-LM-step budget on both backends, and
 * Diagnostics().Iterations is the number of accepted steps taken:
 *  - Tiny: caps accepted LM steps; Tiny's max_num_iterations, which bounds
 *    total attempts accepted or rejected, is Iterations * (n + 1).
 *  - Eigen: caps accepted LM steps; maxfev, which bounds function evaluations
 *    including rejected trial steps, is max(Iterations * (n + 1), 1).
 *  - Iterations == 0 performs no step on either backend (the initial point is
 *    still evaluated; the outcome is a FitFailure with unchanged parameters).
 *  - Budgets too large for the backend's integer type saturate at its
 *    maximum; they never wrap.
 *
 * Zero parameters: no solver runs. The cost is evaluated once (residuals
 * only); InitialCost == FinalCost == 0.5 * sum(w_i * r_i^2),
 * Iterations == 0, FunctionEvaluations == 1, JacobianEvaluations == 0, and
 * the parameter vectors are empty. The outcome is a FitFailure (no
 * improvement), or a FitEvaluationError if that single evaluation fails (costs
 * are then NaN).
 */
[[nodiscard]] inline auto FitLeastSquares(
    LeastSquaresCostFunction const& cost,
    ConstScalarSpan initialParameters,
    LeastSquaresFitOptions const& options = {}) -> FitOutcome
{
    FitDiagnostics diag;
    diag.InitialParameters.assign(initialParameters.begin(), initialParameters.end());

    if (initialParameters.size() != cost.NumParameters()) {
        return detail::MakeUnevaluatedFitEvaluationError(
            GradientError { .Code = GradientErrorCode::InvalidShape, .Expected = cost.NumParameters(), .Actual = initialParameters.size() },
            std::move(diag));
    }
    if (auto validWeights = ValidateWeights(options.Weights, cost.NumResiduals()); !validWeights) {
        diag.FinalParameters = diag.InitialParameters;
        return detail::MakeFitConfigurationError(validWeights.error(), std::move(diag));
    }

    EXPECT(options.Backend == OptimizerType::Tiny || options.Backend == OptimizerType::Eigen);
    LeastSquaresLMAdapter<> adapter { &cost, options.Weights, options.RecoverNonFinite };
    if (options.Backend == OptimizerType::Eigen) {
        return detail::RunLeastSquares<OptimizerType::Eigen>(adapter, options.Iterations, std::move(diag));
    }
    return detail::RunLeastSquares<OptimizerType::Tiny>(adapter, options.Iterations, std::move(diag));
}

} // namespace Operon

#endif
