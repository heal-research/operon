// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

// This header is intentionally separate from optimizer.hpp because <ceres/ceres.h>
// transitively includes glog, which defines a fatal CHECK macro that conflicts with
// Catch2 and other test frameworks. Include this header only in translation units
// that do not also use framework-level CHECK macros, or protect the include with
// #pragma push_macro("CHECK") / #pragma pop_macro("CHECK").

#ifndef OPERON_CERES_LM_OPTIMIZER_HPP
#define OPERON_CERES_LM_OPTIMIZER_HPP

#if defined(HAVE_CERES)

#include <ceres/ceres.h>

#include "dynamic_cost_function.hpp"
#include "../lm_cost_function.hpp"
#include "../optimizer.hpp"

namespace Operon {

// LM via the full Ceres Solver library (requires HAVE_CERES).
template<typename DTable>
struct CeresLMOptimizer final : public LMOptimizerBase<DTable> {
    explicit CeresLMOptimizer(gsl::not_null<DTable const*> dtable, gsl::not_null<Problem const*> problem)
        : LMOptimizerBase<DTable>{dtable, problem}
    {
    }

    [[nodiscard]] auto Optimize(Operon::RandomGenerator& /*unused*/, Operon::Tree const& tree) const -> OptimizerSummary final
    {
        auto const* dtable = this->GetDispatchTable();
        auto const* problem = this->GetProblem();
        auto const* dataset = problem->GetDataset();
        auto range  = problem->TrainingRange();
        auto target = problem->TargetValues(range);
        auto iterations = this->Iterations();

        Operon::Interpreter<Operon::Scalar, DTable> interpreter{dtable, dataset, &tree};
        Operon::LMCostFunction<Operon::Scalar> cf{
            gsl::not_null<Operon::InterpreterBase<Operon::Scalar> const*>{&interpreter}, target, range
        };

        auto x0 = tree.GetCoefficients();
        OptimizerSummary summary;
        summary.InitialParameters = x0;

        ceres::Solver::Summary s;
        if (!x0.empty()) {
            auto* costFunction = new Operon::DynamicCostFunction<decltype(cf)>(cf); // NOLINT
            auto* cfPtr = costFunction; // keep for stats after Solve

            Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
            Eigen::VectorXd params = m0.template cast<double>();

            ceres::Problem ceresProblem;
            ceresProblem.AddResidualBlock(costFunction, nullptr, params.data());

            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.logging_type = ceres::LoggingType::SILENT;
            options.max_num_iterations = static_cast<int>(iterations);
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 1;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.use_inner_iterations = false;
            ceres::Solve(options, &ceresProblem, &s);

            m0 = params.cast<Operon::Scalar>();
            summary.FunctionEvaluations = static_cast<int>(cfPtr->Functor().ResidualCalls());
            summary.JacobianEvaluations = static_cast<int>(cfPtr->Functor().JacobianCalls());
        }

        summary.FinalParameters = x0;
        summary.InitialCost = static_cast<Operon::Scalar>(s.initial_cost);
        summary.FinalCost   = static_cast<Operon::Scalar>(s.final_cost);
        summary.Iterations  = static_cast<int>(s.iterations.size());
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return summary;
    }
};

} // namespace Operon

#endif // HAVE_CERES
#endif // OPERON_CERES_LM_OPTIMIZER_HPP
