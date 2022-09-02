// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_OPTIMIZER_HPP
#define OPERON_OPTIMIZER_HPP

#include <unsupported/Eigen/LevenbergMarquardt>

#include "operon/core/dual.hpp"
#include "operon/core/comparison.hpp"
#include "residual_evaluator.hpp"
#include "tiny_cost_function.hpp"
#include "operon/ceres/tiny_solver.h"

#if defined(HAVE_CERES)
#include "dynamic_cost_function.hpp"
#endif

namespace Operon {

enum class OptimizerType : int { TINY, EIGEN,
    CERES };
enum class DerivativeMethod : int { NUMERIC,
    AUTODIFF };

struct OptimizerSummary {
    double InitialCost;
    double FinalCost;
    int Iterations;
    int FunctionEvaluations;
    int JacobianEvaluations;
    bool Success;
};

struct OptimizerBase {
private:
    std::reference_wrapper<Interpreter const> interpreter_;
    std::reference_wrapper<Tree const> tree_;
    std::reference_wrapper<Dataset const> dataset_;

public:
    OptimizerBase(Interpreter const& interpreter, Tree const& tree, Dataset const& dataset)
        : interpreter_(interpreter)
        , tree_(tree)
        , dataset_(dataset)
    {
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_.get(); }
    [[nodiscard]] auto GetTree() const -> Tree const& { return tree_.get(); }
    [[nodiscard]] auto GetDataset() const -> Dataset const& { return dataset_.get(); }
    [[nodiscard]] auto GetCoefficients() const -> std::vector<Operon::Scalar> { return GetTree().GetCoefficients(); }
};

namespace detail {
    inline auto CheckSuccess(double initialCost, double finalCost) {
        constexpr auto CHECK_NAN{true};
        return Operon::Less<CHECK_NAN>{}(finalCost, initialCost);
    }
} // namespace detail

template <OptimizerType = OptimizerType::TINY>
struct NonlinearLeastSquaresOptimizer : public OptimizerBase {
    NonlinearLeastSquaresOptimizer(Interpreter const& interpreter, Tree const& tree, Dataset const& dataset)
        : OptimizerBase(interpreter, tree, dataset)
    {
    }

    template <DerivativeMethod D = DerivativeMethod::AUTODIFF>
    auto Optimize(Operon::Span<Operon::Scalar const> target, Range range, size_t iterations, OptimizerSummary& summary) -> std::vector<Operon::Scalar> 
    {
        static_assert(D == DerivativeMethod::AUTODIFF, "The tiny optimizer only supports autodiff.");
        ResidualEvaluator re(GetInterpreter(), GetTree(), GetDataset(), target, range);
        Operon::TinyCostFunction<ResidualEvaluator, Operon::Dual, Operon::Scalar, Eigen::ColMajor> cf(re);
        ceres::TinySolver<decltype(cf)> solver;
        solver.options.max_num_iterations = static_cast<int>(iterations);

        auto x0 = GetCoefficients();
        auto m0 = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size()); 
        if (!x0.empty()) {
            decltype(solver)::Parameters p = m0.cast<typename decltype(cf)::Scalar>();
            solver.Solve(cf, &p);
            m0 = p.cast<Operon::Scalar>();
        }
        summary.InitialCost = solver.summary.initial_cost;
        summary.FinalCost = solver.summary.final_cost;
        summary.Iterations = solver.summary.iterations;
        summary.FunctionEvaluations = solver.summary.iterations;
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return x0;
    }
};

template <>
struct NonlinearLeastSquaresOptimizer<OptimizerType::EIGEN> : public OptimizerBase {
    NonlinearLeastSquaresOptimizer(Interpreter const& interpreter, Tree const& tree, Dataset const& dataset)
        : OptimizerBase(interpreter, tree, dataset)
    {
    }

    template <DerivativeMethod D = DerivativeMethod::AUTODIFF>
    auto Optimize(Operon::Span<Operon::Scalar const> target, Range range, size_t iterations, OptimizerSummary& summary) -> std::vector<Operon::Scalar>
    {
        static_assert(D == DerivativeMethod::AUTODIFF, "Eigen::LevenbergMarquardt only supports autodiff.");
        ResidualEvaluator re(GetInterpreter(), GetTree(), GetDataset(), target, range);
        Operon::TinyCostFunction<ResidualEvaluator, Operon::Dual, Operon::Scalar, Eigen::ColMajor> cf(re);
        Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
        lm.setMaxfev(static_cast<int>(iterations+1));

        auto x0 = GetCoefficients();
        Eigen::ComputationInfo info{};
        if (!x0.empty()) {
            Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), static_cast<Eigen::Index>(x0.size()));
            Eigen::Matrix<Operon::Scalar, -1, 1> m = m0;
            
            // do the minimization loop manually because we want to extract the initial cost
            Eigen::LevenbergMarquardtSpace::Status status = lm.minimizeInit(m);
            summary.InitialCost = summary.FinalCost = lm.fnorm() * lm.fnorm(); // get the initial cost after calling minimizeInit()
            if (status != Eigen::LevenbergMarquardtSpace::ImproperInputParameters) {
                do {
                    status = lm.minimizeOneStep(m);
                } while (status == Eigen::LevenbergMarquardtSpace::Running);
            }
            m0 = m;
        }
        summary.FinalCost = lm.fnorm() * lm.fnorm();
        summary.Iterations = static_cast<int>(lm.iterations());
        summary.FunctionEvaluations = static_cast<int>(lm.nfev());
        summary.JacobianEvaluations = static_cast<int>(lm.njev());
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return x0;
    }
};

#if HAVE_CERES
template <>
struct NonlinearLeastSquaresOptimizer<OptimizerType::CERES> : public OptimizerBase {
    NonlinearLeastSquaresOptimizer(Interpreter const& interpreter, Tree const& tree, Dataset const& dataset)
        : OptimizerBase(interpreter, tree, dataset)
    {
    }

    template <DerivativeMethod D = DerivativeMethod::AUTODIFF>
    auto Optimize(Operon::Span<Operon::Scalar const> target, Range range, size_t iterations, bool writeCoefficients = true, OptimizerSummary& summary) -> std::vector<Operon::Scalar> 
    {
        auto x0 = GetCoeff();

        auto const& interpreter = GetInterpreter();
        auto const& dataset = GetDataset();

        if (x0.empty()) {
            return OptimizerSummary {};
        }

        ceres::DynamicCostFunction* costFunction = nullptr;
        if constexpr (D == DerivativeMethod::AUTODIFF) {
            ResidualEvaluator re(interpreter, tree, dataset, target, range);
            TinyCostFunction<ResidualEvaluator, Operon::Dual, Operon::Scalar, Eigen::RowMajor> f(re);
            costFunction = new Operon::DynamicCostFunction<decltype(f)>(f);
        } else {
            auto* eval = new ResidualEvaluator(interpreter, tree, dataset, target, range); // NOLINT
            costFunction = new ceres::DynamicNumericDiffCostFunction(eval);
            costFunction->AddParameterBlock(static_cast<int>(coef.size()));
            costFunction->SetNumResiduals(static_cast<int>(target.size()));
        }

        auto sz = static_cast<Eigen::Index>(x0.size());
        Eigen::MatrixXd params = Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>>(x0.data(), sz).template cast<double>();
        ceres::Problem problem;
        problem.AddResidualBlock(costFunction, nullptr, params.data());

        ceres::Solver::Options options;
        options.max_num_iterations = static_cast<int>(iterations - 1); // workaround since for some reason ceres sometimes does 1 more iteration
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = report;
        options.num_threads = 1;
        options.logging_type = ceres::LoggingType::SILENT;

        ceres::Solver::Summary s;
        Solve(options, &problem, &s);
        sum.InitialCost = s.initial_cost;
        sum.FinalCost = s.final_cost;
        sum.Iterations = static_cast<int>(s.iterations.size());
        sum.FunctionEvaluations = s.num_residual_evaluations;
        sum.JacobianEvaluations = s.num_jacobian_evaluations;
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return x0;
    }
};
#endif

} // namespace Operon
#endif
