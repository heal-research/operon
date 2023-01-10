// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_OPTIMIZER_HPP
#define OPERON_OPTIMIZER_HPP

#include <ceres/types.h>
#include <unsupported/Eigen/LevenbergMarquardt>
#include "operon/autodiff/forward/forward.hpp"
#include "operon/core/comparison.hpp"
#include "tiny_cost_function.hpp"
#include "operon/ceres/tiny_solver.h"
#include "operon/autodiff/autodiff.hpp"

#if defined(HAVE_CERES)
#include "dynamic_cost_function.hpp"
#endif

namespace Operon {

enum class OptimizerType : int { Tiny, Eigen, Ceres };

struct OptimizerSummary {
    double InitialCost;
    double FinalCost;
    int Iterations;
    int FunctionEvaluations;
    int JacobianEvaluations;
    bool Success;
};

template<typename DerivativeCalculator>
struct OptimizerBase {
private:
    DerivativeCalculator& calculator_;
    Operon::Tree const& tree_;
    Operon::Dataset const& dataset_;

public:
    OptimizerBase(DerivativeCalculator& calculator, Tree const& tree, Dataset const& dataset)
        : calculator_(calculator)
        , tree_(tree)
        , dataset_(dataset)
    {
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return calculator_.GetInterpreter(); }
    [[nodiscard]] auto GetTree() const -> Tree const& { return tree_; }
    [[nodiscard]] auto GetDataset() const -> Dataset const& { return dataset_; }
    [[nodiscard]] auto GetDerivativeCalculator() -> DerivativeCalculator& { return calculator_; }
    [[nodiscard]] auto GetDerivativeCalculator() const -> DerivativeCalculator const& { return calculator_; }
};

namespace detail {
    inline auto CheckSuccess(double initialCost, double finalCost) {
        constexpr auto CHECK_NAN{true};
        return Operon::Less<CHECK_NAN>{}(finalCost, initialCost);
    }
} // namespace detail

template <typename DerivativeCalculator, OptimizerType = OptimizerType::Tiny>
struct NonlinearLeastSquaresOptimizer : public OptimizerBase<DerivativeCalculator> {
    NonlinearLeastSquaresOptimizer(DerivativeCalculator& calculator, Tree const& tree, Dataset const& dataset)
        : OptimizerBase<DerivativeCalculator>(calculator, tree, dataset)
    {
    }

    auto Optimize(Operon::Span<Operon::Scalar const> target, Range range, size_t iterations, OptimizerSummary& summary) -> std::vector<Operon::Scalar>
    {
        auto const& tree = this->GetTree();
        auto const& ds = this->GetDataset();
        auto& dc = this->GetDerivativeCalculator();

        Operon::CostFunction cf(tree, ds, target, range, dc);
        ceres::TinySolver<decltype(cf)> solver;
        solver.options.max_num_iterations = static_cast<int>(iterations);

        auto x0 = tree.GetCoefficients();
        auto m0 = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size());
        if (!x0.empty()) {
            typename decltype(solver)::Parameters p = m0.cast<typename decltype(cf)::Scalar>();
            solver.Solve(cf, &p);
            m0 = p.template cast<Operon::Scalar>();
        }
        summary.InitialCost = solver.summary.initial_cost;
        summary.FinalCost = solver.summary.final_cost;
        summary.Iterations = solver.summary.iterations;
        summary.FunctionEvaluations = solver.summary.iterations;
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return x0;
    }
};

template <typename DerivativeCalculator>
struct NonlinearLeastSquaresOptimizer<DerivativeCalculator, OptimizerType::Eigen> : public OptimizerBase<DerivativeCalculator> {
    NonlinearLeastSquaresOptimizer(DerivativeCalculator& calculator, Tree const& tree, Dataset const& dataset)
        : OptimizerBase<DerivativeCalculator>(calculator, tree, dataset)
    {
    }

    auto Optimize(Operon::Span<Operon::Scalar const> target, Range range, size_t iterations, OptimizerSummary& summary) -> std::vector<Operon::Scalar>
    {
        auto const& tree = this->GetTree();
        auto const& ds = this->GetDataset();
        auto& dc = this->GetDerivativeCalculator();

        Operon::CostFunction cf(tree, ds, target, range, dc);
        Eigen::LevenbergMarquardt<decltype(cf)> lm(cf);
        lm.setMaxfev(static_cast<int>(iterations+1));

        auto x0 = tree.GetCoefficients();
        Eigen::ComputationInfo info{};
        if (!x0.empty()) {
            Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
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
template <typename DerivativeCalculator>
struct NonlinearLeastSquaresOptimizer<DerivativeCalculator, OptimizerType::Ceres> : public OptimizerBase<DerivativeCalculator> {
    NonlinearLeastSquaresOptimizer(DerivativeCalculator& interpreter, Tree const& tree, Dataset const& dataset)
        : OptimizerBase<DerivativeCalculator>(interpreter, tree, dataset)
    {
    }

    auto Optimize(Operon::Span<Operon::Scalar const> target, Range range, size_t iterations, OptimizerSummary& summary) -> std::vector<Operon::Scalar>
    {
        auto const& tree = this->GetTree();
        auto const& ds = this->GetDataset();
        auto& dc = this->GetDerivativeCalculator();

        auto x0 = tree.GetCoefficients();

        Operon::CostFunction<DerivativeCalculator, Eigen::RowMajor> cf(tree, ds, target, range, dc);
        auto costFunction = new Operon::DynamicCostFunction(cf); // NOLINT

        ceres::Solver::Summary s;
        if (!x0.empty()) {
            Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>> m0(x0.data(), std::ssize(x0));
            auto sz = static_cast<Eigen::Index>(x0.size());
            Eigen::VectorXd params = Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>>(x0.data(), sz).template cast<double>();
            ceres::Problem problem;
            problem.AddResidualBlock(costFunction, nullptr, params.data());
            ceres::Solver::Options options;
            options.linear_solver_type = ceres::DENSE_QR;
            options.logging_type = ceres::LoggingType::SILENT;
            options.max_num_iterations = static_cast<int>(iterations - 1); // workaround since for some reason ceres sometimes does 1 more iteration
            options.minimizer_progress_to_stdout = false;
            options.num_threads = 1;
            options.trust_region_strategy_type = ceres::LEVENBERG_MARQUARDT;
            options.use_inner_iterations = false;
            Solve(options, &problem, &s);
            m0 = params.cast<Operon::Scalar>();
        }
        summary.InitialCost = s.initial_cost;
        summary.FinalCost = s.final_cost;
        summary.Iterations = static_cast<int>(s.iterations.size());
        summary.FunctionEvaluations = s.num_residual_evaluations;
        summary.JacobianEvaluations = s.num_jacobian_evaluations;
        summary.Success = detail::CheckSuccess(summary.InitialCost, summary.FinalCost);
        return x0;
    }
};
#endif

} // namespace Operon
#endif
