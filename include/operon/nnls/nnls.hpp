// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2022 Heal Research

#ifndef OPERON_NNLS_HPP
#define OPERON_NNLS_HPP

#include <unsupported/Eigen/NonLinearOptimization>

#include "operon/core/dual.hpp"
#include "residual_evaluator.hpp"
#include "tiny_cost_function.hpp"
#include "ceres/tiny_solver.h"

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
};

struct OptimizerBase {
    OptimizerBase(Interpreter const& interpreter, Tree& tree, Dataset const& dataset)
        : interpreter_(interpreter)
        , tree_(tree)
        , dataset_(dataset)
    {
    }

    [[nodiscard]] auto GetInterpreter() const -> Interpreter const& { return interpreter_.get(); }
    [[nodiscard]] auto GetTree() const -> Tree& { return tree_.get(); }
    [[nodiscard]] auto GetDataset() const -> Dataset const& { return dataset_.get(); }

private:
    std::reference_wrapper<Interpreter const> interpreter_;
    std::reference_wrapper<Tree> tree_;
    std::reference_wrapper<Dataset const> dataset_;
};

template <OptimizerType = OptimizerType::TINY>
struct NonlinearLeastSquaresOptimizer : public OptimizerBase {
    NonlinearLeastSquaresOptimizer(Interpreter const& interpreter, Tree& tree, Dataset const& dataset)
        : OptimizerBase(interpreter, tree, dataset)
    {
    }

    template <DerivativeMethod D = DerivativeMethod::AUTODIFF>
    auto Optimize(Operon::Span<const Operon::Scalar> const target, Range range, size_t iterations, bool writeCoefficients = true, bool /*unused*/ = false /* not used */) -> OptimizerSummary
    {
        static_assert(D == DerivativeMethod::AUTODIFF, "The tiny optimizer only supports autodiff.");
        ResidualEvaluator re(GetInterpreter(), GetTree(), GetDataset(), target, range);
        Operon::TinyCostFunction<ResidualEvaluator, Operon::Dual, Operon::Scalar, Eigen::ColMajor> cf(re);
        ceres::TinySolver<decltype(cf)> solver;
        solver.options.max_num_iterations = static_cast<int>(iterations);

        auto& tree = GetTree();
        auto x0 = tree.GetCoefficients();
        if (!x0.empty()) {
            decltype(solver)::Parameters params = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size()).cast<typename decltype(cf)::Scalar>();
            solver.Solve(cf, &params);
            if (writeCoefficients) {
                tree.SetCoefficients({ params.data(), x0.size() });
            }
        }
        OptimizerSummary sum {};
        sum.InitialCost = solver.summary.initial_cost;
        sum.FinalCost = solver.summary.final_cost;
        sum.Iterations = solver.summary.iterations;
        return sum;
    }
};

template <>
struct NonlinearLeastSquaresOptimizer<OptimizerType::EIGEN> : public OptimizerBase {
    NonlinearLeastSquaresOptimizer(Interpreter const& interpreter, Tree& tree, Dataset const& dataset)
        : OptimizerBase(interpreter, tree, dataset)
    {
    }

    template <DerivativeMethod D = DerivativeMethod::AUTODIFF>
    auto Optimize(Operon::Span<const Operon::Scalar> const target, Range range, size_t iterations, bool writeCoefficients = true, bool /*unused*/ = false) -> OptimizerSummary
    {
        static_assert(D == DerivativeMethod::AUTODIFF, "Eigen::LevenbergMarquardt only supports autodiff.");
        ResidualEvaluator re(GetInterpreter(), GetTree(), GetDataset(), target, range);
        Operon::TinyCostFunction<ResidualEvaluator, Operon::Dual, Operon::Scalar, Eigen::ColMajor> cf(re);
        Eigen::LevenbergMarquardt<decltype(cf), Operon::Scalar> lm(cf);
        lm.parameters.maxfev = static_cast<int>(iterations);

        auto& tree = GetTree();
        auto coeff = tree.GetCoefficients();
        if (!coeff.empty()) {
            Eigen::Matrix<Operon::Scalar, -1, 1> x0;
            x0 = Eigen::Map<decltype(x0)>(coeff.data(), static_cast<int>(coeff.size()));
            lm.minimize(x0);
            if (writeCoefficients) {
                tree.SetCoefficients({ x0.data(), static_cast<size_t>(x0.size()) });
            }
        }
        OptimizerSummary sum {};
        sum.InitialCost = -1;
        sum.FinalCost = -1;
        sum.Iterations = static_cast<int>(iterations);
        return sum;
    }
};

#if HAVE_CERES
template <>
struct NonlinearLeastSquaresOptimizer<OptimizerType::CERES> : public OptimizerBase {
    NonlinearLeastSquaresOptimizer(Interpreter const& interpreter, Tree& tree, Dataset const& dataset)
        : OptimizerBase(interpreter, tree, dataset)
    {
    }

    template <DerivativeMethod D = DerivativeMethod::AUTODIFF>
    auto Optimize(Operon::Span<const Operon::Scalar> const target, Range range, size_t iterations, bool writeCoefficients = true, bool report = false) -> OptimizerSummary
    {
        auto& tree = GetTree();
        auto coef = tree.GetCoefficients();

        auto const& interpreter = GetInterpreter();
        auto const& dataset = GetDataset();

        if (coef.empty()) {
            return OptimizerSummary {};
        }

        if (report) {
            fmt::print("x_0: ");
            for (auto c : coef) {
                fmt::print("{} ", c);
            }
            fmt::print("\n");
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

        auto sz = static_cast<Eigen::Index>(coef.size());
        Eigen::MatrixXd params = Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>>(coef.data(), sz).template cast<double>();
        ceres::Problem problem;
        problem.AddResidualBlock(costFunction, nullptr, params.data());

        ceres::Solver::Options options;
        options.max_num_iterations = static_cast<int>(iterations - 1); // workaround since for some reason ceres sometimes does 1 more iteration
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = report;
        options.num_threads = 1;
        options.logging_type = ceres::LoggingType::SILENT;

        ceres::Solver::Summary summary;
        Solve(options, &problem, &summary);

        if (report) {
            fmt::print("{}\n", summary.BriefReport());
            fmt::print("x_final: ");
            for (auto c : coef) {
                fmt::print("{} ", c);
            }
            fmt::print("\n");
        }
        if (writeCoefficients) {
            std::copy(params.data(), params.data() + params.size(), coef.begin());
            tree.SetCoefficients(coef);
        }
        OptimizerSummary sum {};
        sum.InitialCost = summary.initial_cost;
        sum.FinalCost = summary.final_cost;
        sum.Iterations = static_cast<int>(summary.iterations.size());
        return sum;
    }
};
#endif

} // namespace Operon
#endif
