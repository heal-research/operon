// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_NNLS_HPP
#define OPERON_NNLS_HPP

#include "core/types.hpp"
#include "tiny_optimizer.hpp"

#if defined(HAVE_CERES)
#include "ceres_optimizer.hpp"
#endif

namespace Operon {

enum class OptimizerType : int { TINY,
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

protected:
    std::reference_wrapper<Interpreter const> interpreter_;
    std::reference_wrapper<Tree> tree_;
    std::reference_wrapper<Dataset const> dataset_;
};

template<OptimizerType = OptimizerType::TINY>
struct NonlinearLeastSquaresOptimizer : public OptimizerBase {
    NonlinearLeastSquaresOptimizer(Interpreter const& interpreter, Tree& tree, Dataset const& dataset)
        : OptimizerBase(interpreter, tree, dataset)
    {
    }

    template<DerivativeMethod D = DerivativeMethod::AUTODIFF>
    OptimizerSummary Optimize(gsl::span<const Operon::Scalar> const target, Range range, size_t iterations, bool writeCoefficients = true, bool = false /* not used */)
    {
        static_assert(D == DerivativeMethod::AUTODIFF, "The tiny optimizer only supports autodiff.");
        ResidualEvaluator re(this->interpreter_, tree_, dataset_, target, range);
        Operon::TinyCostFunction<ResidualEvaluator, Dual, Eigen::ColMajor> cf(re);
        ceres::TinySolver<decltype(cf)> solver;
        solver.options.max_num_iterations = static_cast<int>(iterations);

        auto& tree = this->tree_.get();
        auto x0 = tree.GetCoefficients();
        if (!x0.empty()) {
            decltype(solver)::Parameters params = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size()).cast<typename decltype(cf)::Scalar>();
            solver.Solve(cf, &params);
            if (writeCoefficients) {
                tree.SetCoefficients({ params.data(), x0.size() });
            }
        }
        OptimizerSummary summary;
        summary.InitialCost = solver.summary.initial_cost;
        summary.FinalCost = solver.summary.final_cost;
        summary.Iterations = solver.summary.iterations;
        return summary;
    };
};

template<>
struct NonlinearLeastSquaresOptimizer<OptimizerType::CERES> : public OptimizerBase {
    NonlinearLeastSquaresOptimizer(Interpreter const& interpreter, Tree& tree, Dataset const& dataset)
        : OptimizerBase(interpreter, tree, dataset)
    {
    }

    template<DerivativeMethod D = DerivativeMethod::AUTODIFF>
    OptimizerSummary Optimize(gsl::span<const Operon::Scalar> const target, Range range, size_t iterations, bool writeCoefficients = true, bool report = false)
    {
        auto& tree = this->tree_.get();
        auto coef = tree.GetCoefficients();

        if (coef.empty()) {
            return OptimizerSummary{};
        }

        if (report) {
            fmt::print("x_0: ");
            for (auto c : coef)
                fmt::print("{} ", c);
            fmt::print("\n");
        }

        ceres::DynamicCostFunction* costFunction;
        if constexpr (D == DerivativeMethod::AUTODIFF) {
            ResidualEvaluator re(this->interpreter_, tree_, this->dataset_, target, range);
            TinyCostFunction<ResidualEvaluator, Operon::Dual, Eigen::RowMajor> f(re);
            costFunction = new Operon::DynamicCostFunction<decltype(f)>(f);
        } else {
            auto eval = new ResidualEvaluator(this->interpreter_, tree_, dataset_, target, range);
            costFunction = new ceres::DynamicNumericDiffCostFunction(eval);
            costFunction->AddParameterBlock(gsl::narrow<int>(coef.size()));
            costFunction->SetNumResiduals(gsl::narrow<int>(target.size()));
        }

        Eigen::MatrixXd params = Eigen::Map<Eigen::Matrix<Operon::Scalar, -1, 1>>(coef.data(), coef.size()).template cast<double>();
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
            for (auto c : coef)
                fmt::print("{} ", c);
            fmt::print("\n");
        }
        if (writeCoefficients) {
            std::copy(params.data(), params.data()+params.size(), coef.begin());
            tree.SetCoefficients(coef);
        }
        OptimizerSummary sum;
        sum.InitialCost = summary.initial_cost;
        sum.FinalCost = summary.final_cost;
        sum.Iterations = static_cast<int>(summary.iterations.size());
        return sum;
    }
};

} // namespace Operon
#endif
