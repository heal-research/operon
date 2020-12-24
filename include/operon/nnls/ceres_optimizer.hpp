#ifndef OPERON_NNLS_CERES_OPTIMIZER
#define OPERON_NNLS_CERES_OPTIMIZER

#include "nnls/tiny_optimizer.hpp"

#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>

namespace Operon {
template <typename CostFunctor, typename JetT, int StorageOrder = Eigen::RowMajor>
struct DynamicAutoDiffCostFunction final : public ceres::DynamicCostFunction {
    using Scalar = typename JetT::Scalar;

    DynamicAutoDiffCostFunction(const Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range)
        : cf_(tree, dataset, targetValues, range)
    {
        mutable_parameter_block_sizes()->push_back(cf_.NumParameters());
        set_num_residuals(cf_.NumResiduals());
    }

    // required by ceres
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        EXPECT(parameters != nullptr);

        if constexpr (std::is_same_v<Scalar, double>) {
            return cf_(parameters[0], residuals, jacobians != nullptr ? jacobians[0] : nullptr);
        } else {
            // we need to make a copy
            int numResiduals = num_residuals();
            int numParameters = parameter_block_sizes().front();

            Eigen::Map<const Eigen::Matrix<double, -1, 1>> pMap(parameters[0], numParameters);
            Eigen::Map<Eigen::Matrix<double, -1, 1>> rMap(residuals, numResiduals);

            Eigen::Matrix<Scalar, -1, 1> param = pMap.cast<Scalar>();
            Eigen::Matrix<Scalar, -1, 1> resid(numResiduals);

            bool success;
            if (jacobians == nullptr) {
                success = cf_(param.data(), resid.data(), nullptr);
                if (!success) {
                    return false;
                }
            } else {
                Eigen::Map<Eigen::Matrix<double, -1, -1>> jMap(jacobians[0], numResiduals, numParameters);
                Eigen::Matrix<Scalar, -1, -1> jacob(numResiduals, numParameters);

                success = cf_(param.data(), resid.data(), jacob.data());
                if (!success) {
                    return false;
                }

                jMap = jacob.template cast<double>();
            }
            rMap = resid.template cast<double>();

            return true;
        }
    }

    void AddParameterBlock(int) override
    {
        throw new std::runtime_error("This method should not be used.");
    }

    void SetNumResiduals(int) override
    {
        throw new std::runtime_error("This method should not be used.");
    }

private:
    TinyCostFunction<CostFunctor, JetT, StorageOrder> cf_;
};

template <DerivativeMethod M>
struct Optimizer<M, OptimizerType::CERES> : public OptimizerBase {
    virtual OptimizerSummary Optimize(Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range, size_t iterations = 50, bool writeCoefficients = true, bool report = false) const override
    {
        using ceres::CauchyLoss;
        using ceres::DynamicAutoDiffCostFunction;
        using ceres::DynamicCostFunction;
        using ceres::DynamicNumericDiffCostFunction;
        using ceres::Problem;
        using ceres::Solver;

        Solver::Summary summary;
        std::vector<double> coef;
        for (auto const& node : tree.Nodes()) {
            if (node.IsLeaf())
                coef.push_back(node.Value);
        }

        if (coef.empty()) {
            return OptimizerSummary{};
        }

        if (report) {
            fmt::print("x_0: ");
            for (auto c : coef)
                fmt::print("{} ", c);
            fmt::print("\n");
        }

        DynamicCostFunction* costFunction;
        if constexpr (M == DerivativeMethod::AUTODIFF) {
            costFunction = new Operon::DynamicAutoDiffCostFunction<ResidualEvaluator, Dual, Eigen::RowMajor>(tree, dataset, targetValues, range);
        } else {
            auto eval = new ResidualEvaluator(tree, dataset, targetValues, range);
            costFunction = new DynamicNumericDiffCostFunction(eval);
        }

        Problem problem;
        problem.AddResidualBlock(costFunction, nullptr, coef.data());

        Solver::Options options;
        options.max_num_iterations = static_cast<int>(iterations - 1); // workaround since for some reason ceres sometimes does 1 more iteration
        options.linear_solver_type = ceres::DENSE_QR;
        options.minimizer_progress_to_stdout = report;
        options.num_threads = 1;
        options.logging_type = ceres::LoggingType::SILENT;
        Solve(options, &problem, &summary);

        if (report) {
            fmt::print("{}\n", summary.BriefReport());
            fmt::print("x_final: ");
            for (auto c : coef)
                fmt::print("{} ", c);
            fmt::print("\n");
        }
        if (writeCoefficients) {
            size_t idx = 0;
            for (auto& node : tree.Nodes()) {
                if (node.IsLeaf()) {
                    node.Value = static_cast<Operon::Scalar>(coef[idx++]);
                }
            }
        }
        OptimizerSummary sum;
        sum.InitialCost = summary.initial_cost;
        sum.FinalCost = summary.final_cost;
        sum.Iterations = static_cast<int>(summary.iterations.size());
        return sum;
    }
};
} // namespace ceres

#endif
