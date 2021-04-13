#ifndef OPERON_NNLS_CERES_OPTIMIZER
#define OPERON_NNLS_CERES_OPTIMIZER

#include "nnls/tiny_optimizer.hpp"
#include <ceres/dynamic_autodiff_cost_function.h>
#include <ceres/dynamic_numeric_diff_cost_function.h>
#include <ceres/loss_function.h>
#include <ceres/solver.h>

namespace Operon {
template <typename CostFunctor>
struct DynamicCostFunction final : public ceres::DynamicCostFunction {
    using Scalar = typename CostFunctor::Scalar;

    DynamicCostFunction(CostFunctor& cf)
        : cf_(cf)
    {
        static_assert(CostFunctor::Storage == Eigen::RowMajor, "Operon::DynamicCostFunction requires col-major storage.");
        mutable_parameter_block_sizes()->push_back(cf_.NumParameters());
        set_num_residuals(cf_.NumResiduals());

        ENSURE(cf_.NumParameters() > 0);
        ENSURE(cf_.NumResiduals() > 0);
    }

    // required by ceres
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        EXPECT(parameters != nullptr);

        if constexpr (std::is_same_v<Scalar, double>) {
            return cf_(parameters[0], residuals, jacobians == nullptr ? nullptr : jacobians[0]);
        } else {
            // we need to make a copy
            int numResiduals = num_residuals();
            int numParameters = parameter_block_sizes().front();

            ENSURE(numResiduals > 0);
            ENSURE(numParameters > 0);

            Eigen::Map<const Eigen::Matrix<double, -1, 1>> pMap(parameters[0], numParameters);
            Eigen::Map<Eigen::Matrix<double, -1, 1>> rMap(residuals, numResiduals);

            Eigen::Matrix<Scalar, -1, 1> param = pMap.cast<Scalar>();
            Eigen::Matrix<Scalar, -1, 1> resid(numResiduals);

            if (jacobians == nullptr) {
                auto success = cf_(param.data(), resid.data(), nullptr);
                if (!success) {
                    return false;
                }
            } else {
                Eigen::Matrix<Scalar, -1, -1> jacob(numResiduals, numParameters);
                auto success = cf_(param.data(), resid.data(), jacob.data());
                if (!success) {
                    return false;
                }

                Eigen::Map<Eigen::Matrix<double, -1, -1>> jMap(jacobians[0], numResiduals, numParameters);
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

    CostFunctor& Functor() { return cf_; }
    CostFunctor const& Functor() const { return cf_; }

private:
    CostFunctor cf_;
};
} // namespace ceres

#endif
