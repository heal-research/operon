#ifndef OPERON_DYNAMIC_AUTODIFF_COST_FUNCTION_HPP
#define OPERON_DYNAMIC_AUTODIFF_COST_FUNCTION_HPP

#include <cmath>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include "ceres/dynamic_cost_function.h"
#include "ceres/types.h"

#include "core/types.hpp"

namespace Operon {

// this class represents a simplification of ceres::DynamicAutoDiffCostFunction
// more tailored for our use case: a single parameter block and a single jacobian
// (exactly the same as the TinyCostFunction, which it uses to get back results)
// notably, this function can operate in single- or double-precision
// (Ceres supports double only)
template <typename CostFunctor, int Stride = 4>
class DynamicAutoDiffCostFunction : public ceres::DynamicCostFunction {
public:
    using Scalar = typename CostFunctor::Scalar;

    // Takes ownership by default.
    DynamicAutoDiffCostFunction(CostFunctor* functor,
        ceres::Ownership ownership = ceres::TAKE_OWNERSHIP)
        : functor_(functor)
        , ownership_(ownership)
    {
    }

    explicit DynamicAutoDiffCostFunction(DynamicAutoDiffCostFunction&& other)
        : functor_(std::move(other.functor_))
        , ownership_(other.ownership_)
    {
    }

    virtual ~DynamicAutoDiffCostFunction()
    {
        // Manually release pointer if configured to not take ownership
        // rather than deleting only if ownership is taken.  This is to
        // stay maximally compatible to old user code which may have
        // forgotten to implement a virtual destructor, from when the
        // AutoDiffCostFunction always took ownership.
        if (ownership_ == ceres::DO_NOT_TAKE_OWNERSHIP) {
            functor_.release();
        }
    }

    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        EXPECT(parameters != nullptr);
        if constexpr(std::is_same_v<Scalar, double>) {
            return (*functor_).Evaluate(parameters[0], residuals, jacobians != nullptr ? jacobians[0] : nullptr);
        } else {
            // we need to make a copy
            int numResiduals  = (*functor_).NumResiduals();
            int numParameters = (*functor_).NumParameters();

            Eigen::Map<const Eigen::Matrix<double, -1, 1>> pMap(parameters[0], numParameters);
            Eigen::Map<Eigen::Matrix<double, -1, 1>> rMap(residuals, numResiduals);

            Eigen::Matrix<Scalar, -1, 1> param = pMap.cast<Scalar>(); 
            Eigen::Matrix<Scalar, -1, 1> resid(numResiduals); 

            bool success;
            if (jacobians == nullptr) {
                success = (*functor_).Evaluate(param.data(), resid.data(), nullptr);
                if (!success) { return false; }
            } else {
                Eigen::Map<Eigen::Matrix<double, -1, -1>> jMap(jacobians[0], numResiduals, numParameters);
                Eigen::Matrix<Scalar, -1, -1> jacob(numResiduals, numParameters);

                success = (*functor_).Evaluate(param.data(), resid.data(), jacob.data());
                if (!success) { return false; }

                jMap = jacob.template cast<double>();
            }
            rMap = resid.template cast<double>();

            return true;
        }
    }

private:
    std::unique_ptr<CostFunctor> functor_;
    ceres::Ownership ownership_;
};
} // namespace ceres

#endif
