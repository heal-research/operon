// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_OPTIMIZER_COST_FUNCTION_HPP
#define OPERON_OPTIMIZER_COST_FUNCTION_HPP

#ifdef HAVE_CERES
#ifndef CERES_EXPORT
#define CERES_EXPORT OPERON_EXPORT
#endif

#include <ceres/ceres.h>
#include "operon/core/contracts.hpp"

namespace Operon {
template <typename CostFunctor>
struct DynamicCostFunction final : public ceres::DynamicCostFunction {
    using Scalar = typename CostFunctor::Scalar;

    explicit DynamicCostFunction(CostFunctor const& cf)
        : cf_(cf)
    {
        static_assert(CostFunctor::Storage == Eigen::RowMajor, "Operon::DynamicCostFunction requires row-major storage.");
        this->mutable_parameter_block_sizes()->push_back(cf_.NumParameters());
        set_num_residuals(cf_.NumResiduals());

        ENSURE(cf_.NumParameters() > 0);
        ENSURE(cf_.NumResiduals() > 0);
    }

    // required by ceres
    auto Evaluate(double const* const* parameters, double* residuals, double** jacobians) const -> bool override
    {
        EXPECT(parameters != nullptr && parameters[0] != nullptr);

        if constexpr (std::is_same_v<Scalar, double>) {
            if (jacobians != nullptr) { EXPECT(jacobians[0] != nullptr); }
            return cf_(parameters[0], residuals, jacobians == nullptr ? nullptr : jacobians[0]);
        } else {
            // we need to make a copy
            int const nr = this->num_residuals();
            int const np = this->parameter_block_sizes().front();

            ENSURE(nr > 0);
            ENSURE(np > 0);

            Eigen::Map<const Eigen::Matrix<double, -1, 1>> pMap(parameters[0], np);
            Eigen::Map<Eigen::Matrix<double, -1, 1>> rMap(residuals, nr);

            Eigen::Matrix<Scalar, -1, 1> param = pMap.cast<Scalar>();
            Eigen::Matrix<Scalar, -1, 1> resid(nr);

            if (jacobians == nullptr) {
                auto success = cf_(param.data(), resid.data(), nullptr);
                if (!success) {
                    return false;
                }
            } else {
                Eigen::Matrix<Scalar, -1, -1, CostFunctor::Storage> jacob(nr, np);
                auto success = cf_(param.data(), resid.data(), jacob.data());
                if (!success) {
                    return false;
                }

                Eigen::Map<Eigen::Matrix<double, -1, -1, CostFunctor::Storage>> jMap(jacobians[0], nr, np);
                jMap = jacob.template cast<double>();
            }
            rMap = resid.template cast<double>();

            return true;
        }
    }

    void AddParameterBlock(int /*size*/) override
    {
        throw std::runtime_error("This method should not be used.");
    }

    void SetNumResiduals(int /*num_residuals*/) override
    {
        throw std::runtime_error("This method should not be used.");
    }

    auto Functor() -> CostFunctor& { return cf_; }
    [[nodiscard]] auto Functor() const -> CostFunctor const& { return cf_; }

private:
    CostFunctor cf_;
};
} // namespace Operon
#endif
#endif
