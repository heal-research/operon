#ifndef OPERON_DYNAMIC_AUTODIFF_COST_FUNCTION_HPP
#define OPERON_DYNAMIC_AUTODIFF_COST_FUNCTION_HPP

#include <cmath>
#include <memory>
#include <numeric>
#include <type_traits>
#include <vector>

#include <ceres/dynamic_cost_function.h>
#include <ceres/internal/fixed_array.h>
#include <ceres/types.h>

#include "core/eval.hpp"
#include "core/types.hpp"

namespace Operon {
// this cost function is adapted to work with both solvers from Ceres: the normal one and the tiny solver
// for this, a number of template parameters are necessary:
// - the CostFunctor is the actual functor for computing the residuals
// - the JetT type represents a dual number, the user can specify the type for the Scalar part (float, double) and the Stride (Ceres-specific)
// - the StorageOrder specifies the format of the jacobian (row-major for the big Ceres solver, column-major for the tiny solver)
template <typename CostFunctor, typename JetT, int StorageOrder = Eigen::RowMajor>
struct DynamicAutoDiffCostFunction final : public ceres::DynamicCostFunction {
    using Scalar = typename JetT::Scalar;
    const int Stride = JetT::DIMENSION;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,
        NUM_PARAMETERS = Eigen::Dynamic,
    };

    DynamicAutoDiffCostFunction(const Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range)
        : functor_(tree, dataset, targetValues, range)
    {
        numParameters_ = static_cast<int>(tree.GetCoefficients().size());
        numResiduals_ = static_cast<int>(targetValues.size());

        mutable_parameter_block_sizes()->push_back(numParameters_);
        set_num_residuals(numResiduals_);
    }

    bool Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const
    {
        if (jacobian == nullptr) {
            return functor_(&parameters, residuals);
        }

        // Allocate scratch space for the strided evaluation.
        ceres::internal::FixedArray<JetT, (256 * 7) / sizeof(JetT)> input_jets(numParameters_);
        ceres::internal::FixedArray<JetT, (256 * 7) / sizeof(JetT)> output_jets(numResiduals_);

        // Evaluate all of the strides. Each stride is a chunk of the derivative to
        // evaluate, typically some size proportional to the size of the SIMD
        // registers of the CPU.
        int num_strides = static_cast<int>(
            std::ceil(static_cast<float>(numParameters_) / static_cast<float>(Stride)));

        auto ptr = &input_jets[0];

        for (int j = 0; j < numParameters_; ++j) {
            input_jets[j].a = static_cast<Scalar>(parameters[j]);
        }

        int current_derivative_section = 0;
        int current_derivative_section_cursor = 0;

        Eigen::Map<Eigen::Matrix<typename JetT::Scalar, -1, -1, StorageOrder>> jMap(jacobian, numResiduals_, numParameters_);

        for (int pass = 0; pass < num_strides; ++pass) {
            // Set most of the jet components to zero, except for
            // non-constant #Stride parameters.
            const int initial_derivative_section = current_derivative_section;
            const int initial_derivative_section_cursor = current_derivative_section_cursor;

            int active_parameter_count = 0;
            for (int j = 0; j < numParameters_; ++j) {
                input_jets[j].v.setZero();
                if (active_parameter_count < Stride && j >= current_derivative_section_cursor) {
                    input_jets[j].v[active_parameter_count] = 1.0;
                    ++active_parameter_count;
                }
            }

            if (!functor_(&ptr, &output_jets[0])) {
                return false;
            }

            active_parameter_count = 0;
            current_derivative_section = initial_derivative_section;
            current_derivative_section_cursor = initial_derivative_section_cursor;

            // Copy the pieces of the jacobians into their final place.
            for (int j = current_derivative_section_cursor; j < numParameters_; ++j) {
                if (active_parameter_count < Stride) {
                    for (int k = 0; k < numResiduals_; ++k) {
                        jMap(k, j) = output_jets[k].v[active_parameter_count];
                    }
                    ++active_parameter_count;
                    ++current_derivative_section_cursor;
                }
            }

            // Only copy the residuals over once (even though we compute them on every loop).
            if (pass == num_strides - 1) {
                std::transform(output_jets.begin(), output_jets.end(), residuals, [](auto const& jet) { return jet.a; });
            }
        }
        return true;
    }

    // this method gets called by the Ceres solver
    bool Evaluate(double const* const* parameters, double* residuals, double** jacobians) const override
    {
        EXPECT(parameters != nullptr);
        if constexpr(std::is_same_v<Scalar, double>) {
            return Evaluate(parameters[0], residuals, jacobians != nullptr ? jacobians[0] : nullptr);
        } else {
            // we need to make a copy
            int numResiduals  = NumResiduals();
            int numParameters = NumParameters();

            Eigen::Map<const Eigen::Matrix<double, -1, 1>> pMap(parameters[0], numParameters);
            Eigen::Map<Eigen::Matrix<double, -1, 1>> rMap(residuals, numResiduals);

            Eigen::Matrix<Scalar, -1, 1> param = pMap.cast<Scalar>(); 
            Eigen::Matrix<Scalar, -1, 1> resid(numResiduals); 

            bool success;
            if (jacobians == nullptr) {
                success = Evaluate(param.data(), resid.data(), nullptr);
                if (!success) { return false; }
            } else {
                Eigen::Map<Eigen::Matrix<double, -1, -1>> jMap(jacobians[0], numResiduals, numParameters);
                Eigen::Matrix<Scalar, -1, -1> jacob(numResiduals, numParameters);

                success = Evaluate(param.data(), resid.data(), jacob.data());
                if (!success) { return false; }

                jMap = jacob.template cast<double>();
            }
            rMap = resid.template cast<double>();

            return true;
        }
    }

    // this method gets called by the Ceres tiny solver
    bool operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const
    {
        return Evaluate(parameters, residuals, jacobian);
    }

    void AddParameterBlock(int) override {
        throw new std::runtime_error("This method should not be used.");
    }

    void SetNumResiduals(int) override {
        throw new std::runtime_error("This method should not be used.");
    }

    // required by tiny solver
    int NumResiduals() const { return num_residuals(); }
    int NumParameters() const { return parameter_block_sizes().front(); }

private:
    CostFunctor functor_;
    int numResiduals_;
    int numParameters_;
};



} // namespace ceres

#endif
