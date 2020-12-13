/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef NNLS_TINY_HPP
#define NNLS_TINY_HPP

#include <Eigen/Core>
#include <ceres/internal/fixed_array.h>
#include <ceres/jet.h>

#include "core/eval.hpp"
#include "nnls/cost_function.hpp"

namespace Operon {
template <typename CostFunctor, typename JetT>
struct TinyCostFunction {
    typedef typename JetT::Scalar Scalar;
    const int Stride = JetT::DIMENSION;

    enum {
        NUM_RESIDUALS = Eigen::Dynamic,
        NUM_PARAMETERS = Eigen::Dynamic,
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TinyCostFunction(const Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range)
        : functor_(tree, dataset, targetValues, range)
    {
        numParameters_ = static_cast<int>(tree.GetCoefficients().size());
        numResiduals_ = static_cast<int>(targetValues.size());

        jacobian_.resize(numResiduals_, numParameters_);
    }

    bool Evaluate(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const
    {
        if (jacobian == nullptr) {
            return functor_(&parameters, residuals);
        }

        // Allocate scratch space for the strided evaluation.
        ceres::internal::FixedArray<JetT, (256 * 7) / sizeof(JetT)> input_jets(numParameters_);
        ceres::internal::FixedArray<JetT, (256 * 7) / sizeof(JetT)> output_jets(numResiduals_);

        for (int j = 0; j < numParameters_; ++j) {
            input_jets[j].a = static_cast<Scalar>(parameters[j]);
        }

        // Evaluate all of the strides. Each stride is a chunk of the derivative to
        // evaluate, typically some size proportional to the size of the SIMD
        // registers of the CPU.
        int num_strides = static_cast<int>(
            std::ceil(static_cast<float>(numParameters_) / static_cast<float>(Stride)));

        int current_derivative_section = 0;
        int current_derivative_section_cursor = 0;

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

            EXPECT(current_derivative_section == 0);

            auto ptr = &input_jets[0];
            if (!functor_(&ptr, &output_jets[0])) {
                return false;
            }

            // Copy the pieces of the jacobians into their final place.
            active_parameter_count = 0;

            current_derivative_section = initial_derivative_section;
            current_derivative_section_cursor = initial_derivative_section_cursor;

            for (int j = 0; j < numParameters_; ++j) {
                if (active_parameter_count < Stride && j >= current_derivative_section_cursor) {
                    for (int k = 0; k < numResiduals_; ++k) {
                        jacobian[k * numParameters_ + j] = output_jets[k].v[active_parameter_count];
                    }
                    ++active_parameter_count;
                    ++current_derivative_section_cursor;
                }
            }

            // Only copy the residuals over once (even though we compute them on
            // every loop).
            if (pass == num_strides - 1) {
                std::transform(output_jets.begin(), output_jets.end(), residuals, [](auto const& jet) { return jet.a; });
            }
        }
        return true;
    }

    // necessary in order to enable usage as a functor inside ceres::DynamicAutodiffCostFunction
    template <typename T>
    bool operator()(T const* const* parameters, T* residuals) const
    {
        return functor_(parameters, residuals);
    }

    // calculates jacobian in row-major order and then flips the storage order
    bool operator()(Scalar const* parameters, Scalar* residuals, Scalar* jacobian) const
    {
        if (!jacobian) {
            return Evaluate(parameters, residuals, nullptr);
        }

        if (!Evaluate(parameters, residuals, jacobian_.data())) {
            return false;
        }
        
        //auto res = Evaluate(parameters, residuals, jacobian);
        //if (!jacobian) return res;

        // The Function object used by TinySolver takes its Jacobian in a
        // column-major layout, and the CostFunction objects use row-major
        // Jacobian matrices. So the following bit of code does the
        // conversion from row-major Jacobians to column-major Jacobians.
        Eigen::Map<Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS>>
            col_major_jacobian(jacobian, NumResiduals(), NumParameters());
        col_major_jacobian = jacobian_;

        return true;
    }

    int NumResiduals() const { return numResiduals_; }
    int NumParameters() const { return numParameters_; }

private:
    CostFunctor functor_;
    mutable Eigen::Matrix<Scalar, NUM_RESIDUALS, NUM_PARAMETERS, Eigen::RowMajor> jacobian_; // row-major
    int numResiduals_;
    int numParameters_;
};
}
#endif
