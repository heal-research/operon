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

#include "core/nnls.hpp"
#include <ceres/tiny_solver.h>

namespace Operon {

struct TinyCostFunction {
    typedef double Scalar;

    enum {
        NUM_RESIDUALS  = Eigen::Dynamic,
        NUM_PARAMETERS = Eigen::Dynamic,
    };

    EIGEN_MAKE_ALIGNED_OPERATOR_NEW

    TinyCostFunction(const Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range) 
        : costFunction(new ResidualEvaluator(tree, dataset, targetValues, range))
    {
        int nParameters = tree.GetCoefficients().size();
        int nResiduals = targetValues.size();
        costFunction.AddParameterBlock(nParameters);
        costFunction.SetNumResiduals(nResiduals);

        jacobian_.resize(nResiduals, nParameters); 
    }

    bool operator()(const double* parameters, double* residuals, double* jacobian) const {
        if (!jacobian) {
            return costFunction.Evaluate(&parameters, residuals, nullptr);
        }

        double* jacobians[1] = { jacobian_.data() };
        if (!costFunction.Evaluate(&parameters, residuals, jacobians)) {
            return false;
        }

        // The Function object used by TinySolver takes its Jacobian in a
        // column-major layout, and the CostFunction objects use row-major
        // Jacobian matrices. So the following bit of code does the
        // conversion from row-major Jacobians to column-major Jacobians.
        Eigen::Map<Eigen::Matrix<double, NUM_RESIDUALS, NUM_PARAMETERS>>
            col_major_jacobian(jacobian, NumResiduals(), NumParameters());
        col_major_jacobian = jacobian_;

        return true;
    }

    int NumResiduals() const { return costFunction.num_residuals(); }
    int NumParameters() const { return costFunction.parameter_block_sizes()[0]; }

    private:
        ceres::DynamicAutoDiffCostFunction<ResidualEvaluator> costFunction;
        mutable Eigen::Matrix<double, NUM_RESIDUALS, NUM_PARAMETERS, Eigen::RowMajor> jacobian_; // row-major
};
}

#endif

