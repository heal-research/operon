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

#ifndef NNLS_HPP
#define NNLS_HPP

#include "core/eval.hpp"

namespace Operon {
// returns an array of optimized parameters
template <bool autodiff = true>
ceres::Solver::Summary Optimize(Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range, size_t iterations = 50, bool writeCoefficients = true, bool report = false)
{
    using ceres::CauchyLoss;
    using ceres::DynamicAutoDiffCostFunction;
    using ceres::DynamicCostFunction;
    using ceres::DynamicNumericDiffCostFunction;
    using ceres::Problem;
    using ceres::Solve;
    using ceres::Solver;

    Solver::Summary summary;
    auto coef = tree.GetCoefficients();
    if (coef.empty()) {
        return summary;
    }
    if (report) {
        fmt::print("x_0: ");
        for (auto c : coef)
            fmt::print("{} ", c);
        fmt::print("\n");
    }

    auto eval = new ResidualEvaluator(tree, dataset, targetValues, range);
    DynamicCostFunction* costFunction;
    if constexpr (autodiff) {
        costFunction = new DynamicAutoDiffCostFunction<ResidualEvaluator>(eval);
    } else {
        costFunction = new DynamicNumericDiffCostFunction(eval);
    }
    costFunction->AddParameterBlock(coef.size());
    costFunction->SetNumResiduals(range.Size());
    //auto lossFunction = new CauchyLoss(0.5); // see http://ceres-solver.org/nnls_tutorial.html#robust-curve-fitting

    Problem problem;
    problem.AddResidualBlock(costFunction, nullptr, coef.data());

    Solver::Options options;
    options.max_num_iterations = iterations - 1; // workaround since for some reason ceres sometimes does 1 more iteration
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
        tree.SetCoefficients(coef);
    }
    return summary;
}

// set up some convenience methods using perfect forwarding
template <typename... Args>
auto OptimizeAutodiff(Args&&... args)
{
    return Optimize<true>(std::forward<Args>(args)...);
}

template <typename... Args>
auto OptimizeNumeric(Args&&... args)
{
    return Optimize<false>(std::forward<Args>(args)...);
}
}
#endif

