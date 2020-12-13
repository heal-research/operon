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

#ifndef OPERON_NNLS_HPP
#define OPERON_NNLS_HPP

#include "nnls/cost_function.hpp"
#include "nnls/tiny_cost_function.hpp"
#include "nnls/tiny_solver.hpp"
#include <ceres/dynamic_autodiff_cost_function.h>

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
    std::vector<double> coef;
    for (auto const& node : tree.Nodes()) {
        if (node.IsLeaf()) coef.push_back(node.Value);
    }

    if (coef.empty()) {
        return summary;
    }

    if (report) {
        fmt::print("x_0: ");
        for (auto c : coef)
            fmt::print("{} ", c);
        fmt::print("\n");
    }

    DynamicCostFunction* costFunction;
    if constexpr (autodiff) {
        using CostFunctor = TinyCostFunction<ResidualEvaluator, Dual>;
        auto eval = new CostFunctor(tree, dataset, targetValues, range);
        costFunction = new Operon::DynamicAutoDiffCostFunction<CostFunctor, Dual::DIMENSION>(eval);
        //auto eval = new ResidualEvaluator(tree, dataset, targetValues, range);
        //costFunction = new DynamicAutoDiffCostFunction<ResidualEvaluator, Dual::DIMENSION>(eval);
    } else {
        auto eval = new ResidualEvaluator(tree, dataset, targetValues, range);
        costFunction = new DynamicNumericDiffCostFunction(eval);
    }

    int nParameters = static_cast<int>(coef.size());
    int nResiduals = static_cast<int>(range.Size());
    costFunction->AddParameterBlock(nParameters);
    costFunction->SetNumResiduals(nResiduals);
    //auto lossFunction = new CauchyLoss(0.5); // see http://ceres-solver.org/nnls_tutorial.html#robust-curve-fitting

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

template <bool autodiff = true>
ceres::Solver::Summary OptimizeTiny(Tree& tree, const Dataset& dataset, const gsl::span<const Operon::Scalar> targetValues, const Range range, size_t iterations = 50, bool writeCoefficients = true, bool = false /* ignored */)
{
    ceres::TinySolver<TinyCostFunction<ResidualEvaluator, Dual>> solver;
    solver.options.max_num_iterations = static_cast<int>(iterations);

    TinyCostFunction<ResidualEvaluator, Dual> cf(tree, dataset, targetValues, range);
    auto x0 = tree.GetCoefficients();
    decltype(solver)::Parameters params = Eigen::Map<Eigen::Matrix<Operon::Scalar, Eigen::Dynamic, 1>>(x0.data(), x0.size()).cast<Scalar>();
    solver.Solve(cf, &params);

    if (writeCoefficients) {
        tree.SetCoefficients({ params.data(), x0.size() });
    }

    ceres::Solver::Summary summary;
    summary.initial_cost = solver.summary.initial_cost;
    summary.final_cost = solver.summary.final_cost;
    summary.iterations.resize(solver.summary.iterations);
    return summary;
}

// set up some convenience methods using perfect forwarding
template <typename... Args>
auto OptimizeTinyAutodiff(Args&&... args)
{
    return OptimizeTiny<true>(std::forward<Args>(args)...);
}

template <typename... Args>
auto OptimizeTinyNumeric(Args&&... args)
{
    return OptimizeTiny<false>(std::forward<Args>(args)...);
}
}
#endif

