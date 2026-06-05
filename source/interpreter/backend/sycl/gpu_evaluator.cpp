// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <cmath>
#include <vector>

#include "operon/core/problem.hpp"
#include "operon/interpreter/backend/sycl/gpu_evaluator.hpp"
#include "operon/interpreter/backend/sycl/gpu_kernel.hpp"
#include "operon/interpreter/backend/sycl/population_encoder.hpp"
#include "operon/operators/evaluator.hpp"

namespace Operon::Sycl {

GpuEvaluator::GpuEvaluator(gsl::not_null<Operon::Problem const*> problem,
                            Operon::ErrorMetric error,
                            bool linearScaling)
    : EvaluatorBase(problem)
    , error_(error)
    , scaling_(linearScaling)
{ }

GpuEvaluator::~GpuEvaluator() = default;

void GpuEvaluator::Prepare(Operon::Span<Operon::Individual const> pop) const
{
    if (pop.empty()) { return; }

    auto const* problem = GetProblem();
    auto const& dataset = *problem->GetDataset();
    auto const  range   = problem->TrainingRange();

    auto enc = EncodePopulation(pop, dataset, range);

    cachedOutputs_ = RunKernel(enc);
    cachedNRows_   = enc.NRows;

    treeToSortedIdx_.clear();
    treeToSortedIdx_.reserve(enc.PopSize);
    for (uint32_t si = 0; si < enc.PopSize; ++si) {
        treeToSortedIdx_[&pop[enc.SortedIndices[si]].Genotype] = si;
    }
}

auto GpuEvaluator::operator()(Operon::RandomGenerator& /*rng*/,
                               Operon::Individual const& ind,
                               Operon::Span<Operon::Scalar> buf) const -> ReturnType
{
    ++CallCount;

    auto it = treeToSortedIdx_.find(&ind.Genotype);
    if (it == treeToSortedIdx_.end()) {
        return { ErrMax };
    }

    auto const si     = it->second;
    auto const nRows  = cachedNRows_;
    auto const* begin = cachedOutputs_.data() + si * nRows;

    std::transform(begin, begin + nRows, buf.data(),
                   [](float v) { return static_cast<Operon::Scalar>(v); });

    auto const* problem = GetProblem();
    auto const& target  = problem->TargetValues(problem->TrainingRange());

    ++ResidualEvaluations;

    if (scaling_) {
        auto [a, b] = Operon::FitLeastSquares(
            { buf.data(), nRows },
            { target.data(), nRows });

        std::transform(buf.begin(), buf.end(), buf.begin(),
                       [a = static_cast<Operon::Scalar>(a),
                        b = static_cast<Operon::Scalar>(b)](auto v) {
                           return static_cast<Operon::Scalar>(a) +
                                  static_cast<Operon::Scalar>(b) * v;
                       });
    }

    auto err = static_cast<Operon::Scalar>(
        error_({ buf.data(), nRows }, { target.data(), nRows }));

    return { std::isfinite(err) ? err : ErrMax };
}

auto GpuEvaluator::operator()(Operon::RandomGenerator& rng,
                               Operon::Individual const& ind) const -> ReturnType
{
    std::vector<Operon::Scalar> buf(cachedNRows_);
    return (*this)(rng, ind, { buf.data(), buf.size() });
}

} // namespace Operon::Sycl
