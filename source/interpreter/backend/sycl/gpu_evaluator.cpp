// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/problem.hpp"
#include "operon/interpreter/backend/sycl/gpu_evaluator.hpp"
#include "operon/interpreter/backend/sycl/population_encoder.hpp"
#include "operon/operators/evaluator.hpp"

namespace Operon::Sycl {

namespace {
auto ToGpuFitType(Operon::ErrorType t) -> GpuFitType
{
    // GpuFitType mirrors ErrorType's enumerator values by design.
    return static_cast<GpuFitType>(static_cast<int>(t));
}
} // namespace

GpuEvaluator::GpuEvaluator(gsl::not_null<Operon::Problem const*> problem,
                            Operon::ErrorMetric error,
                            bool linearScaling)
    : EvaluatorBase(problem)
    , fitType_(ToGpuFitType(error.Type()))
    , scaling_(linearScaling)
    , ctx_(GpuContextCreate(), GpuContextDestroy)
{ }

GpuEvaluator::~GpuEvaluator() = default;

void GpuEvaluator::Prepare(Operon::Span<Operon::Individual const> pop) const
{
    if (pop.empty()) { return; }

    auto const* problem = GetProblem();
    auto const& dataset = *problem->GetDataset();
    auto const  range   = problem->TrainingRange();

    bool const needDataset = (datasetNVars_ == 0);
    auto enc = EncodePopulation(pop, dataset, range, executor_, needDataset);

    // Upload dataset to device on first call or if dimensions changed
    if (enc.NVars != datasetNVars_ || enc.NRows != datasetNRows_) {
        GpuContextUploadDataset(ctx_.get(),
                                enc.DataBuffer.data(),
                                enc.NVars, enc.NRows);
        datasetNVars_ = enc.NVars;
        datasetNRows_ = enc.NRows;
    }

    // Upload target column to device on first call or if nRows changed
    if (enc.NRows != cachedTargetNRows_) {
        auto const target = problem->TargetValues(range);
        std::vector<float> targetF(enc.NRows);
        std::transform(target.begin(), target.end(), targetF.begin(),
                       [](Operon::Scalar v) { return static_cast<float>(v); });
        GpuContextUploadTarget(ctx_.get(), targetF.data(), enc.NRows);
        cachedTargetNRows_ = enc.NRows;
    }

    cachedFitness_.resize(enc.PopSize);

    GpuContextEvaluate(ctx_.get(),
                       enc.Ops.data(),
                       enc.Lengths.data(),
                       enc.PopSize,
                       enc.MaxLen,
                       fitType_,
                       scaling_,
                       cachedFitness_.data());

    treeToSortedIdx_.clear();
    treeToSortedIdx_.reserve(enc.PopSize);
    for (uint32_t si = 0; si < enc.PopSize; ++si) {
        treeToSortedIdx_[&pop[enc.SortedIndices[si]].Genotype] = si;
    }
    prepared_ = true;
}

auto GpuEvaluator::operator()(Operon::RandomGenerator& /*rng*/,
                               Operon::Individual const& ind,
                               Operon::Span<Operon::Scalar> /*buf*/) const -> ReturnType
{
    ++CallCount;

    ENSURE(prepared_);
    auto it = treeToSortedIdx_.find(&ind.Genotype);
    if (it == treeToSortedIdx_.end()) { return { ErrMax }; }

    ++ResidualEvaluations;

    auto const fitness = cachedFitness_[it->second];
    return { std::isfinite(fitness) ? static_cast<Operon::Scalar>(fitness) : ErrMax };
}

auto GpuEvaluator::operator()(Operon::RandomGenerator& rng,
                               Operon::Individual const& ind) const -> ReturnType
{
    std::vector<Operon::Scalar> buf(cachedTargetNRows_);
    return (*this)(rng, ind, { buf.data(), buf.size() });
}

} // namespace Operon::Sycl
