// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GPU_EVALUATOR_HPP
#define OPERON_GPU_EVALUATOR_HPP

#include <memory>
#include <vector>

// NOLINTBEGIN(misc-include-cleaner)
#include <taskflow/core/executor.hpp>
// NOLINTEND(misc-include-cleaner)

#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/evaluator.hpp"
#include "gpu_kernel.hpp"

namespace Operon::Sycl {

// Evaluator that computes all forward passes on GPU using a SYCL stack machine.
// Prepare() encodes the population, uploads ops to device, runs the eval kernel,
// then runs the fitness reduction kernel on GPU. Only O(popSize) fitness scalars
// are downloaded to host — eliminating the O(pop × nRows) PCIe transfer bottleneck.
// operator() returns the precomputed fitness for a given individual.
// The L-M optimizer uses its own CPU Interpreter and is unaffected.
class GpuEvaluator : public Operon::EvaluatorBase {
public:
    explicit GpuEvaluator(gsl::not_null<Operon::Problem const*> problem,
                          Operon::ErrorMetric error = Operon::R2{},
                          bool linearScaling = true);

    GpuEvaluator(GpuEvaluator const&)                    = delete;
    GpuEvaluator(GpuEvaluator&&)                         = delete;
    auto operator=(GpuEvaluator const&) -> GpuEvaluator& = delete;
    auto operator=(GpuEvaluator&&) -> GpuEvaluator&      = delete;
    ~GpuEvaluator() override;

    void Prepare(Operon::Span<Operon::Individual const> pop) const override;

    [[nodiscard]] auto IsBatch() const -> bool override { return true; }

    auto operator()(Operon::RandomGenerator& rng,
                    Operon::Individual const& ind,
                    Operon::Span<Operon::Scalar> buf) const -> ReturnType override;

    auto operator()(Operon::RandomGenerator& rng,
                    Operon::Individual const& ind) const -> ReturnType override;

private:
    GpuFitType fitType_;
    bool scaling_;

    // Persistent GPU context: queue + device allocations
    mutable std::unique_ptr<GpuContext, void(*)(GpuContext*)> ctx_;

    // Per-individual precomputed fitness, set by Prepare() (O(popSize) floats)
    mutable std::vector<float> cachedFitness_;
    // Maps Tree pointer → sorted index in cachedFitness_
    mutable Operon::Map<Operon::Tree const*, std::size_t> treeToSortedIdx_;
    // tracks whether the target has been uploaded to device
    mutable uint32_t cachedTargetNRows_{0};
    // tracks whether the dataset has been uploaded to device
    mutable uint32_t datasetNVars_{0};
    mutable uint32_t datasetNRows_{0};
    // set to true after the first Prepare() call; operator() asserts if still false
    mutable bool prepared_{false};
    // Executor for parallel EncodePopulation
    mutable tf::Executor executor_;
};

} // namespace Operon::Sycl

#endif // OPERON_GPU_EVALUATOR_HPP
