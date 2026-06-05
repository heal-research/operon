// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GPU_EVALUATOR_HPP
#define OPERON_GPU_EVALUATOR_HPP

#include <unordered_map>
#include <vector>

#include "operon/core/individual.hpp"
#include "operon/core/problem.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"
#include "operon/operators/evaluator.hpp"

namespace Operon::Sycl {

// Evaluator that computes all forward passes on GPU using a SYCL stack machine.
// Prepare() encodes the population, runs the GPU kernel, and caches raw outputs.
// operator() applies linear scaling and the error metric on CPU (cheap per-individual work).
// The L-M optimizer uses its own CPU Interpreter and is unaffected.
class GpuEvaluator : public Operon::EvaluatorBase {
public:
    explicit GpuEvaluator(gsl::not_null<Operon::Problem const*> problem,
                          Operon::ErrorMetric error = Operon::MSE{},
                          bool linearScaling = true);

    GpuEvaluator(GpuEvaluator const&)                    = delete;
    GpuEvaluator(GpuEvaluator&&)                         = delete;
    auto operator=(GpuEvaluator const&) -> GpuEvaluator& = delete;
    auto operator=(GpuEvaluator&&) -> GpuEvaluator&      = delete;
    ~GpuEvaluator() override;

    // Encode population, run GPU kernel, cache results.
    // Must be called before operator() for each new generation.
    void Prepare(Operon::Span<Operon::Individual const> pop) const override;

    auto operator()(Operon::RandomGenerator& rng,
                    Operon::Individual const& ind,
                    Operon::Span<Operon::Scalar> buf) const -> ReturnType override;

    auto operator()(Operon::RandomGenerator& rng,
                    Operon::Individual const& ind) const -> ReturnType override;

private:
    Operon::ErrorMetric error_;
    bool scaling_;

    // [PopSize × NRows] raw model outputs, in sorted order set by Prepare()
    mutable std::vector<float> cachedOutputs_;
    // maps Tree pointer → sorted index in cachedOutputs_
    mutable std::unordered_map<Operon::Tree const*, std::size_t> treeToSortedIdx_;
    mutable uint32_t cachedNRows_{0};
};

} // namespace Operon::Sycl

#endif // OPERON_GPU_EVALUATOR_HPP
