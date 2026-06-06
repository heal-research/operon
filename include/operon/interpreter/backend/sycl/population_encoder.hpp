// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GPU_POPULATION_ENCODER_HPP
#define OPERON_GPU_POPULATION_ENCODER_HPP

#include <algorithm>
#include <execution>
#include <numeric>
#include <unordered_map>
#include <vector>

// NOLINTBEGIN(misc-include-cleaner)
#include <taskflow/algorithm/for_each.hpp>
#include <taskflow/core/executor.hpp>
// NOLINTEND(misc-include-cleaner)

#include "operon/core/dataset.hpp"
#include "operon/core/individual.hpp"
#include "operon/core/node.hpp"
#include "operon/core/range.hpp"
#include "operon/core/types.hpp"
#include "gpu_encoded.hpp"

// Verify that the SYCL-safe NodeType mirror (gpu_node_types.hpp) is in sync.
// Variable must be the last enum value; its index encodes the full ordering.
static_assert(static_cast<uint8_t>(Operon::NodeType::Variable) == 31U,
    "gpu_node_types.hpp NodeType is out of sync with core/node.hpp — update both");

namespace Operon::Sycl {

// Encode a population for GPU evaluation.
// Individuals are sorted by tree length to minimise inter-warp load imbalance.
// The dataset range is transposed into a column-major float buffer only when
// needDataset=true; on subsequent calls (dataset already on device) pass false
// to skip the 400 KB allocation/fill.
inline auto EncodePopulation(
    Operon::Span<Operon::Individual const> pop,
    Operon::Dataset const& dataset,
    Operon::Range range,
    tf::Executor& executor,
    bool needDataset = true) -> EncodedPopulation
{
    auto const nRows   = static_cast<uint32_t>(range.Size());
    auto const nVars   = static_cast<uint32_t>(dataset.GetVariables().size());
    auto const popSize = static_cast<uint32_t>(pop.size());

    // hash → column index (nVars ≤ ~100, always L1-cached after first access)
    std::unordered_map<Operon::Hash, uint32_t> hashToIdx;
    hashToIdx.reserve(nVars);
    for (auto const& v : dataset.GetVariables()) {
        hashToIdx[v.Hash] = static_cast<uint32_t>(v.Index);
    }

    // sort individuals by ascending tree length
    std::vector<int> order(popSize);
    std::iota(order.begin(), order.end(), 0);
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return pop[a].Genotype.Length() < pop[b].Genotype.Length();
    });

    uint32_t maxLen{0};
    for (auto idx : order) {
        maxLen = std::max(maxLen, static_cast<uint32_t>(pop[idx].Genotype.Length()));
    }

    // encode ops: padding with GpuNoop; parallelise over sorted individuals
    std::vector<GpuOp>    ops(static_cast<std::size_t>(popSize) * maxLen,
                              GpuOp{GpuNoop, 0U, 0U, 0.0F});
    std::vector<uint32_t> lengths(popSize);

    tf::Taskflow tf;
    tf.for_each_index(0U, popSize, 1U, [&](uint32_t si) {
        auto const& nodes = pop[order[si]].Genotype.Nodes();
        lengths[si] = static_cast<uint32_t>(nodes.size());

        for (uint32_t ni = 0; ni < lengths[si]; ++ni) {
            auto const& n = nodes[ni];
            GpuOp op{};
            op.Type  = static_cast<uint8_t>(n.Type);
            op.Arity = static_cast<uint8_t>(n.Arity);

            if (n.IsVariable()) {
                op.VarIdx = hashToIdx.at(n.HashValue);
                op.Value  = static_cast<float>(n.Value);
            } else if (n.IsConstant()) {
                op.VarIdx = 0U;
                op.Value  = static_cast<float>(n.Value);
            }
            ops[(static_cast<std::size_t>(si) * maxLen) + ni] = op;
        }
    });
    executor.run(tf).wait();

    // encode dataset: column-major [nVars × nRows] — skipped when already on device
    std::vector<float> dataBuffer;
    if (needDataset) {
        dataBuffer.resize(static_cast<std::size_t>(nVars) * nRows);
        for (uint32_t vi = 0; vi < nVars; ++vi) {
            auto col = dataset.GetValues(static_cast<int64_t>(vi)).subspan(range.Start(), nRows);
            for (uint32_t ri = 0; ri < nRows; ++ri) {
                dataBuffer[(static_cast<std::size_t>(vi) * nRows) + ri] = static_cast<float>(col[ri]);
            }
        }
    }

    return EncodedPopulation{
        .Ops           = std::move(ops),
        .Lengths       = std::move(lengths),
        .DataBuffer    = std::move(dataBuffer),
        .SortedIndices = std::move(order),
        .PopSize       = popSize,
        .MaxLen        = maxLen,
        .NVars         = nVars,
        .NRows         = nRows,
    };
}

// Convenience overload for tests that don't have a tf::Executor handy.
inline auto EncodePopulation(
    Operon::Span<Operon::Individual const> pop,
    Operon::Dataset const& dataset,
    Operon::Range range) -> EncodedPopulation
{
    tf::Executor ex;
    return EncodePopulation(pop, dataset, range, ex, /*needDataset=*/true);
}

} // namespace Operon::Sycl

#endif // OPERON_GPU_POPULATION_ENCODER_HPP
