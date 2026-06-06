// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GPU_KERNEL_HPP
#define OPERON_GPU_KERNEL_HPP

#include <cstdint>
#include "gpu_op.hpp"

namespace Operon::Sycl {

// Fitness metric type passed to the GPU reduction kernel.
// Values mirror Operon::ErrorType; defined here to keep gpu_kernel.hpp
// free of heavy operon headers (the SYCL TU cannot include Eigen etc.).
enum class GpuFitType : uint8_t {
    SSE  = 0,
    MSE  = 1,
    NMSE = 2,
    RMSE = 3,
    MAE  = 4,
    R2   = 5,
    C2   = 6,
};

// Opaque context owning a persistent SYCL queue and device allocations.
// All SYCL types are hidden inside gpu_kernel.cpp; this header is safe to
// include from plain C++ translation units.
struct GpuContext;

auto GpuContextCreate()              -> GpuContext*;
void GpuContextDestroy(GpuContext*)  noexcept;

// Upload the dataset to device memory. Call once per problem (or whenever the
// dataset changes). colMajor is a [nVars × nRows] column-major float buffer.
void GpuContextUploadDataset(GpuContext*,
                              float const* colMajor,
                              uint32_t nVars, uint32_t nRows);

// Upload the target column to device memory. Call once per training range.
// target is a [nRows] float buffer.
void GpuContextUploadTarget(GpuContext*,
                             float const* target,
                             uint32_t nRows);

// Evaluate the population and compute per-individual fitness on the GPU.
// Only h_fitness[popSize] is downloaded to the host; the raw outputs stay on
// device, eliminating the O(pop × nRows) PCIe transfer.
//
// h_ops      : [popSize × maxLen] GpuOp buffer (individual-major, padded)
// h_lengths  : [popSize] tree lengths in sorted order
// fitType    : fitness metric
// doScale    : apply optimal linear scaling before computing the metric
// h_fitness  : [popSize] output buffer (caller-allocated, host)
void GpuContextEvaluate(GpuContext*,
                         GpuOp    const* ops,
                         uint32_t const* lengths,
                         uint32_t popSize,
                         uint32_t maxLen,
                         GpuFitType fitType,
                         bool doScale,
                         float* fitness);

} // namespace Operon::Sycl

#endif // OPERON_GPU_KERNEL_HPP
