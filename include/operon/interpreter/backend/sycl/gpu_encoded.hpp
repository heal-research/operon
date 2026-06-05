// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GPU_ENCODED_HPP
#define OPERON_GPU_ENCODED_HPP

#include <cstdint>
#include <vector>
#include "gpu_op.hpp"

namespace Operon::Sycl {

// Plain data structure produced by EncodePopulation and consumed by RunKernel.
// Contains no Operon-specific types so it can be included in the SYCL TU
// without pulling in dataset.hpp / dispatch.hpp / Eigen.
struct EncodedPopulation {
    std::vector<GpuOp>    Ops;           // [PopSize × MaxLen] individual-major
    std::vector<uint32_t> Lengths;       // per-individual tree length, in sorted order
    std::vector<float>    DataBuffer;    // [NVars × NRows] column-major
    std::vector<int>      SortedIndices; // SortedIndices[sortedPos] = original index
    uint32_t PopSize{0};
    uint32_t MaxLen{0};
    uint32_t NVars{0};
    uint32_t NRows{0};
};

} // namespace Operon::Sycl

#endif // OPERON_GPU_ENCODED_HPP
