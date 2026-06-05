// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_GPU_OP_HPP
#define OPERON_GPU_OP_HPP

#include <cstdint>

namespace Operon::Sycl {

// A single instruction in the GPU stack machine.
// Leaves: Arity=0. Variable uses VarIdx + Value (coefficient). Constant uses Value only.
// Operators: Arity matches node arity; VarIdx and Value are unused.
struct GpuOp {
    uint8_t  Type;    // NodeType cast to uint8_t; GpuNoop = 0xFF (padding)
    uint8_t  Arity;
    uint32_t VarIdx;  // dataset column index (Variable only)
    float    Value;   // coefficient (Variable) or constant value (Constant)
};

static constexpr uint8_t GpuNoop = 0xFFU;

} // namespace Operon::Sycl

#endif // OPERON_GPU_OP_HPP
