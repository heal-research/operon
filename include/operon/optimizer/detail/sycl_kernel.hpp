// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_SYCL_KERNEL_HPP
#define OPERON_SYCL_KERNEL_HPP

#include <cstddef>
#include <cstdint>

namespace Operon::PopulationOptimization::Sycl::detail {

enum class Opcode : uint8_t { Constant, Variable, Ref, Add, Sub, Mul, Div, Square, Exp, Log, Sin, Cos };

struct DeviceNode {
    float Value;
    uint32_t Operand;
    uint32_t Coefficient;
    uint16_t Arity;
    uint16_t Length;
    Opcode Op;
    uint8_t Optimize;
};

struct DeviceTreeRange {
    uint32_t NodeOffset;
    uint32_t NodeCount;
    uint32_t CoefficientOffset;
    uint32_t CoefficientCount;
};


struct KernelArgs {
    DeviceNode const* Nodes;
    DeviceTreeRange const* Trees;
    float const* Columns;
    float const* Target;
    float const* Weights;
    float* Coefficients;
    uint8_t* Status;
    uint32_t* Iterations;
    uint32_t* Accepted;
    std::size_t TreeCount;
    std::size_t RowCount;
    uint32_t MaxIterations;
    bool HasWeights;
};

struct EvaluationArgs {
    DeviceNode const* Nodes;
    DeviceTreeRange const* Trees;
    float const* Columns;
    float const* Coefficients;
    float* Output;
    std::size_t TreeCount;
    std::size_t RowCount;
};

// `queue` is a `sycl::queue*`, intentionally opaque so host orchestration
// remains a normal C++ translation unit. Only the compact kernel TU is passed
// through AdaptiveCpp's device compiler.
void RunKernel(void* queue, KernelArgs args);
void RunEvaluate(void* queue, EvaluationArgs args);

} // namespace Operon::PopulationOptimization::Sycl::detail

#endif // OPERON_SYCL_KERNEL_HPP
