// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_HIP_KERNEL_HPP
#define OPERON_HIP_KERNEL_HPP

#include <cstddef>
#include <cstdint>

namespace Operon::PopulationOptimization::Hip::detail {

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

struct Runtime;

[[nodiscard]] auto Create() -> Runtime*;
void Destroy(Runtime*) noexcept;
void Upload(Runtime*, DeviceNode const*, std::size_t, DeviceTreeRange const*, std::size_t);
void UpdateCoefficients(Runtime*, float const* coefficients, std::size_t coefficientCount);
void Evaluate(Runtime*, float const* columns, std::size_t variableCount, std::size_t rowCount,
              float const* coefficients, std::size_t coefficientCount, float* output);
void EvaluateResident(Runtime*, float const* columns, std::size_t variableCount, std::size_t rowCount, float* output);
void GaussianCosts(Runtime*, float const* columns, std::size_t variableCount, std::size_t rowCount,
                   float const* target, float const* weights, bool hasWeights, float* costs, uint8_t* valid);
void JacRev(Runtime*, float const* columns, std::size_t variableCount, std::size_t rowCount,
            float const* coefficients, std::size_t coefficientCount, float* jacobian);
void OptimizeGaussian(Runtime*, float const* columns, std::size_t variableCount, std::size_t rowCount,
                      float const* target, float const* weights, bool hasWeights,
                      float const* coefficients, std::size_t coefficientCount, uint32_t maxIterations,
                      float* finalCoefficients, float* initialCosts, float* finalCosts,
                      uint8_t* status, uint32_t* iterations, uint32_t* acceptedSteps);
void NormalEquations(Runtime*, float const* columns, std::size_t variableCount, std::size_t rowCount,
                     float const* target, float const* weights, bool hasWeights,
                     float const* coefficients, std::size_t coefficientCount,
                     float* normal, float* gradient, float* costs, uint8_t* valid);

} // namespace Operon::PopulationOptimization::Hip::detail

#endif // OPERON_HIP_KERNEL_HPP
