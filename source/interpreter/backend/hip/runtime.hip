// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <hip/hip_runtime.h>

#include <array>
#include <cmath>
#include <cstddef>
#include <span>
#include <stdexcept>
#include <vector>
#include "operon/optimizer/detail/hip_kernel.hpp"

namespace Operon::PopulationOptimization::Hip::detail {

struct Runtime {
    hipStream_t Stream{};
    DeviceNode* Nodes{};
    DeviceTreeRange* Trees{};
    float* Columns{};
    float* Coefficients{};
    float* Output{};
    float* Jacobian{};
    float* Target{};
    float* Weights{};
    float* Residual{};
    float* TrialResidual{};
    float* Normal{};
    float* Gradient{};
    float* Trial{};
    float* InitialCosts{};
    float* Costs{};
    float* TrialCosts{};
    float* Damping{};
    uint8_t* State{};
    uint8_t* TrialValid{};
    uint8_t* Status{};
    uint32_t* Iterations{};
    uint32_t* Accepted{};
    float* CostPartials{};
    uint8_t* CostPartialValid{};
    std::array<uint32_t*, 3> Buckets{};
    std::array<std::size_t, 3> BucketCounts{};
    std::array<std::size_t, 3> BucketCapacities{};
    std::size_t TreeCount{};
    std::size_t CoefficientCount{};
    std::size_t NodesCapacity{};
    std::size_t TreesCapacity{};
    std::size_t ColumnsCapacity{};
    std::size_t CoefficientsCapacity{};
    std::size_t OutputCapacity{};
    std::size_t JacobianCapacity{};
    std::size_t TargetCapacity{};
    std::size_t WeightsCapacity{};
    std::size_t ResidualCapacity{};
    std::size_t TrialResidualCapacity{};
    std::size_t NormalCapacity{};
    std::size_t GradientCapacity{};
    std::size_t TrialCapacity{};
    std::size_t InitialCostsCapacity{};
    std::size_t CostsCapacity{};
    std::size_t TrialCostsCapacity{};
    std::size_t DampingCapacity{};
    std::size_t CostPartialsCapacity{};
    std::size_t CostPartialValidCapacity{};
    std::size_t StateCapacity{};
    std::size_t TrialValidCapacity{};
    std::size_t StatusCapacity{};
    std::size_t IterationsCapacity{};
    std::size_t AcceptedCapacity{};
};

namespace {
constexpr auto WaveSize = 32U;
constexpr auto MaxNodes = 128U;
constexpr auto MaxParameters = 16U;

void Check(hipError_t error)
{
    if (error != hipSuccess) { throw std::runtime_error(hipGetErrorString(error)); }
}

template<typename T>
void Reserve(T*& pointer, std::size_t& capacity, std::size_t required)
{
    if (required <= capacity) { return; }
    if (pointer != nullptr) { Check(hipFree(pointer)); }
    pointer = nullptr;
    capacity = 0;
    Check(hipMalloc(&pointer, required * sizeof(T)));
    capacity = required;
}

__device__ bool IsNodeBucket(DeviceTreeRange tree, uint32_t minNodes, uint32_t maxNodes)
{
    return tree.NodeCount >= minNodes && tree.NodeCount <= maxNodes;
}

__device__ bool IsCoefficientBucket(DeviceTreeRange tree, uint32_t lowerExclusive, uint32_t upperInclusive)
{
    return tree.CoefficientCount > lowerExclusive && tree.CoefficientCount <= upperInclusive;
}

template<uint32_t TapeCapacity>
__device__ float EvaluateOne(DeviceNode const* nodes, DeviceTreeRange tree, float const* columns,
                             std::size_t rowCount, std::size_t row, float const* coefficients,
                             float* tape)
{
    for (uint32_t local = 0; local < tree.NodeCount; ++local) {
        auto const node = nodes[tree.NodeOffset + local];
        float value{};
        switch (node.Op) {
        case Opcode::Constant:
            value = node.Optimize ? coefficients[tree.CoefficientOffset + node.Coefficient] : node.Value;
            break;
        case Opcode::Variable:
            value = columns[static_cast<std::size_t>(node.Operand) * rowCount + row]
                * (node.Optimize ? coefficients[tree.CoefficientOffset + node.Coefficient] : node.Value);
            break;
        case Opcode::Ref:
            value = node.Operand < local ? tape[node.Operand] : NAN;
            break;
        case Opcode::Square:
            if (node.Arity != 1 || local == 0) { return NAN; }
            value = tape[local - 1] * tape[local - 1];
            break;
        case Opcode::Exp:
            if (node.Arity != 1 || local == 0) { return NAN; }
            value = expf(tape[local - 1]);
            break;
        case Opcode::Log:
            if (node.Arity != 1 || local == 0) { return NAN; }
            value = logf(tape[local - 1]);
            break;
        case Opcode::Sin:
            if (node.Arity != 1 || local == 0) { return NAN; }
            value = sinf(tape[local - 1]);
            break;
        case Opcode::Cos:
            if (node.Arity != 1 || local == 0) { return NAN; }
            value = cosf(tape[local - 1]);
            break;
        default: {
            if (node.Arity != 2 || local < 2) { return NAN; }
            auto const lhsRoot = local - 1;
            auto const lhsLength = nodes[tree.NodeOffset + lhsRoot].Length;
            if (lhsLength > local - 2) { return NAN; }
            auto const rhsRoot = local - 2 - lhsLength;
            auto const lhs = tape[lhsRoot];
            auto const rhs = tape[rhsRoot];
            switch (node.Op) {
            case Opcode::Add: value = lhs + rhs; break;
            case Opcode::Sub: value = lhs - rhs; break;
            case Opcode::Mul: value = lhs * rhs; break;
            case Opcode::Div: value = lhs / rhs; break;
            default: return NAN;
            }
            break;
        }
        }
        tape[local] = value;
    }
    return tape[tree.NodeCount - 1];
}

template<uint32_t TapeCapacity>
__device__ bool DifferentiateOne(DeviceNode const* nodes, DeviceTreeRange tree, float const* columns,
                                 std::size_t rowCount, std::size_t row, float const* tape,
                                 float* adjoint, float* gradient)
{
    for (uint32_t local = 0; local < tree.NodeCount; ++local) { adjoint[local] = 0.0F; }
    for (uint32_t parameter = 0; parameter < tree.CoefficientCount; ++parameter) { gradient[parameter] = 0.0F; }
    adjoint[tree.NodeCount - 1] = 1.0F;
    for (int local = static_cast<int>(tree.NodeCount) - 1; local >= 0; --local) {
        auto const node = nodes[tree.NodeOffset + static_cast<uint32_t>(local)];
        auto const a = adjoint[local];
        if (node.Optimize) {
            if (node.Coefficient >= tree.CoefficientCount) { return false; }
            auto const factor = node.Op == Opcode::Variable
                ? columns[static_cast<std::size_t>(node.Operand) * rowCount + row] : 1.0F;
            gradient[node.Coefficient] = a * factor;
        }
        if (node.Op == Opcode::Ref) {
            if (node.Operand >= static_cast<uint32_t>(local)) { return false; }
            adjoint[node.Operand] += a;
            continue;
        }
        if (node.Op == Opcode::Square || node.Op == Opcode::Exp || node.Op == Opcode::Log || node.Op == Opcode::Sin || node.Op == Opcode::Cos) {
            if (local == 0) { return false; }
            auto const child = static_cast<uint32_t>(local) - 1;
            auto const x = tape[child];
            auto const derivative = node.Op == Opcode::Square ? 2.0F * x
                                  : node.Op == Opcode::Exp ? tape[static_cast<uint32_t>(local)]
                                  : node.Op == Opcode::Log ? 1.0F / x
                                  : node.Op == Opcode::Sin ? cosf(x)
                                  : -sinf(x);
            adjoint[child] += a * derivative;
            continue;
        }
        if (node.Op < Opcode::Add) { continue; }
        if (local < 2) { return false; }
        auto const lhsRoot = static_cast<uint32_t>(local) - 1;
        auto const lhsLength = nodes[tree.NodeOffset + lhsRoot].Length;
        if (lhsLength > static_cast<uint32_t>(local) - 2) { return false; }
        auto const rhsRoot = static_cast<uint32_t>(local) - 2 - lhsLength;
        auto const lhs = tape[lhsRoot];
        auto const rhs = tape[rhsRoot];
        switch (node.Op) {
        case Opcode::Add: adjoint[lhsRoot] += a; adjoint[rhsRoot] += a; break;
        case Opcode::Sub: adjoint[lhsRoot] += a; adjoint[rhsRoot] -= a; break;
        case Opcode::Mul: adjoint[lhsRoot] += a * rhs; adjoint[rhsRoot] += a * lhs; break;
        case Opcode::Div: adjoint[lhsRoot] += a / rhs; adjoint[rhsRoot] -= a * lhs / (rhs * rhs); break;
        default: return false;
        }
    }
    return true;
}

template<uint32_t MinNodes, uint32_t NodeCapacity>
__global__ void EvaluateKernel(DeviceNode const* nodes, DeviceTreeRange const* trees,
                               std::size_t treeCount, float const* columns, std::size_t rowCount,
                               float const* coefficients, float* output)
{
    auto const treeIndex = static_cast<std::size_t>(blockIdx.x);
    auto const row = static_cast<std::size_t>(blockIdx.y) * blockDim.x + threadIdx.x;
    if (treeIndex >= treeCount || row >= rowCount) { return; }
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity)) { return; }
    float tape[NodeCapacity];
    output[treeIndex * rowCount + row] = EvaluateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, coefficients, tape);
}

template<uint32_t MinNodes, uint32_t NodeCapacity>
__global__ void JacRevKernel(DeviceNode const* nodes, DeviceTreeRange const* trees,
                             std::size_t treeCount, float const* columns, std::size_t rowCount,
                             float const* coefficients, float* jacobian)
{
    auto const treeIndex = static_cast<std::size_t>(blockIdx.x);
    if (treeIndex >= treeCount) { return; }
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity)) { return; }
    for (auto row = static_cast<std::size_t>(threadIdx.x); row < rowCount; row += blockDim.x) {
        float tape[NodeCapacity];
        float adjoint[NodeCapacity];
        float gradient[MaxParameters];
        auto const value = EvaluateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, coefficients, tape);
        if (!isfinite(value) || !DifferentiateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, tape, adjoint, gradient)) {
            for (uint32_t parameter = 0; parameter < tree.CoefficientCount; ++parameter) {
                jacobian[static_cast<std::size_t>(tree.CoefficientOffset + parameter) * rowCount + row] = NAN;
            }
            continue;
        }
        for (uint32_t parameter = 0; parameter < tree.CoefficientCount; ++parameter) {
            jacobian[static_cast<std::size_t>(tree.CoefficientOffset + parameter) * rowCount + row] = gradient[parameter];
        }
    }
}

template<uint32_t MinNodes, uint32_t NodeCapacity>
__global__ void ResidualKernel(DeviceNode const* nodes, DeviceTreeRange const* trees, uint32_t const* indices,
                               std::size_t treeCount, float const* columns, std::size_t rowCount,
                               float const* target, float const* coefficients, float* residual)
{
    auto const bucketIndex = static_cast<std::size_t>(blockIdx.x);
    if (bucketIndex >= treeCount) { return; }
    auto const treeIndex = static_cast<std::size_t>(indices[bucketIndex]);
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity)) { return; }
    for (auto row = static_cast<std::size_t>(threadIdx.x); row < rowCount; row += blockDim.x) {
        float tape[NodeCapacity];
        auto const value = EvaluateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, coefficients, tape);
        residual[treeIndex * rowCount + row] = value - target[row];
    }
}

template<uint32_t MinNodes, uint32_t NodeCapacity>
__global__ void JacResidualKernel(DeviceNode const* nodes, DeviceTreeRange const* trees,
                                  std::size_t treeCount, float const* columns, std::size_t rowCount,
                                  float const* target, float const* coefficients, float* residual,
                                  float* jacobian)
{
    auto const treeIndex = static_cast<std::size_t>(blockIdx.x);
    if (treeIndex >= treeCount) { return; }
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity)) { return; }
    for (auto row = static_cast<std::size_t>(threadIdx.x); row < rowCount; row += blockDim.x) {
        float tape[NodeCapacity];
        float adjoint[NodeCapacity];
        float gradient[MaxParameters];
        auto const value = EvaluateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, coefficients, tape);
        residual[treeIndex * rowCount + row] = value - target[row];
        if (!isfinite(value) || !DifferentiateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, tape, adjoint, gradient)) {
            for (uint32_t parameter = 0; parameter < tree.CoefficientCount; ++parameter) {
                jacobian[static_cast<std::size_t>(tree.CoefficientOffset + parameter) * rowCount + row] = NAN;
            }
            continue;
        }
        for (uint32_t parameter = 0; parameter < tree.CoefficientCount; ++parameter) {
            jacobian[static_cast<std::size_t>(tree.CoefficientOffset + parameter) * rowCount + row] = gradient[parameter];
        }
    }
}

template<uint32_t MinNodes, uint32_t NodeCapacity>
__global__ void GaussianCostKernel(DeviceNode const* nodes, DeviceTreeRange const* trees, uint32_t const* indices,
                                   std::size_t treeCount, float const* columns, std::size_t rowCount,
                                   float const* target, float const* coefficients, float const* weights,
                                   bool hasWeights, float* costs, uint8_t* valid)
{
    auto const bucketIndex = static_cast<std::size_t>(blockIdx.x);
    if (bucketIndex >= treeCount) { return; }
    auto const treeIndex = static_cast<std::size_t>(indices[bucketIndex]);
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity)) { return; }
    __shared__ float partialCost[WaveSize];
    __shared__ uint8_t partialValid[WaveSize];
    auto cost = 0.0F;
    auto isValid = true;
    for (auto row = static_cast<std::size_t>(threadIdx.x); row < rowCount; row += blockDim.x) {
        float tape[NodeCapacity];
        auto const residual = EvaluateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, coefficients, tape) - target[row];
        auto const weight = hasWeights ? weights[row] : 1.0F;
        auto const term = 0.5F * weight * residual * residual;
        if (!isfinite(residual) || !isfinite(weight) || weight < 0.0F || !isfinite(term)) { isValid = false; }
        else { cost += term; }
    }
    partialCost[threadIdx.x] = cost;
    partialValid[threadIdx.x] = static_cast<uint8_t>(isValid);
    __syncthreads();
    if (threadIdx.x == 0) {
        auto total = 0.0F;
        auto allValid = true;
        for (uint32_t lane = 0; lane < WaveSize; ++lane) { total += partialCost[lane]; allValid = allValid && partialValid[lane] != 0; }
        costs[treeIndex] = total;
        valid[treeIndex] = static_cast<uint8_t>(allValid && isfinite(total));
    }
}

template<uint32_t MinNodes, uint32_t NodeCapacity>
__global__ void GaussianCostTiledKernel(DeviceNode const* nodes, DeviceTreeRange const* trees, uint32_t const* indices,
                                        std::size_t treeCount, float const* columns, std::size_t rowCount,
                                        float const* target, float const* coefficients, float const* weights,
                                        bool hasWeights, float* partialCosts, uint8_t* partialValid)
{
    auto const bucketIndex = static_cast<std::size_t>(blockIdx.x);
    auto const tile = static_cast<std::size_t>(blockIdx.y);
    if (bucketIndex >= treeCount || tile >= 2) { return; }
    auto const treeIndex = static_cast<std::size_t>(indices[bucketIndex]);
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity)) { return; }
    auto const tileSize = (rowCount + 1) / 2;
    auto const first = tile * tileSize;
    auto const last = min(rowCount, first + tileSize);
    __shared__ float laneCosts[WaveSize];
    __shared__ uint8_t laneValid[WaveSize];
    auto cost = 0.0F;
    auto isValid = true;
    for (auto row = first + static_cast<std::size_t>(threadIdx.x); row < last; row += blockDim.x) {
        float tape[NodeCapacity];
        auto const residual = EvaluateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, coefficients, tape) - target[row];
        auto const weight = hasWeights ? weights[row] : 1.0F;
        auto const term = 0.5F * weight * residual * residual;
        if (!isfinite(residual) || !isfinite(weight) || weight < 0.0F || !isfinite(term)) { isValid = false; }
        else { cost += term; }
    }
    laneCosts[threadIdx.x] = cost;
    laneValid[threadIdx.x] = static_cast<uint8_t>(isValid);
    __syncthreads();
    if (threadIdx.x == 0) {
        auto total = 0.0F;
        auto allValid = true;
        for (uint32_t lane = 0; lane < WaveSize; ++lane) { total += laneCosts[lane]; allValid = allValid && laneValid[lane] != 0; }
        partialCosts[treeIndex * 2 + tile] = total;
        partialValid[treeIndex * 2 + tile] = static_cast<uint8_t>(allValid && isfinite(total));
    }
}

__global__ void ReduceGaussianCostTilesKernel(uint32_t const* indices, std::size_t treeCount,
                                               float const* partialCosts, uint8_t const* partialValid,
                                               float* costs, uint8_t* valid)
{
    auto const bucketIndex = static_cast<std::size_t>(blockIdx.x);
    if (bucketIndex >= treeCount || threadIdx.x != 0) { return; }
    auto const treeIndex = static_cast<std::size_t>(indices[bucketIndex]);
    auto const cost = partialCosts[treeIndex * 2] + partialCosts[treeIndex * 2 + 1];
    costs[treeIndex] = cost;
    valid[treeIndex] = static_cast<uint8_t>(partialValid[treeIndex * 2] != 0 && partialValid[treeIndex * 2 + 1] != 0 && isfinite(cost));
}


template<uint32_t MinNodes, uint32_t NodeCapacity>
__global__ void InitialCostKernel(DeviceTreeRange const* trees, uint32_t const* indices, std::size_t treeCount,
                                  std::size_t rowCount, float const* residual, float const* weights,
                                  bool hasWeights, float* initialCosts, float* costs, float* damping,
                                  uint8_t* state, uint32_t* iterations, uint32_t* accepted)
{
    auto const bucketIndex = static_cast<std::size_t>(blockIdx.x);
    if (bucketIndex >= treeCount) { return; }
    auto const treeIndex = static_cast<std::size_t>(indices[bucketIndex]);
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity)) { return; }

    __shared__ float partialCost[WaveSize];
    __shared__ uint8_t partialValid[WaveSize];
    auto cost = 0.0F;
    auto valid = true;
    for (auto row = static_cast<std::size_t>(threadIdx.x); row < rowCount; row += blockDim.x) {
        auto const residualValue = residual[treeIndex * rowCount + row];
        auto const weight = hasWeights ? weights[row] : 1.0F;
        auto const term = 0.5F * weight * residualValue * residualValue;
        if (!isfinite(residualValue) || !isfinite(weight) || weight < 0.0F || !isfinite(term)) {
            valid = false;
        } else {
            cost += term;
        }
    }
    partialCost[threadIdx.x] = cost;
    partialValid[threadIdx.x] = static_cast<uint8_t>(valid);
    __syncthreads();
    if (threadIdx.x == 0) {
        auto total = 0.0F;
        auto allValid = true;
        for (uint32_t lane = 0; lane < WaveSize; ++lane) {
            total += partialCost[lane];
            allValid = allValid && partialValid[lane] != 0;
        }
        allValid = allValid && isfinite(total);
        initialCosts[treeIndex] = total;
        costs[treeIndex] = total;
        damping[treeIndex] = 1e-3F;
        state[treeIndex] = static_cast<uint8_t>(allValid);
        iterations[treeIndex] = 0;
        accepted[treeIndex] = 0;
    }
}

template<uint32_t MinNodes, uint32_t NodeCapacity>
__global__ void TrialCostKernel(DeviceTreeRange const* trees, uint32_t const* indices, std::size_t treeCount,
                                std::size_t rowCount, float const* residual, float const* weights,
                                bool hasWeights, float* trialCosts, uint8_t* trialValid)
{
    auto const bucketIndex = static_cast<std::size_t>(blockIdx.x);
    if (bucketIndex >= treeCount) { return; }
    auto const treeIndex = static_cast<std::size_t>(indices[bucketIndex]);
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity)) { return; }

    __shared__ float partialCost[WaveSize];
    __shared__ uint8_t partialValid[WaveSize];
    auto cost = 0.0F;
    auto valid = true;
    for (auto row = static_cast<std::size_t>(threadIdx.x); row < rowCount; row += blockDim.x) {
        auto const residualValue = residual[treeIndex * rowCount + row];
        auto const weight = hasWeights ? weights[row] : 1.0F;
        auto const term = 0.5F * weight * residualValue * residualValue;
        if (!isfinite(residualValue) || !isfinite(weight) || weight < 0.0F || !isfinite(term)) {
            valid = false;
        } else {
            cost += term;
        }
    }
    partialCost[threadIdx.x] = cost;
    partialValid[threadIdx.x] = static_cast<uint8_t>(valid);
    __syncthreads();
    if (threadIdx.x == 0) {
        auto total = 0.0F;
        auto allValid = true;
        for (uint32_t lane = 0; lane < WaveSize; ++lane) {
            total += partialCost[lane];
            allValid = allValid && partialValid[lane] != 0;
        }
        trialCosts[treeIndex] = total;
        trialValid[treeIndex] = static_cast<uint8_t>(allValid && isfinite(total));
    }
}

// Evaluates, reverse-differentiates, and reduces one tree per block. The
// Jacobian oracle retains its materialized path; LM never writes its rows.
template<uint32_t MinNodes, uint32_t NodeCapacity, uint32_t LowerExclusive, uint32_t ParameterCapacity, bool CollectCost>
__global__ void FusedNormalEquationKernel(DeviceNode const* nodes, DeviceTreeRange const* trees,
                                          uint32_t const* indices, std::size_t treeCount, float const* columns, std::size_t rowCount,
                                          float const* target, float const* coefficients, float const* weights,
                                          bool hasWeights, float* normal, float* gradient, float* costs, uint8_t* state)
{
    auto const bucketIndex = static_cast<std::size_t>(blockIdx.x);
    if (bucketIndex >= treeCount) { return; }
    auto const treeIndex = static_cast<std::size_t>(indices[bucketIndex]);
    auto const tree = trees[treeIndex];
    if (!IsNodeBucket(tree, MinNodes, NodeCapacity) || !IsCoefficientBucket(tree, LowerExclusive, ParameterCapacity)) { return; }

    constexpr auto MatrixElements = ParameterCapacity * (ParameterCapacity + 1U) / 2U;
    constexpr auto Elements = MatrixElements + ParameterCapacity + (CollectCost ? 1U : 0U);
    __shared__ float partial[WaveSize * Elements];
    __shared__ uint8_t partialValid[WaveSize];

    float localNormal[MatrixElements]{};
    float localGradient[ParameterCapacity]{};
    auto localCost = 0.0F;
    auto valid = state[treeIndex] != 0;
    auto const count = tree.CoefficientCount;
    for (auto row = static_cast<std::size_t>(threadIdx.x); row < rowCount; row += blockDim.x) {
        float tape[NodeCapacity];
        float adjoint[NodeCapacity];
        float rowGradient[MaxParameters];
        auto const prediction = EvaluateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, coefficients, tape);
        auto const residual = prediction - target[row];
        auto const weight = hasWeights ? weights[row] : 1.0F;
        auto const differentiated = isfinite(prediction)
            && DifferentiateOne<NodeCapacity>(nodes, tree, columns, rowCount, row, tape, adjoint, rowGradient);
        if (!differentiated || !isfinite(residual) || !isfinite(weight) || weight < 0.0F) {
            valid = false;
            continue;
        }
        if constexpr (CollectCost) {
            auto const term = 0.5F * weight * residual * residual;
            if (!isfinite(term)) {
                valid = false;
                continue;
            }
            localCost += term;
        }
        for (uint32_t i = 0; i < count; ++i) {
            auto const ji = rowGradient[i];
            if (!isfinite(ji)) { valid = false; continue; }
            localGradient[i] += weight * ji * residual;
            for (uint32_t j = 0; j <= i; ++j) {
                auto const jj = rowGradient[j];
                if (!isfinite(jj)) { valid = false; continue; }
                localNormal[i * (i + 1U) / 2U + j] += weight * ji * jj;
            }
        }
    }

    auto const laneOffset = threadIdx.x * Elements;
    for (uint32_t i = 0; i < MatrixElements; ++i) { partial[laneOffset + i] = localNormal[i]; }
    for (uint32_t i = 0; i < ParameterCapacity; ++i) { partial[laneOffset + MatrixElements + i] = localGradient[i]; }
    if constexpr (CollectCost) { partial[laneOffset + MatrixElements + ParameterCapacity] = localCost; }
    partialValid[threadIdx.x] = static_cast<uint8_t>(valid);
    __syncthreads();
    if (threadIdx.x == 0) {
        auto allValid = true;
        if constexpr (CollectCost) {
            auto cost = 0.0F;
            for (uint32_t lane = 0; lane < WaveSize; ++lane) {
                allValid = allValid && partialValid[lane] != 0;
                cost += partial[lane * Elements + MatrixElements + ParameterCapacity];
            }
            costs[treeIndex] = cost;
            allValid = allValid && isfinite(cost);
        } else {
            for (uint32_t lane = 0; lane < WaveSize; ++lane) { allValid = allValid && partialValid[lane] != 0; }
        }
        for (uint32_t i = 0; i < count; ++i) {
            auto gradientValue = 0.0F;
            for (uint32_t lane = 0; lane < WaveSize; ++lane) {
                gradientValue += partial[lane * Elements + MatrixElements + i];
            }
            gradient[treeIndex * MaxParameters + i] = gradientValue;
            allValid = allValid && isfinite(gradientValue);
            for (uint32_t j = 0; j <= i; ++j) {
                auto normalValue = 0.0F;
                for (uint32_t lane = 0; lane < WaveSize; ++lane) {
                    normalValue += partial[lane * Elements + i * (i + 1U) / 2U + j];
                }
                normal[treeIndex * MaxParameters * MaxParameters + i * MaxParameters + j] = normalValue;
                normal[treeIndex * MaxParameters * MaxParameters + j * MaxParameters + i] = normalValue;
                allValid = allValid && isfinite(normalValue);
            }
        }
        state[treeIndex] = static_cast<uint8_t>(allValid);
    }
}

__global__ void SolveKernel(DeviceTreeRange const* trees, std::size_t treeCount, float* normal,
                            float const* gradient, float const* damping, float* trial, uint8_t* state)
{
    auto const treeIndex = static_cast<std::size_t>(blockIdx.x);
    if (treeIndex >= treeCount || threadIdx.x != 0 || state[treeIndex] == 0) { return; }
    auto const tree = trees[treeIndex];
    auto const count = tree.CoefficientCount;
    auto* matrix = normal + treeIndex * MaxParameters * MaxParameters;
    auto const rhs = trial + tree.CoefficientOffset;
    for (uint32_t i = 0; i < count; ++i) {
        matrix[i * MaxParameters + i] += damping[treeIndex];
        rhs[i] = -gradient[treeIndex * MaxParameters + i];
    }
    for (uint32_t pivot = 0; pivot < count; ++pivot) {
        auto best = pivot;
        for (uint32_t row = pivot + 1; row < count; ++row) {
            if (fabsf(matrix[row * MaxParameters + pivot]) > fabsf(matrix[best * MaxParameters + pivot])) { best = row; }
        }
        auto const pivotValue = matrix[best * MaxParameters + pivot];
        if (!isfinite(pivotValue) || fabsf(pivotValue) < 1e-10F) { state[treeIndex] = 0; return; }
        if (best != pivot) {
            for (uint32_t column = pivot; column < count; ++column) {
                auto const temporary = matrix[pivot * MaxParameters + column];
                matrix[pivot * MaxParameters + column] = matrix[best * MaxParameters + column];
                matrix[best * MaxParameters + column] = temporary;
            }
            auto const temporary = rhs[pivot]; rhs[pivot] = rhs[best]; rhs[best] = temporary;
        }
        auto const diagonal = matrix[pivot * MaxParameters + pivot];
        for (uint32_t row = pivot + 1; row < count; ++row) {
            auto const factor = matrix[row * MaxParameters + pivot] / diagonal;
            for (uint32_t column = pivot; column < count; ++column) {
                matrix[row * MaxParameters + column] -= factor * matrix[pivot * MaxParameters + column];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }
    for (int row = static_cast<int>(count) - 1; row >= 0; --row) {
        auto value = rhs[row];
        for (uint32_t column = static_cast<uint32_t>(row) + 1; column < count; ++column) {
            value -= matrix[static_cast<uint32_t>(row) * MaxParameters + column] * rhs[column];
        }
        value /= matrix[static_cast<uint32_t>(row) * MaxParameters + static_cast<uint32_t>(row)];
        if (!isfinite(value)) { state[treeIndex] = 0; return; }
        rhs[row] = value;
    }
}

__global__ void AcceptKernel(DeviceTreeRange const* trees, std::size_t treeCount, float* coefficients,
                             float const* trial, float* costs, float const* trialCosts, float* damping,
                             uint8_t const* trialValid, uint8_t const* state, uint32_t* iterations,
                             uint32_t* accepted)
{
    auto const treeIndex = static_cast<std::size_t>(blockIdx.x);
    if (treeIndex >= treeCount || threadIdx.x != 0 || state[treeIndex] == 0) { return; }
    auto const tree = trees[treeIndex];
    auto const improved = trialValid[treeIndex] != 0 && isfinite(trialCosts[treeIndex]) && trialCosts[treeIndex] < costs[treeIndex];
    if (improved) {
        for (uint32_t parameter = 0; parameter < tree.CoefficientCount; ++parameter) {
            coefficients[tree.CoefficientOffset + parameter] += trial[tree.CoefficientOffset + parameter];
        }
        costs[treeIndex] = trialCosts[treeIndex];
        damping[treeIndex] = fmaxf(damping[treeIndex] * 0.3F, 1e-8F);
        ++accepted[treeIndex];
    } else {
        damping[treeIndex] = fminf(damping[treeIndex] * 10.0F, 1e8F);
    }
    ++iterations[treeIndex];
}

__global__ void FinalizeKernel(DeviceTreeRange const* trees, std::size_t treeCount, float const* initialCosts,
                               float const* costs, uint8_t const* state, float* finalCosts, uint8_t* status)
{
    auto const treeIndex = static_cast<std::size_t>(blockIdx.x);
    if (treeIndex >= treeCount || threadIdx.x != 0) { return; }
    auto const valid = state[treeIndex] != 0 && isfinite(costs[treeIndex]);
    finalCosts[treeIndex] = costs[treeIndex];
    status[treeIndex] = valid && costs[treeIndex] < initialCosts[treeIndex] ? 0 : (valid ? 1 : 2);
}

void LaunchEvaluateBuckets(Runtime* runtime, float const* coefficients, float* output, std::size_t rowCount)
{
    auto const block = dim3(WaveSize);
    auto const grid = dim3(static_cast<unsigned>(runtime->TreeCount),
                           static_cast<unsigned>((rowCount + WaveSize - 1) / WaveSize));
    hipLaunchKernelGGL((EvaluateKernel<1, 32>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, coefficients, output);
    hipLaunchKernelGGL((EvaluateKernel<33, 64>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, coefficients, output);
    hipLaunchKernelGGL((EvaluateKernel<65, 128>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, coefficients, output);
}

void LaunchJacobianBuckets(Runtime* runtime, float const* coefficients, float* jacobian, std::size_t rowCount)
{
    auto const grid = dim3(static_cast<unsigned>(runtime->TreeCount));
    auto const block = dim3(WaveSize);
    hipLaunchKernelGGL((JacRevKernel<1, 32>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, coefficients, jacobian);
    hipLaunchKernelGGL((JacRevKernel<33, 64>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, coefficients, jacobian);
    hipLaunchKernelGGL((JacRevKernel<65, 128>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, coefficients, jacobian);
}

void LaunchResidualBuckets(Runtime* runtime, float const* coefficients, float* residual, std::size_t rowCount)
{
#define OPERON_LAUNCH_RESIDUAL(BUCKET, MIN_NODES, NODE_CAPACITY) \
    if (runtime->BucketCounts[BUCKET] != 0) { hipLaunchKernelGGL((ResidualKernel<MIN_NODES, NODE_CAPACITY>), dim3(static_cast<unsigned>(runtime->BucketCounts[BUCKET])), dim3(WaveSize), 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->Buckets[BUCKET], runtime->BucketCounts[BUCKET], runtime->Columns, rowCount, runtime->Target, coefficients, residual); }
    OPERON_LAUNCH_RESIDUAL(0, 1, 32);
    OPERON_LAUNCH_RESIDUAL(1, 33, 64);
    OPERON_LAUNCH_RESIDUAL(2, 65, 128);
#undef OPERON_LAUNCH_RESIDUAL
}

void LaunchJacResidualBuckets(Runtime* runtime, std::size_t rowCount)
{
    auto const grid = dim3(static_cast<unsigned>(runtime->TreeCount));
    auto const block = dim3(WaveSize);
    hipLaunchKernelGGL((JacResidualKernel<1, 32>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, runtime->Target, runtime->Coefficients, runtime->Residual, runtime->Jacobian);
    hipLaunchKernelGGL((JacResidualKernel<33, 64>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, runtime->Target, runtime->Coefficients, runtime->Residual, runtime->Jacobian);
    hipLaunchKernelGGL((JacResidualKernel<65, 128>), grid, block, 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->TreeCount, runtime->Columns, rowCount, runtime->Target, runtime->Coefficients, runtime->Residual, runtime->Jacobian);
}

void LaunchInitialCostBuckets(Runtime* runtime, std::size_t rowCount, bool hasWeights)
{
#define OPERON_LAUNCH_INITIAL_COST(BUCKET, MIN_NODES, NODE_CAPACITY) \
    if (runtime->BucketCounts[BUCKET] != 0) { hipLaunchKernelGGL((InitialCostKernel<MIN_NODES, NODE_CAPACITY>), dim3(static_cast<unsigned>(runtime->BucketCounts[BUCKET])), dim3(WaveSize), 0, runtime->Stream, runtime->Trees, runtime->Buckets[BUCKET], runtime->BucketCounts[BUCKET], rowCount, runtime->Residual, runtime->Weights, hasWeights, runtime->InitialCosts, runtime->Costs, runtime->Damping, runtime->State, runtime->Iterations, runtime->Accepted); }
    OPERON_LAUNCH_INITIAL_COST(0, 1, 32);
    OPERON_LAUNCH_INITIAL_COST(1, 33, 64);
    OPERON_LAUNCH_INITIAL_COST(2, 65, 128);
#undef OPERON_LAUNCH_INITIAL_COST
}

void LaunchTrialCostBuckets(Runtime* runtime, std::size_t rowCount, bool hasWeights)
{
#define OPERON_LAUNCH_TRIAL_COST(BUCKET, MIN_NODES, NODE_CAPACITY) \
    if (runtime->BucketCounts[BUCKET] != 0) { hipLaunchKernelGGL((TrialCostKernel<MIN_NODES, NODE_CAPACITY>), dim3(static_cast<unsigned>(runtime->BucketCounts[BUCKET])), dim3(WaveSize), 0, runtime->Stream, runtime->Trees, runtime->Buckets[BUCKET], runtime->BucketCounts[BUCKET], rowCount, runtime->TrialResidual, runtime->Weights, hasWeights, runtime->TrialCosts, runtime->TrialValid); }
    OPERON_LAUNCH_TRIAL_COST(0, 1, 32);
    OPERON_LAUNCH_TRIAL_COST(1, 33, 64);
    OPERON_LAUNCH_TRIAL_COST(2, 65, 128);
#undef OPERON_LAUNCH_TRIAL_COST
}

template<bool CollectCost>
void LaunchFusedNormalBuckets(Runtime* runtime, std::size_t rowCount, bool hasWeights, float* costs)
{
#define OPERON_LAUNCH_FUSED_NORMAL(BUCKET, MIN_NODES, NODE_CAPACITY) \
    if (runtime->BucketCounts[BUCKET] != 0) { \
        auto const grid = dim3(static_cast<unsigned>(runtime->BucketCounts[BUCKET])); \
        hipLaunchKernelGGL((FusedNormalEquationKernel<MIN_NODES, NODE_CAPACITY, 0, MaxParameters, CollectCost>), grid, dim3(WaveSize), 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->Buckets[BUCKET], runtime->BucketCounts[BUCKET], runtime->Columns, rowCount, runtime->Target, runtime->Coefficients, runtime->Weights, hasWeights, runtime->Normal, runtime->Gradient, costs, runtime->State); \
    }
    OPERON_LAUNCH_FUSED_NORMAL(0, 1, 32);
    OPERON_LAUNCH_FUSED_NORMAL(1, 33, 64);
    OPERON_LAUNCH_FUSED_NORMAL(2, 65, 128);
#undef OPERON_LAUNCH_FUSED_NORMAL
}



void CheckLaunches()
{
    Check(hipGetLastError());
}
} // namespace

void JacRev(Runtime* runtime, float const* columns, std::size_t variableCount, std::size_t rowCount,
            float const* coefficients, std::size_t coefficientCount, float* jacobian)
{
    Reserve(runtime->Columns, runtime->ColumnsCapacity, variableCount * rowCount);
    Reserve(runtime->Coefficients, runtime->CoefficientsCapacity, coefficientCount);
    Reserve(runtime->Jacobian, runtime->JacobianCapacity, coefficientCount * rowCount);
    Check(hipMemcpyAsync(runtime->Columns, columns, variableCount * rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipMemcpyAsync(runtime->Coefficients, coefficients, coefficientCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipMemsetAsync(runtime->Jacobian, 0, coefficientCount * rowCount * sizeof(float), runtime->Stream));
    LaunchJacobianBuckets(runtime, runtime->Coefficients, runtime->Jacobian, rowCount);
    CheckLaunches();
    Check(hipMemcpyAsync(jacobian, runtime->Jacobian, coefficientCount * rowCount * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipStreamSynchronize(runtime->Stream));
}

void NormalEquations(Runtime* runtime, float const* columns, std::size_t variableCount, std::size_t rowCount,
                     float const* target, float const* weights, bool hasWeights,
                     float const* coefficients, std::size_t coefficientCount,
                     float* normal, float* gradient, float* costs, uint8_t* valid)
{
    if (runtime->TreeCount == 0 || rowCount == 0) {
        throw std::invalid_argument("HIP normal equations require trees and rows");
    }
    auto const treeCount = runtime->TreeCount;
    Reserve(runtime->Columns, runtime->ColumnsCapacity, variableCount * rowCount);
    Reserve(runtime->Target, runtime->TargetCapacity, rowCount);
    if (hasWeights) { Reserve(runtime->Weights, runtime->WeightsCapacity, rowCount); }
    Reserve(runtime->Coefficients, runtime->CoefficientsCapacity, coefficientCount);
    Reserve(runtime->Normal, runtime->NormalCapacity, treeCount * MaxParameters * MaxParameters);
    Reserve(runtime->Gradient, runtime->GradientCapacity, treeCount * MaxParameters);

    Check(hipMemcpyAsync(runtime->Columns, columns, variableCount * rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipMemcpyAsync(runtime->Target, target, rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    if (hasWeights) { Check(hipMemcpyAsync(runtime->Weights, weights, rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream)); }
    Check(hipMemcpyAsync(runtime->Coefficients, coefficients, coefficientCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipMemsetAsync(runtime->State, 1, treeCount * sizeof(uint8_t), runtime->Stream));
    LaunchFusedNormalBuckets<true>(runtime, rowCount, hasWeights, runtime->TrialCosts);
    CheckLaunches();
    Check(hipMemcpyAsync(normal, runtime->Normal, treeCount * MaxParameters * MaxParameters * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(gradient, runtime->Gradient, treeCount * MaxParameters * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(costs, runtime->TrialCosts, treeCount * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(valid, runtime->State, treeCount * sizeof(uint8_t), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipStreamSynchronize(runtime->Stream));
}

auto Create() -> Runtime*
{
    auto* runtime = new Runtime{};
    try { Check(hipStreamCreateWithFlags(&runtime->Stream, hipStreamNonBlocking)); }
    catch (...) { delete runtime; throw; }
    return runtime;
}
void Destroy(Runtime* runtime) noexcept
{
    if (runtime == nullptr) { return; }
#define OPERON_HIP_FREE(member) if (runtime->member != nullptr) { (void)hipFree(runtime->member); }
    OPERON_HIP_FREE(Nodes)
    OPERON_HIP_FREE(Trees)
    OPERON_HIP_FREE(Columns)
    OPERON_HIP_FREE(Coefficients)
    OPERON_HIP_FREE(Output)
    OPERON_HIP_FREE(Jacobian)
    OPERON_HIP_FREE(Target)
    OPERON_HIP_FREE(Weights)
    OPERON_HIP_FREE(Residual)
    OPERON_HIP_FREE(TrialResidual)
    OPERON_HIP_FREE(Normal)
    OPERON_HIP_FREE(Gradient)
    OPERON_HIP_FREE(Trial)
    OPERON_HIP_FREE(InitialCosts)
    OPERON_HIP_FREE(Costs)
    OPERON_HIP_FREE(TrialCosts)
    OPERON_HIP_FREE(Damping)
    OPERON_HIP_FREE(State)
    OPERON_HIP_FREE(TrialValid)
    OPERON_HIP_FREE(Status)
    OPERON_HIP_FREE(Iterations)
    OPERON_HIP_FREE(Accepted)
    OPERON_HIP_FREE(CostPartials)
    OPERON_HIP_FREE(CostPartialValid)
    for (auto* bucket : runtime->Buckets) { if (bucket != nullptr) { (void)hipFree(bucket); } }
#undef OPERON_HIP_FREE
    if (runtime->Stream != nullptr) { (void)hipStreamDestroy(runtime->Stream); }
    delete runtime;
}

void Upload(Runtime* runtime, DeviceNode const* nodes, std::size_t nodeCount,
            DeviceTreeRange const* trees, std::size_t treeCount)
{
    for (auto const& tree : std::span{trees, treeCount}) {
        if (tree.NodeCount == 0 || tree.NodeCount > MaxNodes) {
            throw std::invalid_argument("HIP P2 supports 1..128 nodes per tree");
        }
    }
    Reserve(runtime->Nodes, runtime->NodesCapacity, nodeCount);
    Reserve(runtime->Trees, runtime->TreesCapacity, treeCount);
    runtime->TreeCount = treeCount;
    runtime->CoefficientCount = 0;
    for (auto const& tree : std::span{trees, treeCount}) {
        runtime->CoefficientCount = std::max(runtime->CoefficientCount, static_cast<std::size_t>(tree.CoefficientOffset + tree.CoefficientCount));
    }
    Check(hipMemcpyAsync(runtime->Nodes, nodes, nodeCount * sizeof(DeviceNode), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipMemcpyAsync(runtime->Trees, trees, treeCount * sizeof(DeviceTreeRange), hipMemcpyHostToDevice, runtime->Stream));
    std::array<std::vector<uint32_t>, 3> buckets;
    for (uint32_t index = 0; index < treeCount; ++index) {
        auto const count = trees[index].NodeCount;
        buckets[count <= 32U ? 0U : (count <= 64U ? 1U : 2U)].push_back(index);
    }
    for (std::size_t bucket = 0; bucket < buckets.size(); ++bucket) {
        runtime->BucketCounts[bucket] = buckets[bucket].size();
        Reserve(runtime->Buckets[bucket], runtime->BucketCapacities[bucket], buckets[bucket].size());
        if (!buckets[bucket].empty()) {
            Check(hipMemcpyAsync(runtime->Buckets[bucket], buckets[bucket].data(), buckets[bucket].size() * sizeof(uint32_t), hipMemcpyHostToDevice, runtime->Stream));
        }
    }
    Check(hipStreamSynchronize(runtime->Stream));
}

void UpdateCoefficients(Runtime* runtime, float const* coefficients, std::size_t coefficientCount)
{
    if (coefficientCount != runtime->CoefficientCount) {
        throw std::invalid_argument("HIP coefficient update must match the uploaded population");
    }
    Reserve(runtime->Coefficients, runtime->CoefficientsCapacity, coefficientCount);
    Check(hipMemcpyAsync(runtime->Coefficients, coefficients, coefficientCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipStreamSynchronize(runtime->Stream));
}

void OptimizeGaussian(Runtime* runtime, float const* columns, std::size_t variableCount, std::size_t rowCount,
                      float const* target, float const* weights, bool hasWeights,
                      float const* coefficients, std::size_t coefficientCount, uint32_t maxIterations,
                      float* finalCoefficients, float* initialCosts, float* finalCosts,
                      uint8_t* status, uint32_t* iterations, uint32_t* acceptedSteps)
{
    if (runtime->TreeCount == 0 || rowCount == 0 || maxIterations == 0) {
        throw std::invalid_argument("HIP optimizer requires trees, rows, and iterations");
    }
    auto const treeCount = runtime->TreeCount;
    Reserve(runtime->Columns, runtime->ColumnsCapacity, variableCount * rowCount);
    Reserve(runtime->Target, runtime->TargetCapacity, rowCount);
    if (hasWeights) { Reserve(runtime->Weights, runtime->WeightsCapacity, rowCount); }
    Reserve(runtime->Coefficients, runtime->CoefficientsCapacity, coefficientCount);
    Reserve(runtime->Residual, runtime->ResidualCapacity, treeCount * rowCount);

    Reserve(runtime->TrialResidual, runtime->TrialResidualCapacity, treeCount * rowCount);
    Reserve(runtime->Normal, runtime->NormalCapacity, treeCount * MaxParameters * MaxParameters);
    Reserve(runtime->Gradient, runtime->GradientCapacity, treeCount * MaxParameters);
    Reserve(runtime->Trial, runtime->TrialCapacity, coefficientCount);
    Reserve(runtime->InitialCosts, runtime->InitialCostsCapacity, treeCount);
    Reserve(runtime->Costs, runtime->CostsCapacity, treeCount);
    Reserve(runtime->TrialCosts, runtime->TrialCostsCapacity, treeCount);
    Reserve(runtime->Damping, runtime->DampingCapacity, treeCount);
    Reserve(runtime->State, runtime->StateCapacity, treeCount);
    Reserve(runtime->TrialValid, runtime->TrialValidCapacity, treeCount);
    Reserve(runtime->Status, runtime->StatusCapacity, treeCount);
    Reserve(runtime->Iterations, runtime->IterationsCapacity, treeCount);
    Reserve(runtime->Accepted, runtime->AcceptedCapacity, treeCount);

    Check(hipMemcpyAsync(runtime->Columns, columns, variableCount * rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipMemcpyAsync(runtime->Target, target, rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    if (hasWeights) { Check(hipMemcpyAsync(runtime->Weights, weights, rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream)); }
    Check(hipMemcpyAsync(runtime->Coefficients, coefficients, coefficientCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));

    LaunchResidualBuckets(runtime, runtime->Coefficients, runtime->Residual, rowCount);
    LaunchInitialCostBuckets(runtime, rowCount, hasWeights);
    CheckLaunches();

    auto const grid = dim3(static_cast<unsigned>(treeCount));
    for (uint32_t iteration = 0; iteration < maxIterations; ++iteration) {
        LaunchFusedNormalBuckets<false>(runtime, rowCount, hasWeights, nullptr);
        hipLaunchKernelGGL(SolveKernel, grid, dim3(1), 0, runtime->Stream,
                           runtime->Trees, treeCount, runtime->Normal, runtime->Gradient,
                           runtime->Damping, runtime->Trial, runtime->State);
        LaunchResidualBuckets(runtime, runtime->Trial, runtime->TrialResidual, rowCount);
        LaunchTrialCostBuckets(runtime, rowCount, hasWeights);
        hipLaunchKernelGGL(AcceptKernel, grid, dim3(1), 0, runtime->Stream,
                           runtime->Trees, treeCount, runtime->Coefficients, runtime->Trial,
                           runtime->Costs, runtime->TrialCosts, runtime->Damping, runtime->TrialValid,
                           runtime->State, runtime->Iterations, runtime->Accepted);
        CheckLaunches();
    }
    hipLaunchKernelGGL(FinalizeKernel, grid, dim3(1), 0, runtime->Stream,
                       runtime->Trees, treeCount, runtime->InitialCosts, runtime->Costs,
                       runtime->State, runtime->TrialCosts, runtime->Status);
    CheckLaunches();

    Check(hipMemcpyAsync(finalCoefficients, runtime->Coefficients, coefficientCount * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(initialCosts, runtime->InitialCosts, treeCount * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(finalCosts, runtime->TrialCosts, treeCount * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(status, runtime->Status, treeCount * sizeof(uint8_t), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(iterations, runtime->Iterations, treeCount * sizeof(uint32_t), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(acceptedSteps, runtime->Accepted, treeCount * sizeof(uint32_t), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipStreamSynchronize(runtime->Stream));
}

void GaussianCosts(Runtime* runtime, float const* columns, std::size_t variableCount, std::size_t rowCount,
                   float const* target, float const* weights, bool hasWeights, float* costs, uint8_t* valid)
{
    if (runtime->TreeCount == 0 || rowCount == 0) { throw std::invalid_argument("HIP Gaussian costs require trees and rows"); }
    auto const treeCount = runtime->TreeCount;
    Reserve(runtime->Columns, runtime->ColumnsCapacity, variableCount * rowCount);
    Reserve(runtime->Target, runtime->TargetCapacity, rowCount);
    if (hasWeights) { Reserve(runtime->Weights, runtime->WeightsCapacity, rowCount); }
    Reserve(runtime->Costs, runtime->CostsCapacity, treeCount);
    Reserve(runtime->State, runtime->StateCapacity, treeCount);
    Check(hipMemcpyAsync(runtime->Columns, columns, variableCount * rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipMemcpyAsync(runtime->Target, target, rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    if (hasWeights) { Check(hipMemcpyAsync(runtime->Weights, weights, rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream)); }
    if (rowCount < 4096) {
#define OPERON_LAUNCH_GAUSSIAN_COST(BUCKET, MIN_NODES, NODE_CAPACITY) \
        if (runtime->BucketCounts[BUCKET] != 0) { hipLaunchKernelGGL((GaussianCostKernel<MIN_NODES, NODE_CAPACITY>), dim3(static_cast<unsigned>(runtime->BucketCounts[BUCKET])), dim3(WaveSize), 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->Buckets[BUCKET], runtime->BucketCounts[BUCKET], runtime->Columns, rowCount, runtime->Target, runtime->Coefficients, runtime->Weights, hasWeights, runtime->Costs, runtime->State); }
        OPERON_LAUNCH_GAUSSIAN_COST(0, 1, 32);
        OPERON_LAUNCH_GAUSSIAN_COST(1, 33, 64);
        OPERON_LAUNCH_GAUSSIAN_COST(2, 65, 128);
#undef OPERON_LAUNCH_GAUSSIAN_COST
    } else {
        Reserve(runtime->CostPartials, runtime->CostPartialsCapacity, treeCount * 2);
        Reserve(runtime->CostPartialValid, runtime->CostPartialValidCapacity, treeCount * 2);
#define OPERON_LAUNCH_TILED_COST(BUCKET, MIN_NODES, NODE_CAPACITY) \
        if (runtime->BucketCounts[BUCKET] != 0) { \
            auto const grid = dim3(static_cast<unsigned>(runtime->BucketCounts[BUCKET]), 2); \
            hipLaunchKernelGGL((GaussianCostTiledKernel<MIN_NODES, NODE_CAPACITY>), grid, dim3(WaveSize), 0, runtime->Stream, runtime->Nodes, runtime->Trees, runtime->Buckets[BUCKET], runtime->BucketCounts[BUCKET], runtime->Columns, rowCount, runtime->Target, runtime->Coefficients, runtime->Weights, hasWeights, runtime->CostPartials, runtime->CostPartialValid); \
            hipLaunchKernelGGL(ReduceGaussianCostTilesKernel, dim3(static_cast<unsigned>(runtime->BucketCounts[BUCKET])), dim3(1), 0, runtime->Stream, runtime->Buckets[BUCKET], runtime->BucketCounts[BUCKET], runtime->CostPartials, runtime->CostPartialValid, runtime->Costs, runtime->State); \
        }
        OPERON_LAUNCH_TILED_COST(0, 1, 32);
        OPERON_LAUNCH_TILED_COST(1, 33, 64);
        OPERON_LAUNCH_TILED_COST(2, 65, 128);
#undef OPERON_LAUNCH_TILED_COST
    }
    CheckLaunches();
    Check(hipMemcpyAsync(costs, runtime->Costs, treeCount * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipMemcpyAsync(valid, runtime->State, treeCount * sizeof(uint8_t), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipStreamSynchronize(runtime->Stream));
}
void EvaluateResident(Runtime* runtime, float const* columns, std::size_t variableCount, std::size_t rowCount, float* output)
{
    Reserve(runtime->Columns, runtime->ColumnsCapacity, variableCount * rowCount);
    Reserve(runtime->Output, runtime->OutputCapacity, runtime->TreeCount * rowCount);
    Check(hipMemcpyAsync(runtime->Columns, columns, variableCount * rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    LaunchEvaluateBuckets(runtime, runtime->Coefficients, runtime->Output, rowCount);
    CheckLaunches();
    Check(hipMemcpyAsync(output, runtime->Output, runtime->TreeCount * rowCount * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipStreamSynchronize(runtime->Stream));
}

void Evaluate(Runtime* runtime, float const* columns, std::size_t variableCount, std::size_t rowCount,
              float const* coefficients, std::size_t coefficientCount, float* output)
{
    Reserve(runtime->Columns, runtime->ColumnsCapacity, variableCount * rowCount);
    Reserve(runtime->Coefficients, runtime->CoefficientsCapacity, coefficientCount);
    Reserve(runtime->Output, runtime->OutputCapacity, runtime->TreeCount * rowCount);
    Check(hipMemcpyAsync(runtime->Columns, columns, variableCount * rowCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    Check(hipMemcpyAsync(runtime->Coefficients, coefficients, coefficientCount * sizeof(float), hipMemcpyHostToDevice, runtime->Stream));
    LaunchEvaluateBuckets(runtime, runtime->Coefficients, runtime->Output, rowCount);
    CheckLaunches();
    Check(hipMemcpyAsync(output, runtime->Output, runtime->TreeCount * rowCount * sizeof(float), hipMemcpyDeviceToHost, runtime->Stream));
    Check(hipStreamSynchronize(runtime->Stream));
}

} // namespace Operon::PopulationOptimization::Hip::detail
