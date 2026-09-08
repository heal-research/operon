// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <algorithm>
#include <array>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>


#include "operon/core/node.hpp"
#include "operon/optimizer/detail/hip_kernel.hpp"
#include "operon/optimizer/detail/sycl_kernel.hpp"
#include <sycl/sycl.hpp>
#include "operon/optimizer/sycl_context.hpp"
namespace Operon::PopulationOptimization::Sycl {
namespace {
using HostDeviceNode = Hip::detail::DeviceNode;
using HostDeviceTreeRange = Hip::detail::DeviceTreeRange;
using Hip::detail::Opcode;
constexpr auto MaxNodes = 128U;
constexpr auto MaxParameters = 16U;
constexpr auto NaN = std::numeric_limits<float>::quiet_NaN();

[[nodiscard]] auto LowerNode(EncodedNode const& node, std::span<Operon::Hash const> variables) -> HostDeviceNode
{
    auto opcode = Opcode::Constant;
    uint32_t operand{};
    if (node.Type == NodeType::Variable) {
        auto const it = std::ranges::find(variables, node.HashValue);
        if (it == variables.end()) { throw std::invalid_argument("SYCL population contains an unbound variable"); }
        opcode = Opcode::Variable;
        operand = static_cast<uint32_t>(std::distance(variables.begin(), it));
    } else if (node.Type == NodeType::Ref) {
        opcode = Opcode::Ref;
        operand = node.RefTo;
    } else if (node.Type == NodeType::Function) {
        switch (static_cast<BuiltinOp>(node.HashValue)) {
        case BuiltinOp::Add: opcode = Opcode::Add; break;
        case BuiltinOp::Sub: opcode = Opcode::Sub; break;
        case BuiltinOp::Mul: opcode = Opcode::Mul; break;
        case BuiltinOp::Div: opcode = Opcode::Div; break;
        case BuiltinOp::Square: opcode = Opcode::Square; break;
        case BuiltinOp::Exp: opcode = Opcode::Exp; break;
        case BuiltinOp::Log: opcode = Opcode::Log; break;
        case BuiltinOp::Sin: opcode = Opcode::Sin; break;
        case BuiltinOp::Cos: opcode = Opcode::Cos; break;
        default: throw std::invalid_argument("SYCL population contains an unsupported primitive");
        }
    } else if (node.Type != NodeType::Constant) {
        throw std::invalid_argument("SYCL population contains an unsupported node type");
    }
    return {.Value = node.Value, .Operand = operand, .Coefficient = 0, .Arity = node.Arity,
            .Length = node.Length, .Op = opcode, .Optimize = static_cast<uint8_t>(node.Optimize)};
}

[[nodiscard]] auto TrustedCost(std::span<HostDeviceNode const> nodes, HostDeviceTreeRange tree,
                               std::span<Operon::Scalar const> columns, std::size_t variables,
                               std::size_t rows, std::span<Operon::Scalar const> target,
                               std::span<Operon::Scalar const> weights,
                               std::span<Operon::Scalar const> coefficients) -> double
{
    if (tree.NodeCount == 0 || tree.NodeCount > MaxNodes || tree.CoefficientCount > MaxParameters) {
        return std::numeric_limits<double>::quiet_NaN();
    }
    std::array<double, MaxNodes> tape{};
    auto cost = 0.0;
    for (std::size_t row = 0; row < rows; ++row) {
        for (uint32_t local = 0; local < tree.NodeCount; ++local) {
            auto const& node = nodes[tree.NodeOffset + local];
            auto value = std::numeric_limits<double>::quiet_NaN();
            if (node.Op == Opcode::Constant) {
                value = node.Optimize ? coefficients[tree.CoefficientOffset + node.Coefficient] : node.Value;
            } else if (node.Op == Opcode::Variable) {
                value = columns[static_cast<std::size_t>(node.Operand) * rows + row]
                    * (node.Optimize ? coefficients[tree.CoefficientOffset + node.Coefficient] : node.Value);
            } else if (node.Op == Opcode::Ref) {
                if (node.Operand >= local) { return std::numeric_limits<double>::quiet_NaN(); }
                value = tape[node.Operand];
            } else if (node.Op == Opcode::Square || node.Op == Opcode::Exp || node.Op == Opcode::Log || node.Op == Opcode::Sin || node.Op == Opcode::Cos) {
                if (node.Arity != 1 || local == 0) { return std::numeric_limits<double>::quiet_NaN(); }
                auto const x = tape[local - 1];
                if (node.Op == Opcode::Square) { value = x * x; }
                else if (node.Op == Opcode::Exp) { value = std::exp(x); }
                else if (node.Op == Opcode::Log) { value = std::log(x); }
                else if (node.Op == Opcode::Sin) { value = std::sin(x); }
                else { value = std::cos(x); }
            } else {
                if (node.Arity != 2 || local < 2) { return std::numeric_limits<double>::quiet_NaN(); }
                auto const rhs = local - 1;
                auto const length = nodes[tree.NodeOffset + rhs].Length;
                if (length > local - 2) { return std::numeric_limits<double>::quiet_NaN(); }
                auto const lhs = local - 2 - length;
                if (node.Op == Opcode::Add) { value = tape[lhs] + tape[rhs]; }
                else if (node.Op == Opcode::Sub) { value = tape[lhs] - tape[rhs]; }
                else if (node.Op == Opcode::Mul) { value = tape[lhs] * tape[rhs]; }
                else if (node.Op == Opcode::Div) { value = tape[lhs] / tape[rhs]; }
                else { return std::numeric_limits<double>::quiet_NaN(); }
            }
            if (!std::isfinite(value)) { return std::numeric_limits<double>::quiet_NaN(); }
            tape[local] = value;
        }
        auto const residual = tape[tree.NodeCount - 1] - target[row];
        auto const weight = weights.empty() ? 1.0 : weights[row];
        if (!std::isfinite(residual) || !std::isfinite(weight) || weight < 0) { return std::numeric_limits<double>::quiet_NaN(); }
        cost += 0.5 * weight * residual * residual;
    }
    return cost;
}
} // namespace
template<typename T>
struct DeviceBuffer {
    explicit DeviceBuffer(sycl::queue& queue) : Queue(&queue) {}
    DeviceBuffer(DeviceBuffer const&) = delete;
    DeviceBuffer(DeviceBuffer&&) = delete;
    auto operator=(DeviceBuffer const&) -> DeviceBuffer& = delete;
    auto operator=(DeviceBuffer&&) -> DeviceBuffer& = delete;
    ~DeviceBuffer() { sycl::free(Data, *Queue); }

    auto Upload(std::span<T const> source) -> T* {
        if (source.size() > Capacity) {
            sycl::free(Data, *Queue);
            Data = sycl::malloc_device<T>(source.size(), *Queue);
            if (Data == nullptr) { throw std::bad_alloc{}; }
            Capacity = source.size();
        }
        Queue->memcpy(Data, source.data(), source.size_bytes());
        return Data;
    }

    auto Download(std::span<T> destination) const -> void {
        Queue->memcpy(destination.data(), Data, destination.size_bytes());
    }

    sycl::queue* Queue;
    T* Data{};
    std::size_t Capacity{};
};

struct Context::Impl {
    sycl::queue Queue{sycl::gpu_selector_v, sycl::property_list{sycl::property::queue::in_order{}}};
    DeviceBuffer<Sycl::detail::DeviceNode> Nodes{Queue};
    DeviceBuffer<Sycl::detail::DeviceTreeRange> Trees{Queue};
    DeviceBuffer<Operon::Scalar> Columns{Queue};
    DeviceBuffer<Operon::Scalar> Target{Queue};
    DeviceBuffer<Operon::Scalar> Weights{Queue};
    DeviceBuffer<Operon::Scalar> Coefficients{Queue};
    DeviceBuffer<Operon::Scalar> Output{Queue};
    DeviceBuffer<uint8_t> Status{Queue};
    DeviceBuffer<uint32_t> Iterations{Queue};
    DeviceBuffer<uint32_t> Accepted{Queue};
    std::size_t TreeCount{};
    std::size_t CoefficientCount{};
};
Context::Context() : impl_(std::make_unique<Impl>()) {}
Context::Context(Context&&) noexcept = default;
auto Context::operator=(Context&&) noexcept -> Context& = default;
Context::~Context() = default;

auto Context::Upload(EncodedPopulation const& population, std::span<Operon::Hash const> variables) -> void
{
    std::vector<Sycl::detail::DeviceNode> nodes;
    std::vector<Sycl::detail::DeviceTreeRange> trees;
    nodes.reserve(population.Nodes.size());
    trees.reserve(population.Trees.size());
    for (auto const& tree : population.Trees) {
        if (tree.NodeCount == 0 || tree.NodeCount > MaxNodes || tree.CoefficientCount > MaxParameters) {
            throw std::invalid_argument("SYCL evaluation received an unsupported tree");
        }
        uint32_t coefficient{};
        for (auto const& node : std::span{population.Nodes}.subspan(tree.NodeOffset, tree.NodeCount)) {
            auto lowered = LowerNode(node, variables);
            if (node.Optimize) { lowered.Coefficient = coefficient++; }
            nodes.push_back({.Value = lowered.Value, .Operand = lowered.Operand, .Coefficient = lowered.Coefficient,
                             .Arity = lowered.Arity, .Length = lowered.Length,
                             .Op = static_cast<Sycl::detail::Opcode>(lowered.Op), .Optimize = lowered.Optimize});
        }
        trees.push_back({tree.NodeOffset, tree.NodeCount, tree.CoefficientOffset, tree.CoefficientCount});
    }
    impl_->Nodes.Upload(nodes);
    impl_->Trees.Upload(trees);
    impl_->TreeCount = trees.size();
    impl_->CoefficientCount = population.Coefficients.size();
    impl_->Queue.wait_and_throw();
}

auto Context::Evaluate(std::span<Operon::Scalar const> columns, std::size_t variableCount,
                       std::size_t rowCount, std::span<Operon::Scalar const> coefficients) -> std::vector<Operon::Scalar>
{
    if (variableCount == 0 || columns.size() != variableCount * rowCount) {
        throw std::invalid_argument("SYCL input must be contiguous [variable][row]");
    }
    auto const treeCount = impl_->TreeCount;
    if (treeCount == 0 || coefficients.size() != impl_->CoefficientCount) {
        throw std::invalid_argument("SYCL evaluation requires an uploaded population and matching coefficients");
    }
    std::vector<Operon::Scalar> output(treeCount * rowCount);
    auto const* deviceColumns = impl_->Columns.Upload(columns);
    auto const* deviceCoefficients = impl_->Coefficients.Upload(coefficients);
    auto* deviceOutput = impl_->Output.Upload(output);
    detail::RunEvaluate(&impl_->Queue, {.Nodes = impl_->Nodes.Data, .Trees = impl_->Trees.Data,
                        .Columns = deviceColumns, .Coefficients = deviceCoefficients, .Output = deviceOutput,
                        .TreeCount = treeCount, .RowCount = rowCount});
    impl_->Output.Download(output);
    impl_->Queue.wait_and_throw();
    return output;
}

auto Context::Supports(Operon::Tree const& tree, Operon::Span<Operon::Hash const> variables) const -> bool
{
    if (!tree.Validate() || tree.CoefficientsCount() == 0 || tree.CoefficientsCount() > MaxParameters || tree.Length() > MaxNodes) { return false; }
    try {
        for (auto const& node : tree.Nodes()) {
            (void)LowerNode({.HashValue = node.HashValue, .Value = node.Value, .RefTo = node.RefTo, .Arity = node.Arity,
                             .Length = node.Length, .Type = node.Type, .IsEnabled = node.IsEnabled, .Optimize = node.Optimize}, variables);
        }
    } catch (std::invalid_argument const&) { return false; }
    return true;
}

auto Context::Optimize(EncodedPopulation const& population, Operon::Span<Operon::Hash const> variables,
                       Operon::Span<Operon::Scalar const> columns, std::size_t variableCount, std::size_t rowCount,
                       Operon::Span<Operon::Scalar const> target, Operon::Span<Operon::Scalar const> weights,
                       uint32_t maxIterations) -> Operon::PopulationLocalSearchResult
{
    static_assert(std::same_as<Operon::Scalar, float>, "SYCL backend requires USE_SINGLE_PRECISION");
    if (population.Trees.empty() || variableCount == 0 || columns.size() != variableCount * rowCount
        || target.size() != rowCount || (!weights.empty() && weights.size() != rowCount) || maxIterations == 0) {
        throw std::invalid_argument("SYCL optimizer requires trees, contiguous inputs, target, optional weights, and iterations");
    }
    std::vector<HostDeviceNode> hostNodes;
    std::vector<HostDeviceTreeRange> hostTrees;
    std::vector<Sycl::detail::DeviceNode> nodes;
    std::vector<Sycl::detail::DeviceTreeRange> trees;
    hostNodes.reserve(population.Nodes.size()); hostTrees.reserve(population.Trees.size());
    nodes.reserve(population.Nodes.size()); trees.reserve(population.Trees.size());
    for (auto const& tree : population.Trees) {
        if (tree.NodeCount > MaxNodes || tree.CoefficientCount == 0 || tree.CoefficientCount > MaxParameters) {
            throw std::invalid_argument("SYCL optimizer received an unsupported tree");
        }
        uint32_t coefficient{};
        for (auto const& node : std::span{population.Nodes}.subspan(tree.NodeOffset, tree.NodeCount)) {
            auto lowered = LowerNode(node, variables);
            if (node.Optimize) { lowered.Coefficient = coefficient++; }
            nodes.push_back({.Value = lowered.Value, .Operand = lowered.Operand, .Coefficient = lowered.Coefficient,
                             .Arity = lowered.Arity, .Length = lowered.Length,
                             .Op = static_cast<Sycl::detail::Opcode>(lowered.Op), .Optimize = lowered.Optimize});
            hostNodes.push_back(lowered);
        }
        hostTrees.push_back({tree.NodeOffset, tree.NodeCount, tree.CoefficientOffset, tree.CoefficientCount});
        trees.push_back({tree.NodeOffset, tree.NodeCount, tree.CoefficientOffset, tree.CoefficientCount});
    }

    // The first SYCL backend deliberately uses one work-item per heterogeneous
    // tree. It establishes identical device semantics; later work can replace
    // this kernel with subgroup reductions without changing the public contract.
    auto coefficients = population.Coefficients;
    std::vector<uint8_t> status(trees.size(), 1);
    auto const emptyWeight = Operon::Scalar{};
    auto const* weightData = weights.empty() ? &emptyWeight : weights.data();
    auto const weightCount = weights.empty() ? std::size_t{1} : weights.size();
    std::vector<uint32_t> iterations(trees.size()), accepted(trees.size());
    auto const* deviceNodes = impl_->Nodes.Upload(nodes);
    auto const* deviceTrees = impl_->Trees.Upload(trees);
    auto const* deviceColumns = impl_->Columns.Upload(columns);
    auto const* deviceTarget = impl_->Target.Upload(target);
    auto const* deviceWeights = impl_->Weights.Upload({weightData, weightCount});
    auto* deviceCoefficients = impl_->Coefficients.Upload(coefficients);
    auto* deviceStatus = impl_->Status.Upload(status);
    auto* deviceIterations = impl_->Iterations.Upload(iterations);
    auto* deviceAccepted = impl_->Accepted.Upload(accepted);
    auto const hasWeights = !weights.empty();
    detail::RunKernel(&impl_->Queue, {
        .Nodes = deviceNodes,
        .Trees = deviceTrees,
        .Columns = deviceColumns,
        .Target = deviceTarget,
        .Weights = deviceWeights,
        .Coefficients = deviceCoefficients,
        .Status = deviceStatus,
        .Iterations = deviceIterations,
        .Accepted = deviceAccepted,
        .TreeCount = trees.size(),
        .RowCount = rowCount,
        .MaxIterations = maxIterations,
        .HasWeights = hasWeights,
    });
    impl_->Queue.wait_and_throw();
    impl_->Coefficients.Download(coefficients);
    impl_->Status.Download(status);
    impl_->Iterations.Download(iterations);
    impl_->Accepted.Download(accepted);
    impl_->Queue.wait_and_throw();

    Operon::PopulationLocalSearchResult result;
    result.Coefficients = std::move(coefficients);
    result.Status.resize(trees.size()); result.InitialCosts.resize(trees.size()); result.FinalCosts.resize(trees.size());
    result.Iterations = std::move(iterations); result.AcceptedSteps = std::move(accepted);
    for (std::size_t i = 0; i < trees.size(); ++i) {
        auto const initial = TrustedCost(hostNodes, hostTrees[i], columns, variableCount, rowCount, target, weights, population.Coefficients);
        auto const final = TrustedCost(hostNodes, hostTrees[i], columns, variableCount, rowCount, target, weights, result.Coefficients);
        result.InitialCosts[i] = initial;
        if (status[i] != 0 && std::isfinite(final) && final < initial) {
            result.FinalCosts[i] = final; result.Status[i] = PopulationLocalSearchStatus::Improved;
        } else {
            auto const tree = trees[i];
            std::copy_n(population.Coefficients.begin() + tree.CoefficientOffset, tree.CoefficientCount,
                        result.Coefficients.begin() + tree.CoefficientOffset);
            result.FinalCosts[i] = initial; result.Status[i] = PopulationLocalSearchStatus::Retained;
        }
    }
    return result;
}
} // namespace Operon::PopulationOptimization::Sycl
