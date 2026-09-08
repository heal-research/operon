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
#include "operon/optimizer/hip_context.hpp"
#include "operon/operators/local_search.hpp"
#include "operon/operators/evaluator.hpp"

namespace Operon::PopulationOptimization::Hip {

namespace {
[[nodiscard]] auto LowerNode(EncodedNode const& node, std::span<Operon::Hash const> variables) -> detail::DeviceNode
{
    auto opcode = detail::Opcode::Constant;
    uint32_t operand{};
    if (node.Type == NodeType::Variable) {
        auto const it = std::ranges::find(variables, node.HashValue);
        if (it == variables.end()) { throw std::invalid_argument("HIP population contains an unbound variable"); }
        opcode = detail::Opcode::Variable;
        operand = static_cast<uint32_t>(std::distance(variables.begin(), it));
    } else if (node.Type == NodeType::Ref) {
        opcode = detail::Opcode::Ref;
        operand = node.RefTo;
    } else if (node.Type == NodeType::Function) {
        switch (static_cast<BuiltinOp>(node.HashValue)) {
        case BuiltinOp::Add: opcode = detail::Opcode::Add; break;
        case BuiltinOp::Sub: opcode = detail::Opcode::Sub; break;
        case BuiltinOp::Mul: opcode = detail::Opcode::Mul; break;
        case BuiltinOp::Div: opcode = detail::Opcode::Div; break;
        case BuiltinOp::Square: opcode = detail::Opcode::Square; break;
        case BuiltinOp::Exp: opcode = detail::Opcode::Exp; break;
        case BuiltinOp::Log: opcode = detail::Opcode::Log; break;
        case BuiltinOp::Sin: opcode = detail::Opcode::Sin; break;
        case BuiltinOp::Cos: opcode = detail::Opcode::Cos; break;
        default: throw std::invalid_argument("HIP population contains an unsupported primitive");
        }
    } else if (node.Type != NodeType::Constant) {
        throw std::invalid_argument("HIP population contains an unsupported node type");
    }
    return {
        .Value = node.Value,
        .Operand = operand,
        .Coefficient = 0,
        .Arity = node.Arity,
        .Length = node.Length,
        .Op = opcode,
        .Optimize = static_cast<uint8_t>(node.Optimize),
    };
}

constexpr auto MaxNodes = 128U;

[[nodiscard]] auto TrustedGaussianCost(std::span<detail::DeviceNode const> nodes,
                                       detail::DeviceTreeRange tree,
                                       std::span<Operon::Scalar const> columns,
                                       std::size_t variableCount,
                                       std::size_t rowCount,
                                       std::span<Operon::Scalar const> target,
                                       std::span<Operon::Scalar const> weights,
                                       std::span<Operon::Scalar const> coefficients) noexcept -> double
{
    auto const invalid = std::numeric_limits<double>::quiet_NaN();
    auto const nodeBegin = static_cast<std::size_t>(tree.NodeOffset);
    auto const nodeCount = static_cast<std::size_t>(tree.NodeCount);
    auto const coefficientBegin = static_cast<std::size_t>(tree.CoefficientOffset);
    auto const coefficientCount = static_cast<std::size_t>(tree.CoefficientCount);
    if (nodeCount == 0 || nodeCount > MaxNodes || nodeBegin > nodes.size() || nodeCount > nodes.size() - nodeBegin
        || coefficientBegin > coefficients.size() || coefficientCount > coefficients.size() - coefficientBegin) {
        return invalid;
    }

    auto cost = 0.0;
    std::array<double, MaxNodes> tape{};
    for (std::size_t row = 0; row < rowCount; ++row) {
        for (std::size_t local = 0; local < nodeCount; ++local) {
            auto const& node = nodes[nodeBegin + local];
            auto const coefficient = [&]() -> double {
                auto const index = coefficientBegin + static_cast<std::size_t>(node.Coefficient);
                return index < coefficientBegin || index >= coefficientBegin + coefficientCount ? invalid : static_cast<double>(coefficients[index]);
            };
            auto value = invalid;
            switch (node.Op) {
            case detail::Opcode::Constant:
                if (node.Arity != 0) { return invalid; }
                value = node.Optimize ? coefficient() : static_cast<double>(node.Value);
                break;
            case detail::Opcode::Variable:
                if (node.Arity != 0 || node.Operand >= variableCount) { return invalid; }
                value = static_cast<double>(columns[static_cast<std::size_t>(node.Operand) * rowCount + row]);
                value *= node.Optimize ? coefficient() : static_cast<double>(node.Value);
                break;
            case detail::Opcode::Ref:
                if (node.Arity != 0 || node.Optimize != 0 || node.Operand >= local) { return invalid; }
                value = tape[node.Operand];
                break;
            case detail::Opcode::Add:
            case detail::Opcode::Sub:
            case detail::Opcode::Mul:
            case detail::Opcode::Div: {
                if (node.Arity != 2 || node.Optimize != 0 || local < 2) { return invalid; }
                auto const rhs = local - 1;
                auto const rhsLength = static_cast<std::size_t>(nodes[nodeBegin + rhs].Length);
                if (rhsLength > local - 2) { return invalid; }
                auto const lhs = local - 2 - rhsLength;
                if (node.Op == detail::Opcode::Add) { value = tape[lhs] + tape[rhs]; }
                else if (node.Op == detail::Opcode::Sub) { value = tape[lhs] - tape[rhs]; }
                else if (node.Op == detail::Opcode::Mul) { value = tape[lhs] * tape[rhs]; }
                else { value = tape[lhs] / tape[rhs]; }
                break;
            }
            case detail::Opcode::Square:
            case detail::Opcode::Exp:
            case detail::Opcode::Log:
            case detail::Opcode::Sin:
            case detail::Opcode::Cos: {
                if (node.Arity != 1 || node.Optimize != 0 || local == 0) { return invalid; }
                auto const x = tape[local - 1];
                if (node.Op == detail::Opcode::Square) { value = x * x; }
                else if (node.Op == detail::Opcode::Exp) { value = std::exp(x); }
                else if (node.Op == detail::Opcode::Log) { value = std::log(x); }
                else if (node.Op == detail::Opcode::Sin) { value = std::sin(x); }
                else { value = std::cos(x); }
                break;
            }
            default: return invalid;
            }
            if (!std::isfinite(value)) { return invalid; }
            tape[local] = value;
        }
        auto const residual = tape[nodeCount - 1] - static_cast<double>(target[row]);
        auto const weight = weights.empty() ? 1.0 : static_cast<double>(weights[row]);
        if (!std::isfinite(residual) || !std::isfinite(weight) || weight < 0.0) { return invalid; }
        cost += 0.5 * weight * residual * residual;
        if (!std::isfinite(cost)) { return invalid; }
    }
    return cost;
}
} // namespace

struct Context::Impl {
    detail::Runtime* Runtime{detail::Create()};
    std::vector<detail::DeviceNode> Nodes;
    std::vector<detail::DeviceTreeRange> Trees;
    std::vector<Operon::Scalar> Coefficients;

    ~Impl() { detail::Destroy(Runtime); }
};

Context::Context()
    : impl_(std::make_unique<Impl>())
{
    if (impl_->Runtime == nullptr) { throw std::runtime_error("unable to create HIP runtime"); }
}

Context::Context(Context&&) noexcept = default;
auto Context::operator=(Context&&) noexcept -> Context& = default;
Context::~Context() = default;

auto Context::Upload(EncodedPopulation const& population, std::span<Operon::Hash const> variableHashes) -> void
{
    static_assert(std::same_as<Operon::Scalar, float>, "initial HIP backend requires USE_SINGLE_PRECISION");
    impl_->Nodes.clear();
    impl_->Trees.clear();
    impl_->Nodes.reserve(population.Nodes.size());
    impl_->Coefficients = population.Coefficients;
    impl_->Trees.reserve(population.Trees.size());
    for (auto const& tree : population.Trees) {
        uint32_t coefficientIndex{};
        auto const nodes = std::span{population.Nodes}.subspan(tree.NodeOffset, tree.NodeCount);
        for (auto const& node : nodes) {
            auto lowered = LowerNode(node, variableHashes);
            if (node.Optimize) { lowered.Coefficient = coefficientIndex++; }
            impl_->Nodes.push_back(lowered);
        }
        impl_->Trees.push_back({tree.NodeOffset, tree.NodeCount, tree.CoefficientOffset, tree.CoefficientCount});
    }
    detail::Upload(impl_->Runtime, impl_->Nodes.data(), impl_->Nodes.size(), impl_->Trees.data(), impl_->Trees.size());
    detail::UpdateCoefficients(impl_->Runtime, impl_->Coefficients.data(), impl_->Coefficients.size());
}

auto Context::UpdateCoefficients(std::span<Operon::Scalar const> coefficients) -> void
{
    if (coefficients.size() != impl_->Coefficients.size()) {
        throw std::invalid_argument("HIP coefficient update must match the uploaded population");
    }
    impl_->Coefficients.assign(coefficients.begin(), coefficients.end());
    detail::UpdateCoefficients(impl_->Runtime, impl_->Coefficients.data(), impl_->Coefficients.size());
}

auto Context::JacRev(std::span<Operon::Scalar const> columns, std::size_t variableCount,
                     std::size_t rowCount, std::span<Operon::Scalar const> coefficients) -> std::vector<Operon::Scalar>
{
    if (variableCount == 0 || columns.size() != variableCount * rowCount) {
        throw std::invalid_argument("HIP input must be contiguous [variable][row]");
    }
    auto coefficientCount = std::size_t{};
    for (auto const& tree : impl_->Trees) { coefficientCount = std::max(coefficientCount, static_cast<std::size_t>(tree.CoefficientOffset + tree.CoefficientCount)); }
    std::vector<Operon::Scalar> jacobian(coefficientCount * rowCount);
    detail::JacRev(impl_->Runtime, columns.data(), variableCount, rowCount,
                   coefficients.data(), coefficients.size(), jacobian.data());
    return jacobian;
}

auto Context::Evaluate(std::span<Operon::Scalar const> columns, std::size_t variableCount,
                       std::size_t rowCount, std::span<Operon::Scalar const> coefficients) -> std::vector<Operon::Scalar>
{
    UpdateCoefficients(coefficients);
    return EvaluateResident(columns, variableCount, rowCount);
}

auto Context::EvaluateResident(std::span<Operon::Scalar const> columns, std::size_t variableCount,
                               std::size_t rowCount) -> std::vector<Operon::Scalar>
{
    if (variableCount == 0 || columns.size() != variableCount * rowCount) {
        throw std::invalid_argument("HIP input must be contiguous [variable][row]");
    }
    std::vector<Operon::Scalar> output(impl_->Trees.size() * rowCount);
    detail::EvaluateResident(impl_->Runtime, columns.data(), variableCount, rowCount, output.data());
    return output;
}

auto Context::GaussianCosts(std::span<Operon::Scalar const> columns, std::size_t variableCount,
                            std::size_t rowCount, std::span<Operon::Scalar const> target,
                            std::span<Operon::Scalar const> weights)
    -> std::pair<std::vector<Operon::Scalar>, std::vector<uint8_t>>
{
    if (variableCount == 0 || columns.size() != variableCount * rowCount || target.size() != rowCount
        || (!weights.empty() && weights.size() != rowCount)) {
        throw std::invalid_argument("HIP Gaussian costs require contiguous inputs, target, and optional row weights");
    }
    std::vector<Operon::Scalar> costs(impl_->Trees.size());
    std::vector<uint8_t> valid(impl_->Trees.size());
    detail::GaussianCosts(impl_->Runtime, columns.data(), variableCount, rowCount, target.data(), weights.data(), !weights.empty(), costs.data(), valid.data());
    return {std::move(costs), std::move(valid)};
}

auto Context::NormalEquations(std::span<Operon::Scalar const> columns, std::size_t variableCount,
                              std::size_t rowCount, std::span<Operon::Scalar const> target,
                              std::span<Operon::Scalar const> weights) -> NormalEquationDiagnostics
{
    if (variableCount == 0 || columns.size() != variableCount * rowCount || target.size() != rowCount
        || (!weights.empty() && weights.size() != rowCount)) {
        throw std::invalid_argument("HIP normal equations require contiguous inputs, target, and optional row weights");
    }
    auto coefficientCount = std::size_t{};
    for (auto const& tree : impl_->Trees) {
        coefficientCount = std::max(coefficientCount, static_cast<std::size_t>(tree.CoefficientOffset + tree.CoefficientCount));
    }
    NormalEquationDiagnostics result;
    result.Normal.resize(impl_->Trees.size() * 16U * 16U);
    result.Gradient.resize(impl_->Trees.size() * 16U);
    result.Costs.resize(impl_->Trees.size());
    result.Valid.resize(impl_->Trees.size());
    detail::NormalEquations(impl_->Runtime, columns.data(), variableCount, rowCount, target.data(), weights.data(), !weights.empty(),
                            impl_->Coefficients.data(), coefficientCount, result.Normal.data(), result.Gradient.data(),
                            result.Costs.data(), result.Valid.data());
    return result;
}

auto Context::Supports(Operon::Tree const& tree, Operon::Span<Operon::Hash const> variableHashes) const -> bool
{
    if (!tree.Validate() || tree.CoefficientsCount() == 0 || tree.CoefficientsCount() > 16 || tree.Length() > MaxNodes) { return false; }
    try {
        for (auto const& node : tree.Nodes()) { (void) LowerNode(EncodedNode{.HashValue = node.HashValue, .Value = node.Value, .RefTo = node.RefTo, .Arity = node.Arity, .Length = node.Length, .Type = node.Type, .IsEnabled = node.IsEnabled, .Optimize = node.Optimize}, variableHashes); }
    } catch (std::invalid_argument const&) {
        return false;
    }
    return true;
}

auto Context::Optimize(EncodedPopulation const& population,
                       Operon::Span<Operon::Hash const> variableHashes,
                       Operon::Span<Operon::Scalar const> columns,
                       std::size_t variableCount,
                       std::size_t rowCount,
                       Operon::Span<Operon::Scalar const> target,
                       Operon::Span<Operon::Scalar const> weights,
                       uint32_t maxIterations) -> Operon::PopulationLocalSearchResult
{
    Upload(population, variableHashes);
    auto result = OptimizeGaussian(columns, variableCount, rowCount, target, weights, maxIterations);
    Operon::PopulationLocalSearchResult out;
    out.Coefficients = std::move(result.Coefficients);
    out.InitialCosts = std::move(result.InitialCosts);
    out.FinalCosts = std::move(result.FinalCosts);
    out.Iterations = std::move(result.Iterations);
    out.AcceptedSteps = std::move(result.AcceptedSteps);
    out.Status.reserve(result.Status.size());
    for (auto const status : result.Status) {
        switch (status) {
        case OptimizationStatus::Improved: out.Status.push_back(Operon::PopulationLocalSearchStatus::Improved); break;
        case OptimizationStatus::InvalidInput: out.Status.push_back(Operon::PopulationLocalSearchStatus::InvalidInput); break;
        case OptimizationStatus::Unsupported: out.Status.push_back(Operon::PopulationLocalSearchStatus::Unsupported); break;
        case OptimizationStatus::Retained: out.Status.push_back(Operon::PopulationLocalSearchStatus::Retained); break;
        }
    }
    return out;
}
auto Context::OptimizeGaussian(std::span<Operon::Scalar const> columns, std::size_t variableCount,
                               std::size_t rowCount, std::span<Operon::Scalar const> target,
                               std::span<Operon::Scalar const> weights, uint32_t maxIterations) -> OptimizationResult
{
    if (variableCount == 0 || columns.size() != variableCount * rowCount || target.size() != rowCount
        || (!weights.empty() && weights.size() != rowCount) || maxIterations == 0) {
        throw std::invalid_argument("HIP optimizer requires contiguous inputs, target, optional row weights, and iterations");
    }
    if (!std::ranges::all_of(target, [](auto x) { return std::isfinite(x); })
        || !std::ranges::all_of(weights, [](auto x) { return std::isfinite(x) && x >= Operon::Scalar{}; })) {
        throw std::invalid_argument("HIP optimizer requires finite targets and non-negative finite weights");
    }
    auto coefficientCount = std::size_t{};
    for (auto const& tree : impl_->Trees) {
        if (tree.CoefficientCount > 16) { throw std::invalid_argument("HIP LM supports at most 16 coefficients per tree"); }
        coefficientCount = std::max(coefficientCount, static_cast<std::size_t>(tree.CoefficientOffset + tree.CoefficientCount));
    }
    if (coefficientCount == 0) { throw std::invalid_argument("HIP optimizer requires fitted coefficients"); }
    OptimizationResult result;
    result.Coefficients.resize(coefficientCount);
    result.InitialCosts.resize(impl_->Trees.size());
    result.FinalCosts.resize(impl_->Trees.size());
    std::vector<Operon::Scalar> deviceInitialCosts(impl_->Trees.size());
    std::vector<Operon::Scalar> deviceFinalCosts(impl_->Trees.size());
    std::vector<uint8_t> rawStatus(impl_->Trees.size());
    result.Status.resize(impl_->Trees.size());
    result.Iterations.resize(impl_->Trees.size());
    result.AcceptedSteps.resize(impl_->Trees.size());
    detail::OptimizeGaussian(impl_->Runtime, columns.data(), variableCount, rowCount, target.data(),
                             weights.data(), !weights.empty(), impl_->Coefficients.data(), impl_->Coefficients.size(), maxIterations,
                             result.Coefficients.data(), deviceInitialCosts.data(), deviceFinalCosts.data(),
                             rawStatus.data(), result.Iterations.data(), result.AcceptedSteps.data());

    // The device owns all iteration and fp32 acceptance decisions. This one-shot
    // delivery gate independently evaluates the lowered P2 tree in fp64 before
    // exposing any candidate to callers.
    for (std::size_t i = 0; i < impl_->Trees.size(); ++i) {
        auto const tree = impl_->Trees[i];
        auto const initial = TrustedGaussianCost(impl_->Nodes, tree, columns, variableCount, rowCount,
                                                 target, weights, impl_->Coefficients);
        auto const candidate = TrustedGaussianCost(impl_->Nodes, tree, columns, variableCount, rowCount,
                                                   target, weights, result.Coefficients);
        result.InitialCosts[i] = initial;
        if (!std::isfinite(candidate) || !(candidate < initial)) {
            auto const offset = static_cast<std::size_t>(tree.CoefficientOffset);
            auto const count = static_cast<std::size_t>(tree.CoefficientCount);
            std::copy(impl_->Coefficients.begin() + static_cast<std::ptrdiff_t>(offset),
                      impl_->Coefficients.begin() + static_cast<std::ptrdiff_t>(offset + count),
                      result.Coefficients.begin() + static_cast<std::ptrdiff_t>(offset));
            result.FinalCosts[i] = initial;
            result.Status[i] = OptimizationStatus::Retained;
        } else {
            result.FinalCosts[i] = candidate;
            result.Status[i] = OptimizationStatus::Improved;
        }
    }
    impl_->Coefficients = result.Coefficients;
    detail::UpdateCoefficients(impl_->Runtime, impl_->Coefficients.data(), impl_->Coefficients.size());
    return result;
}

void GaussianPopulationOffspringScorer::Score(std::span<Operon::Individual> candidates,
                                              std::span<Operon::RandomGenerator> random,
                                              Operon::EvaluatorBase const& evaluator,
                                              Operon::CoefficientOptimizer const* optimizer,
                                              double localSearchProbability,
                                              double lamarckianProbability,
                                              std::span<Operon::Vector<Operon::Scalar>> scratch)
{
    auto const* gaussian = dynamic_cast<Operon::Evaluator<Operon::ScalarDispatch> const*>(&evaluator);
    if (gaussian == nullptr || gaussian->HasLinearScaling() || gaussian->SkipsNonFinite()) {
        throw std::invalid_argument("HIP population scorer requires the plain scalar Gaussian evaluator without linear scaling or non-finite omission");
    }
    switch (gaussian->Error().Type()) {
    case Operon::ErrorType::SSE:
    case Operon::ErrorType::MSE:
    case Operon::ErrorType::NMSE:
    case Operon::ErrorType::RMSE: break;
    default: throw std::invalid_argument("HIP population scorer supports only SSE, MSE, NMSE, and RMSE");
    }
    if (candidates.size() != random.size() || candidates.size() != scratch.size()) {
        throw std::invalid_argument("HIP population scorer requires matching candidate, RNG, and scratch spans");
    }
    auto const* problem = evaluator.GetProblem();
    auto const range = problem->TrainingRange();
    auto const& variables = problem->GetInputs();
    if (variables.empty()) { throw std::invalid_argument("HIP population scorer requires at least one input variable"); }
    std::vector<std::optional<std::vector<Operon::Scalar>>> original(candidates.size());
    if (optimizer != nullptr && localSearchProbability > 0.0 && optimizer->Iterations() > 0) {
        LocalSearchPopulation(candidates, random, evaluator, optimizer, localSearchProbability, lamarckianProbability,
                              context_, static_cast<uint32_t>(optimizer->Iterations()), original);
    }

    std::vector<std::size_t> selected(candidates.size());
    std::iota(selected.begin(), selected.end(), 0);
    auto encoded = Operon::PopulationOptimization::EncodePopulation(candidates, selected);
    if (!encoded) { throw std::invalid_argument("unable to encode HIP population scorer candidates"); }
    context_.Upload(*encoded, variables);
    std::vector<Operon::Scalar> columns;
    columns.reserve(variables.size() * range.Size());
    for (auto const hash : variables) {
        auto const values = problem->GetDataset()->GetValues(hash).subspan(range.Start(), range.Size());
        columns.insert(columns.end(), values.begin(), values.end());
    }
    auto const target = problem->TargetValues(range);
    auto const weights = problem->Weights(range).value_or(Operon::Span<Operon::Scalar const>{});
    auto const [costs, valid] = context_.GaussianCosts(columns, variables.size(), range.Size(), target, weights);
    auto sumWeights = 0.0;
    auto mean = 0.0;
    auto m2 = 0.0;
    for (std::size_t row = 0; row < range.Size(); ++row) {
        auto const weight = weights.empty() ? 1.0 : static_cast<double>(weights[row]);
        auto const next = sumWeights + weight;
        auto const delta = static_cast<double>(target[row]) - mean;
        mean += weight * delta / next;
        m2 += sumWeights * delta * weight / next;
        sumWeights = next;
    }
    auto const variance = m2 / sumWeights;
    for (std::size_t index = 0; index < candidates.size(); ++index) {
        auto const mse = static_cast<double>(2 * costs[index]) / sumWeights;
        auto fit = std::numeric_limits<Operon::Scalar>::quiet_NaN();
        switch (gaussian->Error().Type()) {
        case Operon::ErrorType::SSE: fit = 2 * costs[index]; break;
        case Operon::ErrorType::MSE: fit = static_cast<Operon::Scalar>(mse); break;
        case Operon::ErrorType::RMSE: fit = static_cast<Operon::Scalar>(std::sqrt(mse)); break;
        case Operon::ErrorType::NMSE: fit = static_cast<Operon::Scalar>(mse / variance); break;
        default: break;
        }
        candidates[index].Fitness = {valid[index] != 0 && std::isfinite(fit) ? fit : Operon::EvaluatorBase::ErrMax};
        ++evaluator.CallCount;
        ++evaluator.ResidualEvaluations;
        if (original[index]) { candidates[index].Genotype.SetCoefficients(*original[index]); }
    }
}

} // namespace Operon::PopulationOptimization::Hip
