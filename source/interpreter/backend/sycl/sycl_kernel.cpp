// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#include <limits>

#include <sycl/sycl.hpp>

#include "operon/optimizer/detail/sycl_kernel.hpp"

namespace Operon::PopulationOptimization::Sycl::detail {
namespace {
constexpr auto MaxNodes = 128U;
constexpr auto MaxParameters = 16U;
constexpr auto NaN = std::numeric_limits<float>::quiet_NaN();
} // namespace
class EvaluationKernel;

void RunEvaluate(void* opaqueQueue, EvaluationArgs args)
{
    auto& queue = *static_cast<sycl::queue*>(opaqueQueue);
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for<EvaluationKernel>(sycl::range<1>{args.TreeCount * args.RowCount}, [=](sycl::id<1> id) {
            auto const treeIndex = static_cast<std::size_t>(id[0]) / args.RowCount;
            auto const row = static_cast<std::size_t>(id[0]) % args.RowCount;
            auto const tree = args.Trees[treeIndex];
            float tape[MaxNodes]{};
            auto valid = tree.NodeCount > 0 && tree.NodeCount <= MaxNodes;
            for (uint32_t local = 0; local < tree.NodeCount; ++local) {
                auto const node = args.Nodes[tree.NodeOffset + local];
                auto value = NaN;
                if (node.Op == Opcode::Constant) { value = node.Optimize ? args.Coefficients[tree.CoefficientOffset + node.Coefficient] : node.Value; }
                else if (node.Op == Opcode::Variable) { value = args.Columns[static_cast<std::size_t>(node.Operand) * args.RowCount + row] * (node.Optimize ? args.Coefficients[tree.CoefficientOffset + node.Coefficient] : node.Value); }
                else if (node.Op == Opcode::Ref) { value = node.Operand < local ? tape[node.Operand] : NaN; }
                else if (node.Op == Opcode::Square) { value = node.Arity == 1 && local > 0 ? tape[local - 1] * tape[local - 1] : NaN; }
                else if (node.Op == Opcode::Exp) { value = node.Arity == 1 && local > 0 ? sycl::exp(tape[local - 1]) : NaN; }
                else if (node.Op == Opcode::Log) { value = node.Arity == 1 && local > 0 ? sycl::log(tape[local - 1]) : NaN; }
                else if (node.Op == Opcode::Sin) { value = node.Arity == 1 && local > 0 ? sycl::sin(tape[local - 1]) : NaN; }
                else if (node.Op == Opcode::Cos) { value = node.Arity == 1 && local > 0 ? sycl::cos(tape[local - 1]) : NaN; }
                else if (node.Arity == 2 && local >= 2) {
                    auto const lhs = local - 1;
                    auto const length = args.Nodes[tree.NodeOffset + lhs].Length;
                    if (length <= local - 2) {
                        auto const rhs = local - 2 - length;
                        if (node.Op == Opcode::Add) { value = tape[lhs] + tape[rhs]; }
                        else if (node.Op == Opcode::Sub) { value = tape[lhs] - tape[rhs]; }
                        else if (node.Op == Opcode::Mul) { value = tape[lhs] * tape[rhs]; }
                        else if (node.Op == Opcode::Div) { value = tape[lhs] / tape[rhs]; }
                    }
                }
                valid = valid && sycl::isfinite(value);
                tape[local] = value;
            }
            args.Output[treeIndex * args.RowCount + row] = valid ? tape[tree.NodeCount - 1] : NaN;
        });
    });
}

void RunKernel(void* opaqueQueue, KernelArgs args)
{
    auto& queue = *static_cast<sycl::queue*>(opaqueQueue);
    queue.submit([&](sycl::handler& handler) {
        handler.parallel_for(sycl::range<1>{args.TreeCount}, [=](sycl::id<1> id) {
            auto const index = id[0]; auto const tree = args.Trees[index];
            float current[MaxParameters]{};
            for (uint32_t i = 0; i < tree.CoefficientCount; ++i) { current[i] = args.Coefficients[tree.CoefficientOffset + i]; }
            auto costAt = [&](float const* parameters, float* normal, float* gradient) {
                auto cost = 0.0F; auto valid = true;
                for (std::size_t row = 0; row < args.RowCount; ++row) {
                    float tape[MaxNodes]{}; float adjoint[MaxNodes]{}; float derivative[MaxParameters]{};
                    for (uint32_t local = 0; local < tree.NodeCount; ++local) {
                        auto const node = args.Nodes[tree.NodeOffset + local]; float value = NaN;
                        if (node.Op == Opcode::Constant) { value = node.Optimize ? parameters[node.Coefficient] : node.Value; }
                        else if (node.Op == Opcode::Variable) { value = args.Columns[static_cast<std::size_t>(node.Operand) * args.RowCount + row] * (node.Optimize ? parameters[node.Coefficient] : node.Value); }
                        else if (node.Op == Opcode::Ref) { value = node.Operand < local ? tape[node.Operand] : NaN; }
                        else if (node.Op == Opcode::Square) { value = node.Arity == 1 && local > 0 ? tape[local - 1] * tape[local - 1] : NaN; }
                        else if (node.Op == Opcode::Exp) { value = node.Arity == 1 && local > 0 ? sycl::exp(tape[local - 1]) : NaN; }
                        else if (node.Op == Opcode::Log) { value = node.Arity == 1 && local > 0 ? sycl::log(tape[local - 1]) : NaN; }
                        else if (node.Op == Opcode::Sin) { value = node.Arity == 1 && local > 0 ? sycl::sin(tape[local - 1]) : NaN; }
                        else if (node.Op == Opcode::Cos) { value = node.Arity == 1 && local > 0 ? sycl::cos(tape[local - 1]) : NaN; }
                        else if (node.Arity == 2 && local >= 2) {
                            auto const lhs = local - 1;
                            auto const length = args.Nodes[tree.NodeOffset + lhs].Length;
                            if (length <= local - 2) {
                                auto const rhs = local - 2 - length;
                                if (node.Op == Opcode::Add) { value = tape[lhs] + tape[rhs]; } else if (node.Op == Opcode::Sub) { value = tape[lhs] - tape[rhs]; }
                                else if (node.Op == Opcode::Mul) { value = tape[lhs] * tape[rhs]; } else if (node.Op == Opcode::Div) { value = tape[lhs] / tape[rhs]; }
                            }
                        }
                        tape[local] = value; valid = valid && sycl::isfinite(value);
                    }
                    auto const residual = tape[tree.NodeCount - 1] - args.Target[row]; auto const weight = args.HasWeights ? args.Weights[row] : 1.0F;
                    valid = valid && sycl::isfinite(residual) && sycl::isfinite(weight) && weight >= 0.0F;
                    cost += 0.5F * weight * residual * residual;
                    if (normal != nullptr) {
                        adjoint[tree.NodeCount - 1] = 1.0F;
                        for (int local = static_cast<int>(tree.NodeCount) - 1; local >= 0; --local) {
                            auto const node = args.Nodes[tree.NodeOffset + static_cast<uint32_t>(local)];
                            auto const a = adjoint[local];
                            if (node.Optimize) {
                                if (node.Coefficient >= tree.CoefficientCount) { valid = false; continue; }
                                auto const factor = node.Op == Opcode::Variable
                                    ? args.Columns[static_cast<std::size_t>(node.Operand) * args.RowCount + row] : 1.0F;
                                derivative[node.Coefficient] = a * factor;
                            }
                            if (node.Op == Opcode::Ref) {
                                if (node.Operand >= static_cast<uint32_t>(local)) { valid = false; continue; }
                                adjoint[node.Operand] += a;
                                continue;
                            }
                            if (node.Op == Opcode::Square || node.Op == Opcode::Exp || node.Op == Opcode::Log || node.Op == Opcode::Sin || node.Op == Opcode::Cos) {
                                if (node.Arity != 1 || local == 0) { valid = false; continue; }
                                auto const child = static_cast<uint32_t>(local) - 1;
                                auto const x = tape[child];
                                auto const localDerivative = node.Op == Opcode::Square ? 2.0F * x : node.Op == Opcode::Exp ? tape[static_cast<uint32_t>(local)] : node.Op == Opcode::Log ? 1.0F / x : node.Op == Opcode::Sin ? sycl::cos(x) : -sycl::sin(x);
                                adjoint[child] += a * localDerivative;
                                continue;
                            }
                            if (node.Op < Opcode::Add || node.Arity != 2 || local < 2) { valid = false; continue; }
                            auto const lhs = static_cast<uint32_t>(local) - 1;
                            auto const length = args.Nodes[tree.NodeOffset + lhs].Length;
                            if (length > static_cast<uint32_t>(local) - 2) { valid = false; continue; }
                            auto const rhs = static_cast<uint32_t>(local) - 2 - length;
                            if (node.Op == Opcode::Add) { adjoint[lhs] += a; adjoint[rhs] += a; } else if (node.Op == Opcode::Sub) { adjoint[lhs] += a; adjoint[rhs] -= a; }
                            else if (node.Op == Opcode::Mul) { adjoint[lhs] += a * tape[rhs]; adjoint[rhs] += a * tape[lhs]; }
                            else { adjoint[lhs] += a / tape[rhs]; adjoint[rhs] -= a * tape[lhs] / (tape[rhs] * tape[rhs]); }
                        }
                        for (uint32_t i = 0; i < tree.CoefficientCount; ++i) { gradient[i] += weight * derivative[i] * residual; for (uint32_t j = 0; j < tree.CoefficientCount; ++j) { normal[i * MaxParameters + j] += weight * derivative[i] * derivative[j]; } }
                    }
                }
                return valid && sycl::isfinite(cost) ? cost : NaN;
            };
            auto currentCost = costAt(current, nullptr, nullptr); auto damping = 1e-3F; uint32_t acceptedSteps{}; uint32_t used{};
            for (; used < args.MaxIterations && sycl::isfinite(currentCost); ++used) {
                float matrix[MaxParameters * MaxParameters]{}; float gradient[MaxParameters]{}; float step[MaxParameters]{};
                (void)costAt(current, matrix, gradient);
                for (uint32_t pivot = 0; pivot < tree.CoefficientCount; ++pivot) { matrix[pivot * MaxParameters + pivot] += damping; auto best = pivot; for (uint32_t row = pivot + 1; row < tree.CoefficientCount; ++row) { if (sycl::fabs(matrix[row * MaxParameters + pivot]) > sycl::fabs(matrix[best * MaxParameters + pivot])) { best = row; } } if (sycl::fabs(matrix[best * MaxParameters + pivot]) < 1e-10F) { currentCost = NaN; break; } for (uint32_t column = pivot; column < tree.CoefficientCount; ++column) { auto tmp = matrix[pivot * MaxParameters + column]; matrix[pivot * MaxParameters + column] = matrix[best * MaxParameters + column]; matrix[best * MaxParameters + column] = tmp; } auto tmp = gradient[pivot]; gradient[pivot] = gradient[best]; gradient[best] = tmp; for (uint32_t row = pivot + 1; row < tree.CoefficientCount; ++row) { auto const factor = matrix[row * MaxParameters + pivot] / matrix[pivot * MaxParameters + pivot]; for (uint32_t column = pivot; column < tree.CoefficientCount; ++column) { matrix[row * MaxParameters + column] -= factor * matrix[pivot * MaxParameters + column]; } gradient[row] -= factor * gradient[pivot]; } }
                for (int row = static_cast<int>(tree.CoefficientCount) - 1; row >= 0; --row) { auto value = -gradient[row]; for (uint32_t col = static_cast<uint32_t>(row) + 1; col < tree.CoefficientCount; ++col) { value -= matrix[static_cast<uint32_t>(row) * MaxParameters + col] * step[col]; } step[row] = value / matrix[static_cast<uint32_t>(row) * MaxParameters + static_cast<uint32_t>(row)]; }
                float trial[MaxParameters]{}; for (uint32_t i = 0; i < tree.CoefficientCount; ++i) { trial[i] = current[i] + step[i]; }
                auto const trialCost = costAt(trial, nullptr, nullptr); if (sycl::isfinite(trialCost) && trialCost < currentCost) { for (uint32_t i = 0; i < tree.CoefficientCount; ++i) { current[i] = trial[i]; } currentCost = trialCost; damping = sycl::fmax(damping * 0.3F, 1e-8F); ++acceptedSteps; } else { damping = sycl::fmin(damping * 10.0F, 1e8F); }
            }
            for (uint32_t i = 0; i < tree.CoefficientCount; ++i) { args.Coefficients[tree.CoefficientOffset + i] = current[i]; }
            args.Status[index] = static_cast<uint8_t>(sycl::isfinite(currentCost)); args.Iterations[index] = used; args.Accepted[index] = acceptedSteps;
        });
    });
}
} // namespace Operon::PopulationOptimization::Sycl::detail
