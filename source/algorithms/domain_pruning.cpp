// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include "operon/algorithms/domain_pruning.hpp"

#include <cmath>
#include <functional>
#include <limits>
#include <optional>
#include <vector>

namespace Operon {
namespace {

using RowStatus = DomainStatus;

[[nodiscard]] auto Combine(DomainPolicy policy, std::vector<RowStatus> const& rows) -> DomainStatus
{
    if (rows.empty()) {
        return DomainStatus::Unknown;
    }
    bool anyValid = false;
    bool anyInvalid = false;
    bool anyUnknown = false;
    for (auto status : rows) {
        anyValid |= status == DomainStatus::Valid;
        anyInvalid |= status == DomainStatus::Invalid;
        anyUnknown |= status == DomainStatus::Unknown;
    }
    if (policy == DomainPolicy::NoFiniteRows) {
        if (anyValid) return DomainStatus::Valid;
        if (anyUnknown) return DomainStatus::Unknown;
        return DomainStatus::Invalid;
    }
    if (anyUnknown) return DomainStatus::Unknown;
    if (anyInvalid) return DomainStatus::Invalid;
    return DomainStatus::Valid;
}

[[nodiscard]] auto Finite(double value) -> RowStatus
{
    return std::isfinite(value) ? DomainStatus::Valid : DomainStatus::Invalid;
}

} // namespace

auto AnalyzeDomain(Tree const& tree, DomainContext const& context, DomainPolicy policy) -> DomainStatus
{
    if (context.Data == nullptr || context.Rows.Size() == 0 || tree.Empty()) {
        return DomainStatus::Unknown;
    }

    auto const rows = context.Rows;
    auto const& nodes = tree.Nodes();
    std::vector<RowStatus> statuses;
    statuses.reserve(rows.Size());

    for (std::size_t row = rows.Start(); row < rows.End(); ++row) {
        std::function<RowStatus(std::size_t)> eval = [&](std::size_t index) -> RowStatus {
            auto const& node = nodes[index];
            if (node.Optimize) {
                return DomainStatus::Unknown;
            }
            if (node.IsConstant()) {
                return Finite(static_cast<double>(node.Value));
            }
            if (node.IsVariable()) {
                auto values = context.Data->GetValues(node.HashValue);
                if (values.empty() || row >= values.size()) return DomainStatus::Unknown;
                return Finite(static_cast<double>(values[row]));
            }
            if (node.IsRef()) {
                if (node.RefTo >= index) return DomainStatus::Unknown;
                return eval(node.RefTo);
            }
            if (!node.IsFunction()) return DomainStatus::Unknown;

            auto const op = static_cast<BuiltinOp>(node.HashValue);
            bool const restricted = op == BuiltinOp::Log || op == BuiltinOp::Logabs || op == BuiltinOp::Sqrt
                || op == BuiltinOp::Sqrtabs || op == BuiltinOp::Div || op == BuiltinOp::Pow;
            if (!restricted) {
                std::vector<RowStatus> children;
                children.reserve(node.Arity);
                auto child = index - 1;
                for (uint16_t i = 0; i < node.Arity; ++i) {
                    if (i != 0) child -= nodes[child].Length + 1;
                    children.push_back(eval(child));
                }
                bool unknownChild = false;
                for (auto status : children) {
                    unknownChild |= status == DomainStatus::Unknown;
                    if (status == DomainStatus::Invalid && unknownChild) return DomainStatus::Unknown;
                }
                if (unknownChild) return DomainStatus::Unknown;
                for (auto status : children) {
                    if (status == DomainStatus::Invalid) return DomainStatus::Invalid;
                }
                return DomainStatus::Unknown;
            }

            std::function<std::optional<double>(std::size_t)> fixed = [&](std::size_t idx) -> std::optional<double> {
                auto const& n = nodes[idx];
                if (n.Optimize) return std::nullopt;
                if (n.IsConstant()) return static_cast<double>(n.Value);
                if (n.IsRef()) return n.RefTo < idx ? fixed(n.RefTo) : std::nullopt;
                if (n.IsVariable()) {
                    auto values = context.Data->GetValues(n.HashValue);
                    if (values.empty() || row >= values.size()) return std::nullopt;
                    return static_cast<double>(values[row]);
                }
                if (!n.IsFunction()) return std::nullopt;
                auto childRoot = idx - 1;
                std::vector<std::size_t> roots;
                roots.reserve(n.Arity);
                for (uint16_t i = 0; i < n.Arity; ++i) {
                    if (i != 0) childRoot -= nodes[childRoot].Length + 1;
                    roots.push_back(childRoot);
                }
                std::vector<double> args;
                args.reserve(roots.size());
                for (auto childIndex : roots) {
                    auto value = fixed(childIndex);
                    if (!value) return std::nullopt;
                    args.push_back(*value);
                }
                auto const nOp = static_cast<BuiltinOp>(n.HashValue);
                switch (nOp) {
                case BuiltinOp::Add: { double result = 0; for (auto value : args) result += value; return result; }
                case BuiltinOp::Mul: { double result = 1; for (auto value : args) result *= value; return result; }
                case BuiltinOp::Sub: return args.size() == 2 ? std::optional<double>{args[0] - args[1]} : std::nullopt;
                case BuiltinOp::Div:
                    if (args.size() == 1) return args[0] == 0 ? std::nullopt : std::optional<double>{1.0 / args[0]};
                    return args.size() == 2 && args[1] != 0 ? std::optional<double>{args[0] / args[1]} : std::nullopt;
                case BuiltinOp::Pow: return args.size() == 2 ? std::optional<double>{std::pow(args[0], args[1])} : std::nullopt;
                case BuiltinOp::Log: return args.size() == 1 && args[0] > 0 ? std::optional<double>{std::log(args[0])} : std::nullopt;
                case BuiltinOp::Logabs: return args.size() == 1 && args[0] != 0 ? std::optional<double>{std::log(std::abs(args[0]))} : std::nullopt;
                case BuiltinOp::Sqrt: return args.size() == 1 && args[0] >= 0 ? std::optional<double>{std::sqrt(args[0])} : std::nullopt;
                case BuiltinOp::Sqrtabs: return args.size() == 1 ? std::optional<double>{std::sqrt(std::abs(args[0]))} : std::nullopt;
                default: return std::nullopt;
                }
            };

            bool const unary = op == BuiltinOp::Log || op == BuiltinOp::Logabs || op == BuiltinOp::Sqrt
                || op == BuiltinOp::Sqrtabs;
            if ((unary && node.Arity != 1) || (op == BuiltinOp::Div && (node.Arity < 1 || node.Arity > 2))
                || (op == BuiltinOp::Pow && node.Arity != 2)) {
                return DomainStatus::Unknown;
            }
            auto childRoot = index - 1;
            std::vector<std::size_t> roots;
            roots.reserve(node.Arity);
            for (uint16_t i = 0; i < node.Arity; ++i) {
                if (i != 0) childRoot -= nodes[childRoot].Length + 1;
                roots.push_back(childRoot);
            }
            auto first = fixed(roots[0]);
            if (!first) return DomainStatus::Unknown;
            auto const a = *first;
            if (op == BuiltinOp::Log) return a > 0 ? Finite(std::log(a)) : DomainStatus::Invalid;
            if (op == BuiltinOp::Logabs) return a != 0 ? Finite(std::log(std::abs(a))) : DomainStatus::Invalid;
            if (op == BuiltinOp::Sqrt) return a >= 0 ? Finite(std::sqrt(a)) : DomainStatus::Invalid;
            if (op == BuiltinOp::Sqrtabs) return Finite(std::sqrt(std::abs(a)));
            if (op == BuiltinOp::Div) {
                if (node.Arity == 1) return a != 0 ? Finite(1.0 / a) : DomainStatus::Invalid;
                if (node.Arity != 2) return DomainStatus::Unknown;
                auto second = fixed(roots[1]);
                if (!second) return DomainStatus::Unknown;
                return *second != 0 ? Finite(a / *second) : DomainStatus::Invalid;
            }
            if (node.Arity != 2) return DomainStatus::Unknown;
            auto second = fixed(roots[1]);
            if (!second) return DomainStatus::Unknown;
            auto const exponent = *second;
            if (a == 0 && exponent < 0) return DomainStatus::Invalid;
            if (a < 0 && std::floor(exponent) != exponent) return DomainStatus::Invalid;
            return Finite(std::pow(a, exponent));
        };
        statuses.push_back(eval(nodes.size() - 1));
    }
    return Combine(policy, statuses);
}
} // namespace Operon

