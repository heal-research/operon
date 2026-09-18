// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2026-present Bogdan Burlacu and contributors

#ifndef OPERON_POSTORDER_EVALUATOR_HPP
#define OPERON_POSTORDER_EVALUATOR_HPP

#include <optional>
#include <string>
#include <utility>

#include <tl/expected.hpp>

#include "operon/core/contracts.hpp"
#include "operon/core/tree.hpp"

namespace Operon::detail {

// Policy-driven structural executor for postfix trees. Policies bind leaves,
// weights, primitive registries, and value algebra; this owns the common
// subtree traversal, n-ary reduction, and builtin dispatch.
template<typename Policy, typename Nodes, typename Primal, typename Weight, typename BindLeaf>
auto EvaluatePostOrder(Nodes const& nodes, Primal& primal, typename Policy::Context const& context,
                       Weight&& weight, BindLeaf&& bindLeaf)
    -> tl::expected<typename Policy::Value, std::string>
{
    using Value = typename Policy::Value;

    Policy::RegisterBuiltins();
    if (nodes.empty()) {
        return tl::unexpected(Policy::EmptyTree());
    }

    primal.clear();
    primal.reserve(nodes.size());

    auto const fold = [&](std::size_t index, std::optional<Value> initial, auto operation) -> Value {
        auto accumulator = std::move(initial);
        for (auto child : Tree::Indices(nodes, index)) {
            if (accumulator) {
                *accumulator = operation(*accumulator, primal[child]);
            } else {
                accumulator = primal[child];
            }
        }
        EXPECT(accumulator.has_value());
        return std::move(*accumulator);
    };

    for (std::size_t index = 0; index < nodes.size(); ++index) {
        auto const& node = nodes[index];
        if (node.IsRef()) {
            EXPECT(static_cast<std::size_t>(node.RefTo) < primal.size());
            primal.push_back(primal[node.RefTo]);
            continue;
        }

        auto const scale = weight(node);
        if (node.IsConstant() || node.IsVariable()) {
            auto value = bindLeaf(node, index, scale);
            if (!value) {
                return tl::unexpected(std::move(value.error()));
            }
            primal.push_back(std::move(*value));
            continue;
        }

        auto const evaluate = [&]() -> tl::expected<Value, std::string> {
            switch (node.HashValue) {
            case Operon::Hash(BuiltinOp::Add):
                return fold(index, Policy::MakeConstant(context, typename Policy::Scalar { 0 }),
                            [&](Value const& lhs, Value const& rhs) { return Policy::Add(context, lhs, rhs); });
            case Operon::Hash(BuiltinOp::Mul):
                return fold(index, Policy::MakeConstant(context, typename Policy::Scalar { 1 }),
                            [&](Value const& lhs, Value const& rhs) { return Policy::Mul(context, lhs, rhs); });
            case Operon::Hash(BuiltinOp::Sub):
                return node.Arity == 1 ? Policy::Neg(context, primal[index - 1])
                                       : fold(index, std::optional<Value> {},
                                              [&](Value const& lhs, Value const& rhs) { return Policy::Sub(context, lhs, rhs); });
            case Operon::Hash(BuiltinOp::Div):
                return node.Arity == 1 ? Policy::Inv(context, primal[index - 1])
                                       : fold(index, std::optional<Value> {},
                                              [&](Value const& lhs, Value const& rhs) { return Policy::Div(context, lhs, rhs); });
            case Operon::Hash(BuiltinOp::Fmin):
                return fold(index, std::optional<Value> {},
                            [&](Value const& lhs, Value const& rhs) { return Policy::Min(context, lhs, rhs); });
            case Operon::Hash(BuiltinOp::Fmax):
                return fold(index, std::optional<Value> {},
                            [&](Value const& lhs, Value const& rhs) { return Policy::Max(context, lhs, rhs); });
            default:
                if (node.Arity == 1) {
                    if (auto const* unary = Policy::UnaryRules().TryGet(node.HashValue)) {
                        return Policy::CallUnary(context, *unary, primal[index - 1]);
                    }
                } else if (node.Arity == 2) {
                    auto const near = index - 1;
                    auto const far = near - (nodes[near].Length + 1);
                    if (auto const* binary = Policy::BinaryRules().TryGet(node.HashValue)) {
                        return Policy::CallBinary(context, *binary, primal[near], primal[far]);
                    }
                }
                return tl::unexpected(Policy::MissingNode(node));
            }
        }();
        if (!evaluate) {
            return tl::unexpected(std::move(evaluate.error()));
        }
        primal.push_back(Policy::Scale(std::move(*evaluate), scale));
    }
    return primal.back();
}

} // namespace Operon::detail

#endif