// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INTERVAL_EVALUATOR_HPP
#define OPERON_INTERVAL_EVALUATOR_HPP

#include <fmt/core.h>
#include <gsl/pointers>
#include <stdexcept>
#include <utility>
#include <vector>

#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

// pappus headers unconditionally redefine EXPECT/ENSURE (operon defines them
// as ASSERT wrappers) and rely on a deprecated implicit copy ctor for
// `interval<T>` (user-provided copy assignment without a copy ctor). Silence
// both for the pappus includes only; the rest of the TU keeps operon warnings.
#if defined(__clang__)
#  pragma clang diagnostic push
#  pragma clang diagnostic ignored "-Wmacro-redefined"
#  pragma clang diagnostic ignored "-Wdeprecated-copy-with-user-provided-copy"
#elif defined(__GNUC__)
#  pragma GCC diagnostic push
#  pragma GCC diagnostic ignored "-Wmacro-redefined"
#  pragma GCC diagnostic ignored "-Wdeprecated-copy-with-user-provided-copy"
#endif
#include <pappus.hpp>
#if defined(__clang__)
#  pragma clang diagnostic pop
#elif defined(__GNUC__)
#  pragma GCC diagnostic pop
#endif

namespace Operon {

// Forward rigorous bounds for an Operon tree over a single input domain.
//
// Walks the tree in post-order (the same order used by the Operon interpreter)
// and computes a `pappus::interval<Operon::Scalar>` enclosure for each node.
// Variables are bound to a user-supplied domain map keyed by `Node::HashValue`;
// constants and node weights come from the coefficient span (mirroring
// `Tree::GetCoefficients()` / `Node::Optimize`).
//
// Batch size is 1: one evaluation produces one interval enclosure for the whole
// domain. This is intentional -- interval/affine arithmetic are single-value
// computations, not per-row fitness evaluations.
class IntervalEvaluator {
public:
    using Scalar = Operon::Scalar;
    using Interval = pappus::interval<Scalar>;
    // (lower, upper) bound for a variable identified by its hash.
    using Domain = std::pair<Scalar, Scalar>;
    using DomainMap = Operon::Map<Operon::Hash, Domain>;

    IntervalEvaluator(gsl::not_null<Operon::Tree const*> tree, DomainMap domains)
        : tree_(tree), domains_(std::move(domains)) {}

    [[nodiscard]] auto GetTree() const noexcept -> Operon::Tree const* { return tree_.get(); }
    [[nodiscard]] auto Domains() const noexcept -> DomainMap const& { return domains_; }

    // Evaluate the tree over the supplied domains. `coeff` follows the same
    // convention as `Interpreter::Evaluate`: one entry per node with
    // `Node::Optimize == true`, consumed in node order.
    [[nodiscard]] auto Evaluate(Operon::Span<Scalar const> coeff) const -> Interval
    {
        auto const& nodes = tree_->Nodes();
        auto const n = nodes.size();
        if (n == 0) { throw std::runtime_error("IntervalEvaluator: empty tree"); }

        primal_.resize(n);
        std::size_t ci = 0;

        // Folds over the immediate children of node `i`, reading from `primal_`.
        auto const addFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{0}};
            for (auto j : Tree::Indices(nodes, i)) { acc = pappus::ops::add<Scalar>(acc, primal_[j]); }
            return acc;
        };
        auto const mulFold = [&](std::size_t i) {
            auto acc = Interval{Scalar{1}};
            for (auto j : Tree::Indices(nodes, i)) { acc = pappus::ops::mul<Scalar>(acc, primal_[j]); }
            return acc;
        };
        // `first - (rest[0] + rest[1] + ...)` matching Operon's n-ary Sub.
        auto const subFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{0}}; // overwritten on first child
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = primal_[j]; first = false; }
                else       { acc = pappus::ops::sub<Scalar>(acc, primal_[j]); }
            }
            EXPECT(!first); // arity > 0 — malformed tree otherwise
            return acc;
        };
        // `first / (rest[0] * rest[1] * ...)` matching Operon's n-ary Div.
        auto const divFold = [&](std::size_t i) {
            bool first = true;
            auto acc = Interval{Scalar{1}}; // overwritten on first child
            for (auto j : Tree::Indices(nodes, i)) {
                if (first) { acc = primal_[j]; first = false; }
                else       { acc = pappus::ops::div<Scalar>(acc, primal_[j]); }
            }
            EXPECT(!first); // arity > 0 — malformed tree otherwise
            return acc;
        };

        for (std::size_t i = 0; i < n; ++i) {
            auto const& node = nodes[i];
            // Per-node weight/value: leaves use it as value (Constant) or weight
            // (Variable); non-leaves apply it as a post-multiply (matches the
            // Operon `EvaluateTree` reference). Non-leaf nodes default to
            // `Optimize == false`, so this is usually 1.0.
            Scalar v;
            if (node.Optimize) {
                EXPECT(ci < coeff.size());
                v = static_cast<Scalar>(coeff[ci++]);
            } else {
                v = static_cast<Scalar>(node.Value);
            }

            switch (node.Type) {
            case NodeType::Constant:
                primal_[i] = pappus::ops::constant<Scalar>(v);
                break;
            case NodeType::Variable: {
                auto it = domains_.find(node.HashValue);
                if (it == domains_.end()) {
                    throw std::runtime_error(fmt::format(
                        "IntervalEvaluator: no domain bound for variable hash {}",
                        node.HashValue));
                }
                auto const& [lo, hi] = it->second;
                primal_[i] = pappus::ops::variable<Scalar>(lo, hi) * v;
                break;
            }
            case NodeType::Ref:
                EXPECT(static_cast<std::size_t>(node.RefTo) < i);
                primal_[i] = primal_[node.RefTo];
                break;
            case NodeType::Add:
                primal_[i] = addFold(i) * v;
                break;
            case NodeType::Mul:
                primal_[i] = mulFold(i) * v;
                break;
            case NodeType::Sub:
                primal_[i] = (node.Arity == 1 ? pappus::ops::neg<Scalar>(primal_[i - 1])
                                              : subFold(i)) * v;
                break;
            case NodeType::Div:
                primal_[i] = (node.Arity == 1 ? pappus::ops::inv<Scalar>(primal_[i - 1])
                                              : divFold(i)) * v;
                break;
            case NodeType::Square:
                primal_[i] = pappus::ops::square<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Sqrt:
                primal_[i] = pappus::ops::sqrt<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Exp:
                primal_[i] = pappus::ops::exp<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Log:
                primal_[i] = pappus::ops::log<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Sin:
                primal_[i] = pappus::ops::sin<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Cos:
                primal_[i] = pappus::ops::cos<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Tan:
                primal_[i] = pappus::ops::tan<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Asin:
                primal_[i] = pappus::ops::asin<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Acos:
                primal_[i] = pappus::ops::acos<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Atan:
                primal_[i] = pappus::ops::atan<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Sinh:
                primal_[i] = pappus::ops::sinh<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Cosh:
                primal_[i] = pappus::ops::cosh<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Tanh:
                primal_[i] = pappus::ops::tanh<Scalar>(primal_[i - 1]) * v;
                break;
            case NodeType::Pow: {
                auto const j = static_cast<std::size_t>(i - 1);
                auto const k = j - (nodes[j].Length + 1);
                primal_[i] = pappus::ops::pow<Scalar>(primal_[j], primal_[k]) * v;
                break;
            }
            default:
                throw std::runtime_error(fmt::format(
                    "IntervalEvaluator: node kind `{}` not yet mapped (Phase 6)",
                    node.Name()));
            }
        }
        return primal_.back();
    }

private:
    gsl::not_null<Operon::Tree const*> tree_;
    DomainMap domains_;
    mutable std::vector<Interval> primal_; // reused across Evaluate calls
};

} // namespace Operon

#endif
