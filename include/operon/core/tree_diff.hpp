// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#pragma once

#include <cstddef>
#include <functional>

#include "contracts.hpp"
#include "node.hpp"
#include "tree.hpp"
#include "types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

// A symbolic derivative rule for a unary built-in or user-defined function:
// given the dag under construction (`dag`/`memo`/`h`, threaded the same way
// Deriv()'s own helpers in tree_diff.cpp thread them, for hash-consing),
// this node's own dag index `i`, and its argument's dag index `j`, returns
// the dag index of f'(j) — or Operon::Vector<Node>::size_type(-1)
// (tree_diff.cpp's `Zero` sentinel) if no derivative is computable. Several
// rules need `i` (e.g. Exp: f'(j) = exp(j) = the node's own result, at `i`),
// not just `j`, hence both are threaded through.
using UnarySymbolicDerivRule = std::function<std::size_t(
    Operon::Vector<Node>& dag,
    Operon::Map<Operon::Hash, std::size_t>& memo,
    Operon::Vector<Operon::Hash>& h,
    std::size_t i,
    std::size_t j)>;

// Register a symbolic derivative rule for a unary function (built-in or
// user-defined), keyed by the same hash the function's Node::HashValue
// carries. Looked up by BuildJacobianDag/BuildHessianDag's internal Deriv()
// for every unary node; a miss degrades to "not differentiable" (Zero),
// identical to today's behavior for an unregistered op. Throws if `hash` is
// already registered (write-once, see HashRegistry::Register).
OPERON_EXPORT void RegisterUnarySymbolicDeriv(Operon::Hash hash, UnarySymbolicDerivRule rule);

// Query whether a symbolic derivative rule is registered for `hash` (built-in
// or user-defined). Mainly useful for coverage checks (e.g. asserting every
// BuiltinOp is either registered here or deliberately excluded as a
// structural/non-smooth case handled directly in Deriv()).
OPERON_EXPORT auto HasUnarySymbolicDeriv(Operon::Hash hash) -> bool;

// A flat postfix array containing the original tree (indices 0..OriginalSize-1)
// plus appended symbolic derivative subtrees. Shared subexpressions between
// derivative columns are referenced via NodeType::Ref back-pointers.
struct JacobianDag {
    Operon::Vector<Node> Nodes;        // original nodes [0..OriginalSize-1] + derivative nodes
    std::size_t OriginalSize{};        // number of nodes in the source tree
    Operon::Vector<std::size_t> Roots; // Roots[k] = dag index of df/dc_k; SIZE_MAX means zero
};

// Build a DAG containing the original tree plus symbolic partial derivatives
// w.r.t. each optimizable coefficient (Node::Optimize == true). Coefficients
// include both Constant nodes and Variable nodes whose weight is being tuned.
// Common subexpressions across derivative columns are deduplicated via
// hash-consing; shared nodes are referenced with NodeType::Ref. The tree does
// not need to have been hashed before calling this.
OPERON_EXPORT auto BuildJacobianDag(Tree const& tree) -> JacobianDag;

// A flat postfix array containing the original tree, first-order derivative
// subtrees (Jacobian), and second-order derivative subtrees (Hessian).
// The Hessian is symmetric; only the upper triangle is stored row-major:
//   H(i,j) with j >= i at index  i*p - i*(i-1)/2 + (j-i)  where p = NumParams.
struct HessianDag {
    Operon::Vector<Node> Nodes;
    std::size_t OriginalSize{};
    std::size_t NumParams{};
    Operon::Vector<std::size_t> JacobianRoots; // [p] roots of df/dc_k
    Operon::Vector<std::size_t> HessianRoots;  // [p*(p+1)/2] upper triangle

    [[nodiscard]] auto UpperIdx(std::size_t i, std::size_t j) const -> std::size_t {
        EXPECT(i <= j);
        return (i * NumParams) - (i * (i - 1) / 2) + (j - i);
    }
};

OPERON_EXPORT auto BuildHessianDag(Tree const& tree) -> HessianDag;

} // namespace Operon
