// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#pragma once

#include "contracts.hpp"
#include "node.hpp"
#include "tree.hpp"
#include "types.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

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
