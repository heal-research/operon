// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_INFORMATION_CRITERIA_WEIGHTED_COMPLEXITY_HPP
#define OPERON_INFORMATION_CRITERIA_WEIGHTED_COMPLEXITY_HPP

#include <cmath>
#include <utility>

#include "operon/core/node.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/types.hpp"

namespace Operon {

// Weighted node/symbol count used by MinimumDescriptionLength and
// FractionalBayesFactor (Bartlett et al. 2023, arXiv:2304.06333, Sec 2.2/2.3;
// matches Julia's func_compl / aifeyn_complexity). A weighted variable node
// (Value != 1) expands to (weight * variable), contributing three logical
// symbols: Mul, Constant, Variable; everything else counts as one node/symbol.
// Returns {k, fComplexity} where k is the weighted node count and
// fComplexity = k * log(q), q = number of unique symbol types.
inline auto WeightedComplexity(Tree const& tree) -> std::pair<double, double>
{
    static auto const MulHash   = static_cast<Operon::Hash>(BuiltinOp::Mul);
    static auto const ParamHash = Node{NodeType::Constant}.HashValue;
    Operon::Set<Operon::Hash> uniqueSymbols;
    auto k = 0.0;
    for (auto const& node : tree.Nodes()) {
        auto const isWeighted = node.IsVariable() && node.Value != Operon::Scalar{1};
        k += isWeighted ? 3.0 : 1.0; // NOLINT(cppcoreguidelines-avoid-magic-numbers)
        uniqueSymbols.insert(node.HashValue);
        if (isWeighted) {
            uniqueSymbols.insert(MulHash);
            uniqueSymbols.insert(ParamHash);
        }
    }
    auto const q           = static_cast<double>(uniqueSymbols.size());
    auto const fComplexity = q > 0.0 ? k * std::log(q) : 0.0;
    return {k, fComplexity};
}

} // namespace Operon

#endif
