// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef BALANCED_TREE_CREATOR_HPP
#define BALANCED_TREE_CREATOR_HPP

#include "core/pset.hpp"
#include "core/operator.hpp"

namespace Operon {

// this tree creator expands bread-wise using a "horizon" of open expansion slots
// at the end the breadth sequence of nodes is converted to a postfix sequence
// if the depth is not limiting, the target length is guaranteed to be reached
class BalancedTreeCreator final : public CreatorBase {
public:
    BalancedTreeCreator(PrimitiveSet const& pset, const gsl::span<const Variable> variables, double bias = 0.0) 
        : CreatorBase(pset, variables)
        , irregularityBias(bias)
    { }
    Tree operator()(Operon::RandomGenerator& random, size_t targetLen, size_t minDepth, size_t maxDepth) const override;

    void SetBias(double bias) { irregularityBias = bias; }
    double GetBias() const { return irregularityBias; }

private:
    double irregularityBias;
};
}
#endif
