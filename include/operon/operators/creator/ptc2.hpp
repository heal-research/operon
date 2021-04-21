// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef PROBABILISTIC_TREE_CREATOR_HPP
#define PROBABILISTIC_TREE_CREATOR_HPP

#include "core/pset.hpp"
#include "core/operator.hpp"

namespace Operon {

class ProbabilisticTreeCreator final : public CreatorBase {
public:
    ProbabilisticTreeCreator(const PrimitiveSet& pset, const gsl::span<const Variable> variables, double bias = 0.0)
        : CreatorBase(pset, variables)
        , irregularityBias(bias)
    {
    }
    Tree operator()(Operon::RandomGenerator& random, size_t targetLen, size_t minDepth, size_t maxDepth) const override;

private:
    double irregularityBias;
};
}

#endif
