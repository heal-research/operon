// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_KOZA_CREATOR_HPP
#define OPERON_KOZA_CREATOR_HPP

#include "core/pset.hpp"
#include "core/operator.hpp"

namespace Operon {
class GrowTreeCreator final : public CreatorBase {
    public:
        GrowTreeCreator(const PrimitiveSet& pset, const Operon::Span<const Variable> variables)
            : CreatorBase(pset, variables) 
        { }

    Tree operator()(Operon::RandomGenerator& random, size_t targetLen, size_t minDepth, size_t maxDepth) const override;
};
}

#endif
