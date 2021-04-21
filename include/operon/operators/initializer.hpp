// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include "core/operator.hpp"

namespace Operon {
// wraps a creator and generates trees from a given size distribution
template <typename TDistribution>
struct Initializer : public OperatorBase<Tree> {
public:
    Initializer(const CreatorBase& creator, TDistribution& dist)
        : creator_(creator)
        , dist_(dist)
        , minDepth_(1)
        , maxDepth_(1000)
    {
    }

    Tree operator()(Operon::RandomGenerator& random) const override
    {
        auto targetLen = std::max(size_t{1}, static_cast<size_t>(std::round(dist_(random))));
        return creator_(random, targetLen, minDepth_, maxDepth_);
    }

    void MinDepth(size_t minDepth) { minDepth_ = minDepth; }
    size_t MinDepth() const { return minDepth_; }

    void MaxDepth(size_t maxDepth) { maxDepth_ = maxDepth; }
    size_t MaxDepth() const { return maxDepth_; }

    const CreatorBase& GetCreator() const { return creator_; }

private:
    std::reference_wrapper<const CreatorBase> creator_;
    mutable TDistribution dist_;
    size_t minDepth_;
    size_t maxDepth_;
};
}
#endif
