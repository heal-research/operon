/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2020 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef INITIALIZER_HPP
#define INITIALIZER_HPP

#include "core/operator.hpp"

namespace Operon {
// wraps a creator and generates trees from a given size distribution
template <typename TCreator, typename TDistribution>
struct Initializer : public OperatorBase<Tree> {
public:
    Initializer(const TCreator& creator, TDistribution& dist)
        : creator_(creator)
        , dist_(dist)
        , maxDepth_(1000)
    {
    }

    Tree operator()(Operon::Random& random) const override
    {
        auto targetLen = std::max(1ul, static_cast<size_t>(std::round(dist_(random))));
        return creator_(random, targetLen, maxDepth_);
    }

    void MaxDepth(size_t maxDepth) { maxDepth_ = maxDepth; }
    size_t MaxDepth() const { return maxDepth_; }

    const TCreator& GetCreator() const { return creator_; }

private:
    std::reference_wrapper<const TCreator> creator_;
    mutable TDistribution dist_;
    size_t maxDepth_;
};
}
#endif
