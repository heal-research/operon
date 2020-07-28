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

#ifndef BALANCED_TREE_CREATOR_HPP
#define BALANCED_TREE_CREATOR_HPP

#include "core/grammar.hpp"
#include "core/operator.hpp"

namespace Operon {

// this tree creator expands bread-wise using a "horizon" of open expansion slots
// at the end the breadth sequence of nodes is converted to a postfix sequence
// if the depth is not limiting, the target length is guaranteed to be reached
class BalancedTreeCreator final : public CreatorBase {
public:
    using U = std::tuple<Node, size_t, size_t>;

    BalancedTreeCreator(const Grammar& grammar, const gsl::span<const Variable> variables, double bias = 0.0) 
        : CreatorBase(grammar, variables)
        , irregularityBias(bias)
    {
    }
    Tree operator()(Operon::Random& random, size_t targetLen, size_t minDepth, size_t maxDepth) const override; 

    void SetBias(double bias) { irregularityBias = bias; }
    double GetBias() const { return irregularityBias; }

private:
    double irregularityBias;
};
}
#endif
