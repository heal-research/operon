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

#ifndef CROSSOVER_HPP
#define CROSSOVER_HPP

#include <vector>

#include "core/operator.hpp"

namespace Operon {
class SubtreeCrossover : public CrossoverBase {
public:
    SubtreeCrossover(double p, size_t d, size_t l)
        : internalProbability(p)
        , maxDepth(d)
        , maxLength(l)
    {
    }
    auto operator()(Operon::RandomGenerator& random, const Tree& lhs, const Tree& rhs) const -> Tree override;
    std::pair<size_t, size_t> FindCompatibleSwapLocations(Operon::RandomGenerator& random, const Tree& lhs, const Tree& rhs) const;

    static inline Tree Cross(const Tree& lhs, const Tree& rhs, size_t i, size_t j)
    {
        auto const& left = lhs.Nodes();
        auto const& right = rhs.Nodes();
        Operon::Vector<Node> nodes;
        using signed_t = std::make_signed<size_t>::type;
        nodes.reserve(right[j].Length - left[i].Length + left.size());
        std::copy_n(left.begin(), i - left[i].Length, back_inserter(nodes));
        std::copy_n(right.begin() + static_cast<signed_t>(j) - right[j].Length, right[j].Length + 1, back_inserter(nodes));
        std::copy_n(left.begin() + static_cast<signed_t>(i) + 1, left.size() - (i + 1), back_inserter(nodes));

        auto child = Tree(nodes).UpdateNodes();
        return child;
    }

    double InternalProbability() const { return internalProbability; }
    size_t MaxDepth() const { return maxDepth; }
    size_t MaxLength() const { return maxLength; }

private:
    double internalProbability;
    size_t maxDepth;
    size_t maxLength;
};
}
#endif
