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

#include "operators/creator/balanced.hpp"

namespace Operon {
Tree BalancedTreeCreator::operator()(Operon::RandomGenerator& random, size_t targetLen, size_t, size_t) const
{
    const auto& pset = pset_.get();
    auto [minFunctionArity, maxFunctionArity] = pset.FunctionArityLimits();

    std::normal_distribution<Operon::Scalar> normalReal(0, 1);
    auto init = [&](Node& node) {
        if (node.IsLeaf()) {
            if (node.IsVariable()) {
                node.HashValue = Operon::Random::Sample(random, variables_.begin(), variables_.end())->Hash;
                node.CalculatedHashValue = node.HashValue;
            }
            node.Value = normalReal(random);
        }
    };

    // length one can be achieved with a single leaf
    // otherwise the minimum achievable length is minFunctionArity+1
    if (targetLen > 1 && targetLen < minFunctionArity + 1)
        targetLen = minFunctionArity + 1;

    using U = std::tuple<Node, size_t, size_t>;

    std::vector<U> tuples;
    tuples.reserve(targetLen);

    auto maxArity = std::min(maxFunctionArity, targetLen - 1);
    auto minArity = std::min(minFunctionArity, maxArity); // -1 because we start with a root

    auto root = pset.SampleRandomSymbol(random, minArity, maxArity);
    init(root);

    if (root.IsLeaf()) {
        return Tree({ root }).UpdateNodes();
    }

    tuples.emplace_back(root, 1, 1);

    size_t openSlots = root.Arity;

    std::bernoulli_distribution sampleIrregular(irregularityBias);

    for (size_t i = 0; i < tuples.size(); ++i) {
        auto [node, nodeDepth, childIndex] = tuples[i];
        auto childDepth = nodeDepth + 1;
        std::get<2>(tuples[i]) = tuples.size();
        for (int j = 0; j < node.Arity; ++j) {
            maxArity = openSlots - tuples.size() > 1 && sampleIrregular(random)
                ? 0
                : std::min(maxFunctionArity, targetLen - openSlots - 1);

            // fall back to a leaf node if the desired arity is not achievable with the current primitive set
            if (maxArity < minFunctionArity) {
                minArity = maxArity = 0;
            }

            auto child = pset.SampleRandomSymbol(random, minArity, maxArity);
            init(child);
            tuples.emplace_back(child, childDepth, 0);
            openSlots += child.Arity;
        }
    }

    Operon::Vector<Node> postfix(tuples.size());
    auto idx = tuples.size();

    auto add = [&](const U& t, auto&& ref) {
        auto [node, _, nodeChildIndex] = t;
        postfix[--idx] = node;
        if (node.IsLeaf()) {
            return;
        }
        for (size_t i = nodeChildIndex; i < nodeChildIndex + node.Arity; ++i) {
            ref(tuples[i], ref);
        }
    };
    add(tuples.front(), add);
    auto tree = Tree(postfix).UpdateNodes();
    return tree;
}
}
