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

#include <deque>
#include <queue>
#include <vector>

#include "operators/creator/ptc2.hpp"

namespace Operon {
Tree ProbabilisticTreeCreator::operator()(Operon::RandomGenerator& random, size_t targetLen, size_t, size_t) const
{
    EXPECT(targetLen > 0);

    std::uniform_int_distribution<size_t> uniformInt(0, variables_.size() - 1);
    std::normal_distribution<double> normalReal(0, 1);
    auto init = [&](Node& node) {
        if (node.IsLeaf()) {
            if (node.IsVariable()) {
                node.HashValue = variables_[uniformInt(random)].Hash;
                node.CalculatedHashValue = node.HashValue;
            }
            node.Value = normalReal(random);
        }
    };

    const auto& pset = pset_.get();
    auto [minFunctionArity, maxFunctionArity] = pset.FunctionArityLimits();

    // length one can be achieved with a single leaf
    // otherwise the minimum achievable length is minFunctionArity+1
    if (targetLen > 1 && targetLen < minFunctionArity + 1)
        targetLen = minFunctionArity + 1;

    Operon::Vector<Node> nodes;
    nodes.reserve(targetLen);

    auto maxArity = std::min(maxFunctionArity, targetLen - 1);
    auto minArity = std::min(minFunctionArity, maxArity);

    auto root = pset.SampleRandomSymbol(random, minArity, maxArity);
    init(root);

    if (root.IsLeaf())
        return Tree({ root }).UpdateNodes();

    root.Depth = 1;
    nodes.push_back(root);

    std::deque<size_t> q;
    for (size_t i = 0; i < root.Arity; ++i) {
        auto d = root.Depth + 1;
        q.push_back(d);
    }

    // emulate a random dequeue operation
    auto random_dequeue = [&]() {
        EXPECT(!q.empty());
        auto j = std::uniform_int_distribution<size_t>(0, q.size() - 1)(random);
        std::swap(q[j], q.front());
        auto t = q.front();
        q.pop_front();
        return t;
    };

    root.Parent = 0;

    std::bernoulli_distribution sampleIrregular(irregularityBias);

    while (q.size() > 0) {
        auto childDepth = random_dequeue();

        maxArity = q.size() > 1 && sampleIrregular(random)
            ? 0
            : std::min(maxFunctionArity, targetLen - q.size() - nodes.size() - 1);

        // certain lengths cannot be generated using available symbols
        // in this case we push the target length towards an achievable value
        if (maxArity > 0 && maxArity < minFunctionArity) {
            EXPECT(targetLen > 0);
            EXPECT(targetLen == 1 || targetLen >= minFunctionArity + 1);
            targetLen -= minFunctionArity - maxArity;
            maxArity = std::min(maxFunctionArity, targetLen - q.size() - nodes.size() - 1);
        }
        minArity = std::min(minFunctionArity, maxArity);

        auto node = pset.SampleRandomSymbol(random, minArity, maxArity);

        init(node);
        node.Depth = childDepth;

        for (size_t i = 0; i < node.Arity; ++i) {
            q.push_back(childDepth + 1);
        }

        nodes.push_back(node);
    }

    std::sort(nodes.begin(), nodes.end(), [](const auto& lhs, const auto& rhs) { return lhs.Depth < rhs.Depth; });
    std::vector<int> childIndices(nodes.size());

    size_t c = 1;
    for (size_t i = 0; i < nodes.size(); ++i) {
        auto& node = nodes[i];

        if (node.IsLeaf()) {
            continue;
        }

        childIndices[i] = c;
        c += nodes[i].Arity;
    }

    Operon::Vector<Node> postfix(nodes.size());
    size_t idx = nodes.size();

    const auto add = [&](size_t i, auto&& add) {
        const auto& node = nodes[i];

        postfix[--idx] = node;

        if (node.IsLeaf()) {
            return;
        }

        for (size_t j = 0; j < node.Arity; ++j) {
            add(childIndices[i] + j, add);
        }
    };

    add(0, add);

    auto tree = Tree(postfix).UpdateNodes();
    return tree;
}
}
