// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cstddef>
#include <cstdint>
#include <vector>
#include <deque>
#include <algorithm>
#include <random>
#include <span>
#include <utility>

#include "operon/operators/creator.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"
#include "operon/random/random.hpp"

namespace Operon {
auto ProbabilisticTreeCreator::operator()(Operon::RandomGenerator& random, size_t targetLen, size_t /*args*/, size_t /*args*/) const -> Tree
{
    EXPECT(targetLen > 0);
    auto const& variables = GetVariables();

    auto init = [&](Node& node) {
        if (node.IsLeaf()) {
            if (node.IsVariable()) {
                node.HashValue = *Random::Sample(random, variables.begin(), variables.end());
                node.CalculatedHashValue = node.HashValue;
            }
            node.Value = 1;
        }
    };

    const auto& pset = GetPrimitiveSet();
    auto [minFunctionArity, maxFunctionArity] = pset.FunctionArityLimits();

    // length one can be achieved with a single leaf
    // otherwise the minimum achievable length is minFunctionArity+1
    if (targetLen > 1 && targetLen < minFunctionArity + 1) {
        targetLen = minFunctionArity + 1;
    }

    Operon::Vector<Node> nodes;
    nodes.reserve(targetLen);

    auto maxArity = std::min(maxFunctionArity, targetLen - 1);
    auto minArity = std::min(minFunctionArity, maxArity);

    auto root = pset.SampleRandomSymbol(random, minArity, maxArity);
    init(root);

    if (root.IsLeaf()) {
        return Tree({ root }).UpdateNodes();
    }

    root.Depth = 1;
    nodes.push_back(root);

    std::deque<size_t> q;
    for (size_t i = 0; i < root.Arity; ++i) {
        auto d = root.Depth + 1U;
        q.push_back(d);
    }

    // emulate a random dequeue operation
    auto randomDequeue = [&]() {
        EXPECT(!q.empty());
        auto j = std::uniform_int_distribution<size_t>(0, q.size() - 1)(random);
        std::swap(q[j], q.front());
        auto t = q.front();
        q.pop_front();
        return t;
    };

    root.Parent = 0;

    std::bernoulli_distribution sampleIrregular(irregularityBias_);

    while (!q.empty()) {
        auto childDepth = randomDequeue();

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
        node.Depth = static_cast<uint16_t>(childDepth);

        for (size_t i = 0; i < node.Arity; ++i) {
            q.push_back(childDepth + 1);
        }

        nodes.push_back(node);
    }

    std::sort(nodes.begin(), nodes.end(), [](const auto& lhs, const auto& rhs) { return lhs.Depth < rhs.Depth; });
    std::vector<size_t> childIndices(nodes.size());

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

    const auto add = [&](size_t i, auto&& ref) {
        const auto& node = nodes[i];

        postfix[--idx] = node;

        if (node.IsLeaf()) {
            return;
        }

        for (size_t j = 0; j < node.Arity; ++j) {
            ref(childIndices[i] + j, ref);
        }
    };

    add(0, add);

    auto tree = Tree(postfix).UpdateNodes();
    return tree;
}
} // namespace Operon
