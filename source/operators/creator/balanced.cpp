// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cstddef>
#include <algorithm>
#include <random>
#include <span>
#include <tuple>
#include <vector>

#include "operon/operators/creator.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"
#include "operon/random/random.hpp"

namespace Operon {
auto BalancedTreeCreator::operator()(Operon::RandomGenerator& random, size_t targetLen, size_t /*args*/, size_t /*args*/) const -> Tree
{
    auto const& pset = GetPrimitiveSet();
    auto [minFunctionArity, maxFunctionArity] = pset.FunctionArityLimits();

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

    // length one can be achieved with a single leaf
    // otherwise the minimum achievable length is minFunctionArity+1
    if (targetLen > 1 && targetLen < minFunctionArity + 1) {
        targetLen = minFunctionArity + 1;
    }

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

    std::bernoulli_distribution sampleIrregular(irregularityBias_);

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
} // namespace Operon
