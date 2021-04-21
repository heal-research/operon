// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operators/creator/koza.hpp"

namespace Operon {
Tree GrowTreeCreator::operator()(Operon::RandomGenerator& random, size_t, size_t minDepth, size_t maxDepth) const
{
    minDepth = std::max(size_t{1}, minDepth);
    EXPECT(minDepth <= maxDepth);
    auto const& pset = pset_.get();

    auto const& t = pset.FunctionArityLimits();

    size_t minFunctionArity = std::get<0>(t);
    size_t maxFunctionArity = std::get<1>(t);

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

    Operon::Vector<Node> nodes;
    size_t minArity = minFunctionArity;
    size_t maxArity = maxFunctionArity;

    std::uniform_int_distribution<size_t> dist(minDepth, maxDepth);
    auto actualDepthLimit = dist(random);

    auto const grow = [&](size_t depth, auto&& ref) {
        auto minDepthReached = depth >= minDepth;

        minArity = 0;
        maxArity = 0;

        if (depth < actualDepthLimit) {
            minArity = minDepthReached ? 0 : minFunctionArity;
            maxArity = maxFunctionArity;
        }

        auto node = pset.SampleRandomSymbol(random, minArity, maxArity);
        init(node);

        nodes.push_back(node);

        if (node.IsLeaf())
            return;

        for (size_t i = 0; i < node.Arity; ++i) {
            ref(depth + 1, ref);
        }
    };

    grow(1, grow);

    std::reverse(nodes.begin(), nodes.end());
    return Tree(nodes).UpdateNodes();
}
}
