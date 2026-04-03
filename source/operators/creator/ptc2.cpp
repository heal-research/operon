// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <cstddef>
#include <cstdint>
#include <vector>
#include <deque>
#include <algorithm>
#include <random>
#include <utility>

#include "operon/operators/creator.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"
#include "operon/random/random.hpp"

namespace {
auto InitNode(Operon::Node& node, Operon::Span<Operon::Hash const> variables, Operon::RandomGenerator& random) -> void {
    if (node.IsLeaf()) {
        if (node.IsVariable()) {
            node.HashValue = *Operon::Random::Sample(random, variables.begin(), variables.end());
            node.CalculatedHashValue = node.HashValue;
        }
        node.Value = 1;
    }
}

auto RandomDequeue(Operon::RandomGenerator& random, std::deque<size_t>& q) -> size_t {
    EXPECT(!q.empty());
    auto j = std::uniform_int_distribution<size_t>(0, q.size() - 1)(random);
    std::swap(q[j], q.front());
    auto t = q.front();
    q.pop_front();
    return t;
}

auto BuildPostfix(
    Operon::Vector<Operon::Node> const& nodes,
    std::vector<size_t> const& childIndices,
    Operon::Vector<Operon::Node>& postfix,
    size_t& idx,
    size_t i
) -> void {
    auto const& node = nodes[i];
    postfix[--idx] = node;
    if (node.IsLeaf()) { return; }
    for (size_t j = 0; j < node.Arity; ++j) {
        BuildPostfix(nodes, childIndices, postfix, idx, childIndices[i] + j);
    }
}

auto ComputeChildIndices(Operon::Vector<Operon::Node> const& nodes) -> std::vector<size_t> {
    std::vector<size_t> childIndices(nodes.size());
    size_t c = 1;
    for (size_t i = 0; i < nodes.size(); ++i) {
        if (!nodes[i].IsLeaf()) {
            childIndices[i] = c;
            c += nodes[i].Arity;
        }
    }
    return childIndices;
}

auto ProcessNextNode(
    Operon::RandomGenerator& random,
    std::deque<size_t>& q,
    Operon::Vector<Operon::Node>& nodes,
    size_t& targetLen,
    size_t minFunctionArity,
    size_t maxFunctionArity,
    Operon::PrimitiveSet const* pset,
    Operon::Span<Operon::Hash const> variables,
    std::bernoulli_distribution& sampleIrregular
) -> void {
    auto childDepth = RandomDequeue(random, q);

    auto maxArity = q.size() > 1 && sampleIrregular(random)
        ? size_t{0}
        : std::min(maxFunctionArity, targetLen - q.size() - nodes.size() - 1);

    if (maxArity > 0 && maxArity < minFunctionArity) {
        EXPECT(targetLen > 0);
        EXPECT(targetLen == 1 || targetLen >= minFunctionArity + 1);
        targetLen -= minFunctionArity - maxArity;
        maxArity = std::min(maxFunctionArity, targetLen - q.size() - nodes.size() - 1);
    }
    auto const minArity = std::min(minFunctionArity, maxArity);

    auto node = pset->SampleRandomSymbol(random, minArity, maxArity);
    InitNode(node, variables, random);
    node.Depth = static_cast<uint16_t>(childDepth);

    for (size_t i = 0; i < node.Arity; ++i) {
        q.push_back(childDepth + 1);
    }
    nodes.push_back(node);
}
} // anonymous namespace

namespace Operon {
auto ProbabilisticTreeCreator::operator()(Operon::RandomGenerator& random, size_t targetLen, size_t /*args*/, size_t /*args*/) const -> Tree
{
    EXPECT(targetLen > 0);
    auto const& variables = GetVariables();
    auto const& pset = GetPrimitiveSet();
    auto [minFunctionArity, maxFunctionArity] = pset->FunctionArityLimits();

    auto const requestedLen = targetLen;
    targetLen = AchievableLength(targetLen);

    Operon::Vector<Node> nodes;
    nodes.reserve(targetLen);

    auto maxArity = std::min(maxFunctionArity, targetLen - 1);
    auto minArity = std::min(minFunctionArity, maxArity);

    auto root = pset->SampleRandomSymbol(random, minArity, maxArity);
    InitNode(root, variables, random);

    if (root.IsLeaf()) {
        return Tree({ root }).UpdateNodes();
    }

    root.Depth = 1;
    nodes.push_back(root);

    std::deque<size_t> q;
    for (size_t i = 0; i < root.Arity; ++i) {
        q.push_back(root.Depth + 1U);
    }

    root.Parent = 0;

    std::bernoulli_distribution sampleIrregular(irregularityBias_);

    while (!q.empty()) {
        ProcessNextNode(random, q, nodes, targetLen, minFunctionArity, maxFunctionArity, pset, variables, sampleIrregular);
    }

    std::sort(nodes.begin(), nodes.end(), [](const auto& lhs, const auto& rhs) -> auto { return lhs.Depth < rhs.Depth; });
    auto const childIndices = ComputeChildIndices(nodes);

    Operon::Vector<Node> postfix(nodes.size());
    size_t idx = nodes.size();
    BuildPostfix(nodes, childIndices, postfix, idx, 0);

    auto tree = Tree(postfix).UpdateNodes();
    ENSURE(tree.Nodes().size() <= requestedLen);
    return tree;
}
} // namespace Operon
