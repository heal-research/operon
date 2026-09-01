// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <cstddef>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include "operon/operators/creator.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
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
} // anonymous namespace

namespace Operon {
auto BalancedTreeCreator::operator()(Operon::RandomGenerator& random, size_t targetLen, size_t /*args*/, size_t /*args*/) const -> Tree
{
    EXPECT(targetLen > 0);
    auto const& pset = GetPrimitiveSet();
    auto [minFunctionArity, maxFunctionArity] = pset->FunctionArityLimits();

    auto const& variables = GetVariables();

    auto const requestedLen = targetLen;

    using U = std::tuple<Node, size_t, size_t>;

    std::vector<U> tuples;
    tuples.reserve(targetLen);

    // dp[i] records whether i additional child slots can be filled by enabled
    // function arities. Build it from the current pset once per tree: creators
    // may outlive pset reconfiguration, so CreatorBase's cached snap table is
    // insufficient here.
    std::vector<bool> completable(requestedLen, false);
    completable[0] = true;
    for (size_t i = 1; i < requestedLen; ++i) {
        for (auto const& [_, primitive] : pset->Primitives()) {
            auto const& [node, frequency, minArity, maxArity] = primitive;
            if (node.IsLeaf() || !node.IsEnabled || frequency == 0) { continue; }
            for (size_t arity = minArity; arity <= std::min(maxArity, i); ++arity) {
                if (completable[i - arity]) {
                    completable[i] = true;
                    break;
                }
            }
            if (completable[i]) { break; }
        }
    }
    while (!completable[targetLen - 1]) {
        --targetLen;
    }

    auto sampleCompletable = [&](size_t max, size_t remaining) -> Node {
        auto total = 0.0;
        for (auto const& [_, primitive] : pset->Primitives()) {
            auto const& [node, frequency, minArity, maxArity] = primitive;
            if (node.IsLeaf() || !node.IsEnabled || frequency == 0 || minArity > max) { continue; }
            auto const upper = std::min(maxArity, max);
            auto const count = upper - minArity + 1;
            for (size_t arity = minArity; arity <= std::min(upper, remaining); ++arity) {
                if (completable[remaining - arity]) {
                    total += static_cast<double>(frequency) / static_cast<double>(count);
                }
            }
        }

        EXPECT(total > 0.0);
        auto selected = std::uniform_real_distribution<double>(0.0, total)(random);
        for (auto const& [_, primitive] : pset->Primitives()) {
            auto const& [node, frequency, minArity, maxArity] = primitive;
            if (node.IsLeaf() || !node.IsEnabled || frequency == 0 || minArity > max) { continue; }
            auto const upper = std::min(maxArity, max);
            auto const weight = static_cast<double>(frequency) / static_cast<double>(upper - minArity + 1);
            for (size_t arity = minArity; arity <= std::min(upper, remaining); ++arity) {
                if (!completable[remaining - arity]) { continue; }
                if (selected < weight) {
                    auto result = node;
                    result.Arity = static_cast<uint16_t>(arity);
                    return result;
                }
                selected -= weight;
            }
        }
        UNREACHABLE();
    };

    auto maxArity = std::min(maxFunctionArity, targetLen - 1);
    auto minArity = std::min(minFunctionArity, maxArity); // -1 because we start with a root

    if (maxArity == 0) {
        auto root = pset->SampleRandomSymbol(random, 0, 0);
        InitNode(root, variables, random);
        return Tree({ root }).UpdateNodes();
    }

    auto root = sampleCompletable(maxArity, targetLen - 1);
    InitNode(root, variables, random);

    if (root.IsLeaf()) {
        return Tree({ root }).UpdateNodes();
    }

    tuples.emplace_back(root, 1, 1);

    // Remaining-slots counter: decrements as each pending slot is filled,
    // increments by each new child's arity.
    size_t openSlots = root.Arity;
    // Total committed node count so far, used to cap a child's arity against
    // the remaining length budget.
    size_t committed = root.Arity + 1;

    std::bernoulli_distribution sampleIrregular(irregularityBias_);

    for (size_t i = 0; i < tuples.size(); ++i) {
        auto [node, nodeDepth, childIndex] = tuples[i];
        auto childDepth = nodeDepth + 1;
        std::get<2>(tuples[i]) = tuples.size();
        for (int j = 0; std::cmp_less(j , node.Arity); ++j) {
            auto candidateMax = std::min(maxFunctionArity, targetLen - committed);
            auto const remaining = targetLen - committed;
            if (candidateMax < minFunctionArity || (openSlots > 1 && sampleIrregular(random))) {
                minArity = maxArity = 0;
                auto child = pset->SampleRandomSymbol(random, minArity, maxArity);
                InitNode(child, variables, random);
                tuples.emplace_back(child, childDepth, 0);
            } else {
                auto child = sampleCompletable(candidateMax, remaining);
                InitNode(child, variables, random);
                tuples.emplace_back(child, childDepth, 0);
            }

            openSlots -= 1;
            auto const childArity = std::get<0>(tuples.back()).Arity;
            committed += childArity;
            openSlots += childArity;
        }
    }

    Operon::Vector<Node> postfix(tuples.size());
    auto idx = tuples.size();

    auto add = [&](const U& t, auto&& ref) -> auto {
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
    ENSURE(tree.Nodes().size() == targetLen);
    ENSURE(tree.Nodes().size() <= requestedLen);
    return tree;
}
} // namespace Operon
