// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <cstddef>
#include <algorithm>
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
    targetLen = AchievableLength(targetLen);

    using U = std::tuple<Node, size_t, size_t>;

    std::vector<U> tuples;
    tuples.reserve(targetLen);

    auto maxArity = std::min(maxFunctionArity, targetLen - 1);
    auto minArity = std::min(minFunctionArity, maxArity); // -1 because we start with a root

    auto root = pset->SampleRandomSymbol(random, minArity, maxArity);
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
            // minArity/maxArity recomputed fresh per child, not left sticky.
            auto candidateMax = std::min(maxFunctionArity, targetLen - committed);
            minArity = std::min(minFunctionArity, candidateMax);
            maxArity = openSlots > 1 && sampleIrregular(random) ? 0 : candidateMax;

            // fall back to a leaf node if the desired arity is not achievable with the current primitive set
            if (maxArity < minFunctionArity) {
                minArity = maxArity = 0;
            }

            auto child = pset->SampleRandomSymbol(random, minArity, maxArity);
            InitNode(child, variables, random);
            tuples.emplace_back(child, childDepth, 0);

            openSlots -= 1;
            committed += child.Arity;
            openSlots += child.Arity;
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
    ENSURE(tree.Nodes().size() <= requestedLen);
    return tree;
}
} // namespace Operon
