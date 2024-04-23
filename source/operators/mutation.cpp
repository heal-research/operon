// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include "operon/operators/mutation.hpp"

#include <algorithm>
#include <cstdint>
#include <iterator>
#include <numeric>
#include <random>
#include <type_traits>

#include "operon/operators/creator.hpp"
#include "operon/operators/initializer.hpp"

namespace Operon {

auto DiscretePointMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();
    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) { return n.IsLeaf(); });
    ENSURE(it < nodes.end());

    auto s = std::reduce(weights_.cbegin(), weights_.cend(), Operon::Scalar { 0 }, std::plus {});
    auto r = std::uniform_real_distribution<Operon::Scalar>(0., s)(random);

    Operon::Scalar c { 0 };
    for (auto i = 0UL; i < weights_.size(); ++i) {
        c += weights_[i];
        if (c > r) {
            it->Value = values_[i];
            break;
        }
    }

    return tree;
}

auto MultiMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto sum = std::reduce(probabilities_.begin(), probabilities_.end());
    auto r = std::uniform_real_distribution<double>(0, sum)(random);
    auto c = 0.0;
    auto i = 0U;
    for (; i < probabilities_.size(); ++i) {
        c += probabilities_[i];
        if (c > r) {
            break;
        }
    }
    auto op = operators_[i];
    return op(random, std::move(tree));
}

auto ChangeVariableMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();
    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) { return n.IsVariable(); });
    if (it == nodes.end()) {
        return tree; // no variables in the tree, nothing to do
    }

    it->HashValue = it->CalculatedHashValue = *Sample(random, variables_.begin(), variables_.end());
    return tree;
}

auto ChangeFunctionMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();

    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) { return !n.IsLeaf(); });
    if (it == nodes.end()) {
        return tree; // no functions in the tree, nothing to do
    }

    auto minArity = std::min(static_cast<size_t>(it->Arity), pset_.MinimumArity(it->HashValue));
    auto maxArity = std::max(static_cast<size_t>(it->Arity), pset_.MaximumArity(it->HashValue));

    auto n = pset_.SampleRandomSymbol(random, minArity, maxArity);
    it->Type = n.Type;
    it->HashValue = n.HashValue;
    return tree;
}

auto ReplaceSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();

    auto i = std::uniform_int_distribution<size_t>(0, nodes.size() - 1)(random);

    auto oldLen = nodes[i].Length + 1U;
    auto oldLevel = nodes[i].Level;

    using Signed = std::make_signed<size_t>::type;

    auto partialLength = nodes.size() - oldLen;

    // the correction below is necessary because it can happen that maxLength_ < nodes.size()
    // (for example when the tree creator can't achieve the target length exactly and creates a slightly larger tree)
    auto maxLength = static_cast<Signed>(maxLength_ - partialLength);
    maxLength = std::max(maxLength, Signed { 1 });

    auto maxDepth = std::max(tree.Depth(), maxDepth_) - oldLevel + 1;

    auto newLen = std::uniform_int_distribution<Signed>(Signed { 1 }, maxLength)(random);
    auto subtree = creator_(random, static_cast<size_t>(newLen), 1, maxDepth);
    coefficientInitializer_(random, subtree);

    Operon::Vector<Node> mutated;
    mutated.reserve(nodes.size() - oldLen + static_cast<size_t>(newLen));

    using Signed = std::make_signed_t<size_t>;
    std::copy(nodes.begin(), nodes.begin() + static_cast<Signed>(i - nodes[i].Length), std::back_inserter(mutated));
    std::copy(subtree.Nodes().begin(), subtree.Nodes().end(), std::back_inserter(mutated));
    std::copy(nodes.begin() + static_cast<Signed>(i + 1), nodes.end(), std::back_inserter(mutated));

    return Tree(mutated).UpdateNodes();
}

auto RemoveSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();

    if (nodes.size() == 1) {
        return tree; // nothing to remove
    }

    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end() - 1); // -1 because we don't want to remove the tree root
    auto const& p = nodes[it->Parent];
    if (p.Arity > pset_.MinimumArity(p.HashValue)) {
        nodes[it->Parent].Arity--;
        nodes.erase(it - it->Length, it + 1);
        tree.UpdateNodes();
    }
    return tree;
}

auto InsertSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    if (tree.Length() >= maxLength_) {
        // we can't insert anything because the tree length is at the limit
        return tree;
    }

    auto& nodes = tree.Nodes();
    auto const& creator = creator_.get();
    auto const& pset = creator.GetPrimitiveSet();

    auto test = [&](auto const& node) {
        return static_cast<bool>(node.Type & (NodeType::Add | NodeType::Mul | NodeType::Sub | NodeType::Div)) && (node.Arity < pset.MaximumArity(node.HashValue));
    };

    auto n = std::count_if(nodes.begin(), nodes.end(), test);

    if (n == 0) {
        return tree;
    }

    auto index = std::uniform_int_distribution<decltype(n)>(1, n)(random);
    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (test(nodes[i]) && --index == 0) {
            break;
        }
    }

    auto availableLength = maxLength_ - nodes.size();
    EXPECT(availableLength > 0);

    auto availableDepth = std::max(tree.Depth(), maxDepth_) - nodes[i].Level;
    EXPECT(availableDepth > 0);

    auto newLen = std::uniform_int_distribution<size_t>(1, availableLength)(random);

    auto subtree = creator_(random, newLen, 1, availableDepth);
    coefficientInitializer_(random, subtree);

    Operon::Vector<Node> mutated;
    mutated.reserve(nodes.size() + newLen);

    // increase parent arity
    nodes[i].Arity++;

    using Signed = std::make_signed_t<size_t>;
    // copy nodes
    std::copy(nodes.begin(), nodes.begin() + static_cast<Signed>(i - nodes[i].Length), std::back_inserter(mutated));
    std::copy(subtree.Nodes().begin(), subtree.Nodes().end(), std::back_inserter(mutated));
    std::copy(nodes.begin() + static_cast<Signed>(i - nodes[i].Length), nodes.end(), std::back_inserter(mutated));

    return Tree(mutated).UpdateNodes();
}

auto ShuffleSubtreesMutation::operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree
{
    auto& nodes = tree.Nodes();
    auto nFunc = std::count_if(nodes.begin(), nodes.end(), [](const auto& node) { return !node.IsLeaf(); });

    if (nFunc == 0) {
        return tree;
    }

    // pick a random function node
    auto idx = std::uniform_int_distribution<std::make_signed_t<size_t>>(1, nFunc)(random);

    // find the function node in the nodes array
    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (nodes[i].IsLeaf()) {
            continue;
        }

        if (--idx == 0) {
            break;
        }
    }
    auto const& s = nodes[i];

    // the child nodes will be shuffled so we keep them in a buffer
    using Signed = std::make_signed_t<size_t>;
    std::vector<Node> buffer(nodes.begin() + static_cast<Signed>(i) - s.Length, nodes.begin() + static_cast<Signed>(i));
    EXPECT(buffer.size() == s.Length);

    // get child indices relative to buffer
    std::vector<size_t> childIndices(s.Arity);

    size_t j = s.Length - 1;
    for (uint16_t k = 0; k < s.Arity; ++k) {
        childIndices[k] = j;
        j -= buffer[j].Length + 1U;
    }

    // shuffle child indices
    std::shuffle(childIndices.begin(), childIndices.end(), random);

    // write back from buffer to nodes in the shuffled order
    auto insertionPoint = nodes.begin() + std::make_signed_t<decltype(i)>(i) - s.Length;

    for (auto k : childIndices) {
        std::copy(buffer.begin() + static_cast<Signed>(k) - buffer[k].Length, buffer.begin() + static_cast<Signed>(k) + 1, insertionPoint);
        insertionPoint += buffer[k].Length + 1U;
    }

    return tree.UpdateNodes();
}
} // namespace Operon
