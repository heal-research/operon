// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "operators/mutation.hpp"

namespace Operon {
Tree OnePointMutation::operator()(Operon::RandomGenerator& random, Tree tree) const
{
    auto& nodes = tree.Nodes();
    // sample a random leaf
    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) { return n.IsLeaf(); });
    EXPECT(it < nodes.end());
    std::normal_distribution<Operon::Scalar> normalReal(0, 1);
    it->Value += normalReal(random);

    return tree;
}

Tree MultiPointMutation::operator()(Operon::RandomGenerator& random, Tree tree) const
{
    std::normal_distribution<Operon::Scalar> normalReal(0, 1);
    for (auto& node : tree.Nodes()) {
        if (node.IsLeaf()) {
            node.Value += normalReal(random);
        }
    }
    return tree;
}

Tree MultiMutation::operator()(Operon::RandomGenerator& random, Tree tree) const
{
    auto sum = std::reduce(probabilities.begin(), probabilities.end());
    auto r = std::uniform_real_distribution<double>(0, sum)(random);
    auto c = 0.0;
    auto i = 0u; 
    for (; i < probabilities.size(); ++i) {
        c += probabilities[i];
        if (c > r) {
            break;
        }
    }
    auto op = operators[i];
    return op(random, std::move(tree));
}

Tree ChangeVariableMutation::operator()(Operon::RandomGenerator& random, Tree tree) const
{
    auto& nodes = tree.Nodes();
    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) { return n.IsVariable(); });
    if (it == nodes.end())
        return tree; // no variables in the tree, nothing to do

    it->HashValue = it->CalculatedHashValue = Sample(random, variables.begin(), variables.end())->Hash;
    return tree;
}

Tree ChangeFunctionMutation::operator()(Operon::RandomGenerator& random, Tree tree) const {
    auto& nodes = tree.Nodes();

    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) { return !n.IsLeaf(); });
    if (it == nodes.end()) 
        return tree; // no functions in the tree, nothing to do

    auto minArity = std::min(static_cast<size_t>(it->Arity), pset.GetMinimumArity(it->Type));
    auto maxArity = std::max(static_cast<size_t>(it->Arity), pset.GetMaximumArity(it->Type));

    auto n = pset.SampleRandomSymbol(random, minArity, maxArity);
    it->Type = n.Type;
    it->HashValue = n.HashValue;
    return tree;
}

Tree ReplaceSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const {
    auto& nodes = tree.Nodes();

    auto i = std::uniform_int_distribution<size_t>(0, nodes.size()-1)(random);

    auto oldLen = nodes[i].Length + 1u;
    auto oldLevel = nodes[i].Level;

    using signed_t = std::make_signed<size_t>::type;

    auto partialLength = nodes.size() - oldLen;

    // the correction below is necessary because it can happen that maxLength_ < nodes.size()
    // (for example when the tree creator cannot achieve exactly a target length and
    //  then it creates a slightly larger tree)
    auto maxLength = static_cast<signed_t>(maxLength_ - partialLength);
    maxLength = std::max(maxLength, signed_t{1});

    auto maxDepth = std::max(tree.Depth(), maxDepth_) - oldLevel + 1; 

    auto newLen = std::uniform_int_distribution<signed_t>(signed_t{1}, maxLength)(random);
    auto subtree = creator_(random, static_cast<size_t>(newLen), 1, maxDepth).Nodes();

    Operon::Vector<Node> mutated;
    mutated.reserve(nodes.size() - oldLen + static_cast<size_t>(newLen));

    std::copy(nodes.begin(),         nodes.begin() + static_cast<long>(i - nodes[i].Length), std::back_inserter(mutated));
    std::copy(subtree.begin(),       subtree.end(),                         std::back_inserter(mutated));
    std::copy(nodes.begin() + static_cast<long>(i + 1), nodes.end(),                           std::back_inserter(mutated));
    
    return Tree(mutated).UpdateNodes();
}

Tree RemoveSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const {
    auto& nodes = tree.Nodes();

    if (nodes.size() == 1)
        return tree; // nothing to remove

    auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end()-1); // -1 because we don't want to remove the tree root
    auto const& p = nodes[it->Parent];
    if (p.Arity > pset.GetMinimumArity(p.Type)) {
        nodes[it->Parent].Arity--;
        nodes.erase(it - it->Length, it + 1);
        tree.UpdateNodes();
    }
    return tree;
}

Tree InsertSubtreeMutation::operator()(Operon::RandomGenerator& random, Tree tree) const {
    if (tree.Length() >= maxLength_) {
        // we can't insert anything because the tree length is at the limit
        return tree;
    }

    auto& nodes = tree.Nodes();

    auto test = [](auto const& node) { return static_cast<bool>(node.Type & (NodeType::Add | NodeType::Mul)); };

    auto n = std::count_if(nodes.begin(), nodes.end(), test);

    if (n == 0) {
        return tree;
    }

    auto index = std::uniform_int_distribution<decltype(n)>(1, n)(random);
    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (test(nodes[i]) && --index == 0) 
            break;
    }

    auto availableLength = maxLength_ - nodes.size();
    EXPECT(availableLength > 0);

    auto availableDepth = std::max(tree.Depth(), maxDepth_) - nodes[i].Level; 
    EXPECT(availableDepth > 0);

    auto newLen = std::uniform_int_distribution<size_t>(1, availableLength)(random);

    auto subtree = creator_(random, newLen, 1, availableDepth).Nodes();

    Operon::Vector<Node> mutated;
    mutated.reserve(nodes.size() + newLen);

    // increase parent arity
    nodes[i].Arity++;

    // copy nodes
    std::copy(nodes.begin(),                         nodes.begin() + static_cast<long>(i - nodes[i].Length), std::back_inserter(mutated));
    std::copy(subtree.begin(),                       subtree.end(),                         std::back_inserter(mutated));
    std::copy(nodes.begin() + static_cast<long>(i - nodes[i].Length), nodes.end(),                           std::back_inserter(mutated));

    return Tree(mutated).UpdateNodes();
}

Tree ShuffleSubtreesMutation::operator()(Operon::RandomGenerator& random, Tree tree) const {
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
        if (nodes[i].IsLeaf()) 
            continue;

        if (--idx == 0) 
            break;
    }
    auto const& s = nodes[i];

    // the child nodes will be shuffled so we keep them in a buffer 
    std::vector<Node> buffer(nodes.begin() + static_cast<long>(i) - s.Length, nodes.begin() + static_cast<long>(i));
    EXPECT(buffer.size() == s.Length);

    // get child indices relative to buffer 
    std::vector<size_t> childIndices(s.Arity);

    size_t j = s.Length-1;
    for (uint16_t k = 0; k < s.Arity; ++k) {
        childIndices[k] = j;
        j -= buffer[j].Length + 1u;
    }

    // shuffle child indices
    std::shuffle(childIndices.begin(), childIndices.end(), random);

    //// write back from buffer to nodes in the shuffled order
    auto insertionPoint = nodes.begin() + static_cast<long>(i) - s.Length;

    for (auto k : childIndices) {
        std::copy(buffer.begin() + static_cast<long>(k) - buffer[k].Length, buffer.begin() + static_cast<long>(k) + 1, insertionPoint);
        insertionPoint += buffer[k].Length + 1u;
    }

    return tree.UpdateNodes();
}

} // namespace Operon

