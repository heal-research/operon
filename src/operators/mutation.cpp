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

#include "operators/mutation.hpp"

namespace Operon {
Tree OnePointMutation::operator()(Operon::Random& random, Tree tree) const
{
    auto& nodes = tree.Nodes();

    auto leafCount = std::count_if(nodes.begin(), nodes.end(), [](const Node& node) { return node.IsLeaf(); });
    std::uniform_int_distribution<gsl::index> uniformInt(1, leafCount);
    auto index = uniformInt(random);

    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (nodes[i].IsLeaf() && --index == 0)
            break;
    }

    std::normal_distribution<double> normalReal(0, 1);
    tree[i].Value += normalReal(random);

    return tree;
}

Tree MultiPointMutation::operator()(Operon::Random& random, Tree tree) const
{
    std::normal_distribution<double> normalReal(0, 1);
    for (auto& node : tree.Nodes()) {
        if (node.IsLeaf()) {
            node.Value += normalReal(random);
        }
    }
    return tree;
}

Tree MultiMutation::operator()(Operon::Random& random, Tree tree) const
{
    //auto i = std::discrete_distribution<gsl::index>(probabilities.begin(), probabilities.end())(random);
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

Tree ChangeVariableMutation::operator()(Operon::Random& random, Tree tree) const
{
    auto& nodes = tree.Nodes();

    auto leafCount = std::count_if(nodes.begin(), nodes.end(), [](const Node& node) { return node.IsLeaf(); });
    std::uniform_int_distribution<gsl::index> uniformInt(1, leafCount);
    auto index = uniformInt(random);

    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (nodes[i].IsLeaf() && --index == 0)
            break;
    }

    std::uniform_int_distribution<gsl::index> normalInt(0, variables.size() - 1);
    tree[i].HashValue = tree[i].CalculatedHashValue = variables[normalInt(random)].Hash;

    return tree;
}

Tree ChangeFunctionMutation::operator()(Operon::Random& random, Tree tree) const {
    auto& nodes = tree.Nodes();
    auto funcCount = std::count_if(nodes.begin(), nodes.end(), [](const Node& node) { return !node.IsLeaf(); });

    if (funcCount == 0) {
        return tree;
    }

    std::uniform_int_distribution<gsl::index> uniformInt(1, funcCount);
    auto index = uniformInt(random);

    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (!nodes[i].IsLeaf() && --index == 0)
            break;
    }
    auto minArity = std::min(static_cast<size_t>(nodes[i].Arity), grammar.GetMinimumArity(nodes[i].Type));
    auto maxArity = std::max(static_cast<size_t>(nodes[i].Arity), grammar.GetMaximumArity(nodes[i].Type));
    nodes[i].Type = grammar.SampleRandomSymbol(random, minArity, maxArity).Type;
    return tree;
}

Tree ReplaceSubtreeMutation::operator()(Operon::Random& random, Tree tree) const {
    auto& nodes = tree.Nodes();

    auto i = std::uniform_int_distribution<size_t>(0, nodes.size()-1)(random);

    size_t oldLen = nodes[i].Length + 1;
    size_t oldLevel = tree.Level(i);

    size_t maxLength = maxLength_ - nodes.size() + oldLen;
    auto maxDepth = std::max(tree.Depth(), maxDepth_) - oldLevel + 1; 

    auto newLen = std::uniform_int_distribution<size_t>(1, maxLength)(random);
    auto subtree = creator_(random, newLen, 1, maxDepth).Nodes();

    Operon::Vector<Node> mutated;
    mutated.reserve(nodes.size() - oldLen + newLen);

    std::copy(nodes.begin(),         nodes.begin() + (i - nodes[i].Length), std::back_inserter(mutated));
    std::copy(subtree.begin(),       subtree.end(),                         std::back_inserter(mutated));
    std::copy(nodes.begin() + i + 1, nodes.end(),                           std::back_inserter(mutated));
    
    return Tree(mutated).UpdateNodes();
}

Tree InsertSubtreeMutation::operator()(Operon::Random& random, Tree tree) const {
    Expects(tree.Length() <= maxLength_);

    if (tree.Length() == maxLength_) {
        // we can't insert anything because the tree length is at the limit
        return tree;
    }

    auto& nodes = tree.Nodes();

    auto test = [](auto const& node) { return node.Type < NodeType::Log && node.Arity < 5; };

    auto n = std::count_if(nodes.begin(), nodes.end(), test);

    if (n == 0) {
        return tree;
    }

    auto index = std::uniform_int_distribution<size_t>(1, n)(random);
    size_t i = 0;
    for (; i < nodes.size(); ++i) {
        if (test(nodes[i]) && --index == 0) 
            break;
    }

    auto availableLength = maxLength_ - nodes.size();
    EXPECT(availableLength > 0);

    auto availableDepth = std::max(tree.Depth(), maxDepth_) - tree.Level(i); 
    EXPECT(availableDepth > 0);

    auto newLen = std::uniform_int_distribution<size_t>(1, availableLength)(random);

    auto subtree = creator_(random, newLen, 1, availableDepth).Nodes();

    Operon::Vector<Node> mutated;
    mutated.reserve(nodes.size() + newLen);

    // increase parent arity
    nodes[i].Arity++;

    // copy nodes
    std::copy(nodes.begin(),                         nodes.begin() + (i - nodes[i].Length), std::back_inserter(mutated));
    std::copy(subtree.begin(),                       subtree.end(),                         std::back_inserter(mutated));
    std::copy(nodes.begin() + (i - nodes[i].Length), nodes.end(),                           std::back_inserter(mutated));

    return Tree(mutated).UpdateNodes();
}
} // namespace Operon

