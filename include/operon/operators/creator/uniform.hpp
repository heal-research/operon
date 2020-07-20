/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
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

#ifndef UNIFORM_TREE_CREATOR_HPP
#define UNIFORM_TREE_CREATOR_HPP

#include <algorithm>
#include <execution>
#include <stack>

#include "core/grammar.hpp"
#include "core/operator.hpp"

namespace Operon {
// this tree creator follows a user-defined tree length distribution
// and produces symbol frequencies according to the grammar
// this comes at the cost of left-leaning trees (heavily unbalanced)
class UniformTreeCreator : public CreatorBase {
public:
    UniformTreeCreator(const Grammar& grammar, const gsl::span<const Variable> variables)
        : CreatorBase(grammar, variables) 
    {
    }
    Tree operator()(Operon::Random& random, size_t targetLen, size_t, size_t maxDepth) const override
    {
        Operon::Vector<Node> nodes;
        std::stack<std::tuple<Node, size_t, size_t>> stk;

        std::uniform_int_distribution<size_t> uniformInt(0, variables_.size() - 1);
        std::normal_distribution<double> normalReal(0, 1);

        assert(targetLen > 0);

        const auto& grammar = grammar_.get();

        auto [grammarMinArity, grammarMaxArity] = grammar.FunctionArityLimits();

        auto minArity = std::min(grammarMinArity, targetLen - 1);
        auto maxArity = std::min(grammarMaxArity, targetLen - 1);

        auto init = [&](Node& node) {
            if (node.IsVariable()) {
                node.HashValue = node.CalculatedHashValue = variables_[uniformInt(random)].Hash;
            }
            node.Value = normalReal(random);
        };

        auto root = grammar.SampleRandomSymbol(random, minArity, maxArity);
        init(root);

        targetLen = targetLen - 1; // because we already added 1 root node
        size_t openSlots = root.Arity;
        stk.emplace(root, root.Arity, 1); // node, slot, depth, available length

        while (!stk.empty()) {
            auto [node, slot, depth] = stk.top();
            stk.pop();

            if (slot == 0) {
                nodes.push_back(node); // this node's children have been filled
                continue;
            }
            stk.emplace(node, slot - 1, depth);

            maxArity = depth == maxDepth - 1u ? 0u : std::min(grammarMaxArity, targetLen - openSlots);
            minArity = std::min(grammarMinArity, maxArity);
            auto child = grammar.SampleRandomSymbol(random, minArity, maxArity);
            init(child);

            targetLen = targetLen - 1;
            openSlots = openSlots + child.Arity - 1;

            stk.emplace(child, child.Arity, depth + 1);
        }
        auto tree = Tree(nodes).UpdateNodes();
        return tree;
    }
};
}
#endif
