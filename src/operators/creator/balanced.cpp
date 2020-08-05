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

#include "operators/creator/balanced.hpp"

namespace Operon {
    Tree BalancedTreeCreator::operator()(Operon::Random& random, size_t targetLen, size_t, size_t maxDepth) const 
    {
        assert(targetLen > 0);
        const auto& grammar = grammar_.get();

        auto [minFunctionArity, maxFunctionArity] = grammar.FunctionArityLimits();
        if (minFunctionArity > 1 && targetLen % 2 == 0) {
            //targetLen = std::bernoulli_distribution(0.5)(random) ? targetLen - 1 : targetLen + 1;
            targetLen = targetLen - 1;
        }

        std::uniform_int_distribution<size_t> uniformInt(0, variables_.size() - 1);
        std::normal_distribution<double> normalReal(0, 1);
        auto init = [&](Node& node) {
            if (node.IsLeaf()) {
                if (node.IsVariable()) {
                    node.HashValue = variables_[uniformInt(random)].Hash;
                    node.CalculatedHashValue = node.HashValue; 
                }
                node.Value = normalReal(random);
            }
        };

        std::vector<U> tuples;
        tuples.reserve(targetLen);

        --targetLen; // we'll have at least a root symbol so we count it out
        auto minArity = std::min(minFunctionArity, targetLen);
        auto maxArity = std::min(maxFunctionArity, targetLen);

        if (maxDepth == 1) {
            minArity = 0;
            maxArity = 0;
        }

        auto root = grammar.SampleRandomSymbol(random, minArity, maxArity);
        init(root);
        tuples.emplace_back(root, 1, 1);

        size_t openSlots = root.Arity;

        std::bernoulli_distribution sampleIrregular(irregularityBias);

        for (size_t i = 0; i < tuples.size(); ++i) {
            auto [node, nodeDepth, childIndex] = tuples[i];
            auto childDepth = nodeDepth + 1;
            std::get<2>(tuples[i]) = tuples.size();
            for (int j = 0; j < node.Arity; ++j) {
                if (openSlots - tuples.size() > 1 && sampleIrregular(random)) {
                    minArity = 0;
                    maxArity = 0;
                } else {
                    maxArity = childDepth == maxDepth ? 0 : std::min(maxFunctionArity, targetLen - openSlots);
                    minArity = std::min(minFunctionArity, maxArity);
                }

                auto child = grammar.SampleRandomSymbol(random, minArity, maxArity);
                init(child);
                tuples.emplace_back(child, childDepth, 0);
                openSlots += child.Arity;
            }
        }

        Operon::Vector<Node> postfix(tuples.size());
        auto idx = tuples.size();

        const auto add = [&](const U& t) {
            auto add_impl = [&](const U& t, auto& add_ref) {
                auto [node, _, nodeChildIndex] = t;
                postfix[--idx] = node;
                if (node.IsLeaf()) {
                    return;
                }
                for (size_t i = nodeChildIndex; i < nodeChildIndex + node.Arity; ++i) {
                    add_ref(tuples[i], add_ref);
                }
            };
            add_impl(t, add_impl);
        };
        add(tuples.front());
        auto tree = Tree(postfix).UpdateNodes();
        if (tree.Depth() > maxDepth) {
            throw std::runtime_error(fmt::format("tree depth {} exceeds max depth {}\n", tree.Depth(), maxDepth));
        }
        return tree;
    }
}

