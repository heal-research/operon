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

#include "operators/creator/ptc2.hpp"

namespace Operon {
    Tree ProbabilisticTreeCreator::operator()(Operon::Random& random, size_t targetLen, size_t maxDepth) const 
    {
        assert(targetLen > 0);
        const auto& grammar = grammar_.get();

        auto [minFunctionArity, maxFunctionArity] = grammar.FunctionArityLimits();
        if (minFunctionArity > 1 && targetLen % 2 == 0) {
            targetLen = std::bernoulli_distribution(0.5)(random) ? targetLen - 1 : targetLen + 1;
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

        // the PTC2 returns the targetLen + at most maxFunctionArity extra leafs
        std::vector<Node> nodes;
        nodes.reserve(targetLen + maxFunctionArity);

        auto maxArity = std::min(maxFunctionArity, targetLen - 1);
        auto minArity = std::min(minFunctionArity, maxArity);

        auto root = grammar.SampleRandomSymbol(random, minArity, maxArity);
        init(root);
        root.Depth = 0;

        nodes.push_back(root);

        if (root.IsLeaf()) {
            nodes.resize(1);
            auto tree = Tree(nodes).UpdateNodes();
            return tree;
        }

        // Tuple U: 1. parent index 2. argPos 3. depth
        using U = std::tuple<size_t, size_t, size_t>; // 
        std::deque<U> q;

        for (size_t i = 0; i < root.Arity; ++i) {
            auto d = root.Depth + 1;
            q.emplace_back(0, i, d); 
        }

        std::unordered_map<int, std::vector<int>> pos;

        // emulate a random dequeue operation 
        auto random_dequeue = [&]() -> U {
            auto j = std::uniform_int_distribution<size_t>(0, q.size()-1)(random);
            std::swap(q[j], q.front());
            auto t = q.front();
            q.pop_front();
            return t;
        };

        while (q.size() > 0 && q.size() + nodes.size() < targetLen) {
            auto [parentIndex, childIndex, childDepth] = random_dequeue();

            maxArity = std::min(maxFunctionArity, targetLen - q.size() - nodes.size() - 1);
            minArity = std::min(minFunctionArity, maxArity);

            auto node = childDepth == maxDepth
                ? grammar.SampleRandomSymbol(random, 0, 0) 
                : grammar.SampleRandomSymbol(random, minArity, maxArity);

            init(node);
            node.Depth = childDepth;

            auto it = pos.find(parentIndex);
            if (it == pos.end()) {
                pos[parentIndex] = std::vector<int>(nodes[parentIndex].Arity);
            }
            pos[parentIndex][childIndex] = nodes.size();

            for (size_t i = 0; i < node.Arity; ++i) {
                q.emplace_back(nodes.size(), i, childDepth+1);
            }

            nodes.push_back(node);
        }

        while (q.size() > 0) {
            auto [parentIndex, childIndex, childDepth] = random_dequeue();
            auto node = grammar.SampleRandomSymbol(random, 0, 0);
            init(node);
            node.Depth = childDepth;
            auto it = pos.find(parentIndex);
            if (it == pos.end()) {
                pos[parentIndex] = std::vector<int>(nodes[parentIndex].Arity);
            }
            pos[parentIndex][childIndex] = nodes.size();
            nodes.push_back(node);
        }

        auto tmp = nodes;
        int idx = nodes.size()-1;

        const auto add = [&](size_t i) {
            auto add_impl = [&](size_t i, auto& add_ref) {
                nodes[idx--] = tmp[i];

                if (tmp[i].IsLeaf()) { 
                    return; 
                }

                for (auto j : pos[i]) {
                    add_ref(j, add_ref);
                }
            };
            add_impl(i, add_impl);
        };
        add(0);

        auto tree = Tree(nodes).UpdateNodes();
        return tree;
    }
}
