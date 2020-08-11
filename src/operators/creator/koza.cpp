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

#include "operators/creator/koza.hpp"

namespace Operon {
    Tree GrowTreeCreator::operator()(Operon::Random& random, size_t, size_t minDepth, size_t maxDepth) const
    { 
        minDepth = std::max(1ul, minDepth);
        EXPECT(minDepth <= maxDepth);
        const auto& grammar = grammar_.get();

        const auto& t = grammar.FunctionArityLimits();
        
        size_t minFunctionArity = std::get<0>(t);
        size_t maxFunctionArity = std::get<1>(t);

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

        Operon::Vector<Node> nodes;
        size_t minArity = minFunctionArity;
        size_t maxArity = maxFunctionArity;

        std::uniform_int_distribution<size_t> dist(minDepth, maxDepth);
        auto actualDepthLimit = dist(random);
        bool minDepthReached = false;

        const auto grow = [&](size_t depth) {
            const auto grow_impl = [&](size_t depth, const auto& grow_ref) {
                minDepthReached = depth >= minDepth;

                if (depth >= actualDepthLimit) {
                    minArity = 0;
                    maxArity = 0;
                } else {
                    minArity = minDepthReached ? 0 : minFunctionArity; 
                    maxArity = maxFunctionArity;
                }

                auto node = grammar.SampleRandomSymbol(random, minArity, maxArity);
                init(node);

                nodes.push_back(node);

                if (node.IsLeaf()) {
                    return;
                }

                for (size_t i = 0; i < node.Arity; ++i) {
                    grow_ref(depth + 1, grow_ref); 
                }
            };
            grow_impl(depth, grow_impl);
        };

        grow(1);

        std::reverse(nodes.begin(), nodes.end());
        return Tree(nodes).UpdateNodes();
    }
}
