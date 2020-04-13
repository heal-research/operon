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

#ifndef BALANCED_TREE_CREATOR_HPP
#define BALANCED_TREE_CREATOR_HPP

#include <algorithm>
#include <execution>
#include <stack>

#include "core/grammar.hpp"
#include "core/operator.hpp"

namespace Operon {

// this tree creator expands bread-wise using a "horizon" of open expansion slots
// at the end the breadth sequence of nodes is converted to a postfix sequence
// if the depth is not limiting, the target length is guaranteed to be reached
template <typename T>
class BalancedTreeCreator : public CreatorBase {
public:
    using U = std::tuple<Node, size_t, size_t>;

    BalancedTreeCreator(T distribution, size_t depth, size_t length, double bias = 1.0)
        : dist(distribution.param())
        , maxDepth(depth)
        , maxLength(length)
        , irregularityBias(bias)
    {
    }
    Tree operator()(Operon::Random& random, const Grammar& grammar, const gsl::span<const Variable> variables) const override
    {
        size_t minLength = 1u;
        size_t targetLen = std::clamp(SampleLength(random), minLength, maxLength);
        assert(targetLen > 0);

        auto [minFunctionArity, maxFunctionArity] = grammar.FunctionArityLimits();
        if (minFunctionArity > 1 && targetLen % 2 == 0) {
            targetLen = std::bernoulli_distribution(0.5)(random) ? targetLen - 1 : targetLen + 1;
        }

        std::uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1);
        std::normal_distribution<double> normalReal(0, 1);
        auto init = [&](Node& node) {
            if (node.IsVariable()) {
                node.HashValue = node.CalculatedHashValue = variables[uniformInt(random)].Hash;
            }
            node.Value = normalReal(random);
        };

        std::vector<U> tuples;
        tuples.reserve(targetLen);

        --targetLen; // we'll have at least a root symbol so we count it out
        auto minArity = std::min(minFunctionArity, targetLen);
        auto maxArity = std::min(maxFunctionArity, targetLen);

        auto root = grammar.SampleRandomSymbol(random, minArity, maxArity);
        init(root);
        tuples.emplace_back(root, 0, 1);

        size_t openSlots = root.Arity;

        std::bernoulli_distribution sampleIrregular(irregularityBias);

        for (size_t i = 0; i < tuples.size(); ++i) {
            auto [node, nodeDepth, childIndex] = tuples[i];
            auto childDepth = nodeDepth + 1;

            for (int j = 0; j < node.Arity; ++j) {
                maxArity = childDepth == maxDepth - 1 ? 0 : std::min(maxFunctionArity, targetLen - openSlots);
                minArity = std::min((openSlots - tuples.size() > 1 && sampleIrregular(random)) ? 0 : minFunctionArity, maxArity);
                auto child = grammar.SampleRandomSymbol(random, minArity, maxArity);
                init(child);
                if (j == 0)
                    std::get<2>(tuples[i]) = tuples.size();
                tuples.emplace_back(child, childDepth, 0);
                openSlots += child.Arity;
            }
        }
        auto nodes = BreadthToPostfix(tuples);
        auto tree = Tree(nodes).UpdateNodes();
        return tree;
    }

private:
    mutable T dist;
    size_t maxDepth;
    size_t maxLength;
    double irregularityBias;

    std::vector<Node> BreadthToPostfix(const std::vector<U>& tuples) const noexcept
    {
        int j = tuples.size();
        std::vector<Node> postfix(j);

        const auto add = [&](const U& t) {
            auto add_impl = [&](const U& t, auto& add_ref) {
                auto [node, nodeDepth, nodeChildIndex] = t;
                postfix[--j] = node;
                if (node.IsLeaf())
                    return;
                for (size_t i = nodeChildIndex; i < nodeChildIndex + node.Arity; ++i) {
                    add_ref(tuples[i], add_ref);
                }
            };
            add_impl(t, add_impl);
        };

        add(tuples.front());
        return postfix;
    }

    inline size_t SampleLength(Operon::Random& random) const
    {
        auto val = dist(random);
        if constexpr (std::is_floating_point_v<typename T::result_type>) {
            val = static_cast<size_t>(std::round(val));
        }
        return val;
    }
};
}
#endif
