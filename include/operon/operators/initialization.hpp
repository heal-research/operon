/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
 */

#ifndef INITIALIZATION_HPP
#define INITIALIZATION_HPP

#include <algorithm>
#include <execution>
#include <stack>

#include "core/grammar.hpp"
#include "core/operator.hpp"

namespace Operon {

template <typename T>
class GrowTreeCreator : public CreatorBase {
public:
    GrowTreeCreator(T distribution, size_t depth, size_t length)
        : dist(distribution.param())
        , maxDepth(depth)
        , maxLength(length)
    {
    }
    Tree operator()(operon::rand_t& random, const Grammar& grammar, const gsl::span<const Variable> variables) const override
    {
        std::vector<Node> nodes;
        std::stack<std::tuple<Node, size_t, size_t, size_t>> stk;

        std::uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1);
        std::normal_distribution<double> normalReal(0, 1);

        size_t minLength = 1u; // a leaf as root node
        size_t targetLen = std::clamp(SampleFromDistribution(random), minLength, maxLength);
        Expects(targetLen > 0);

        auto [grammarMinArity, grammarMaxArity] = grammar.FunctionArityLimits();

        auto minArity = std::min(grammarMinArity, targetLen - 1);
        auto maxArity = std::min(grammarMaxArity, targetLen - 1);

        auto init = [&](Node& node) {
            if (node.IsVariable()) {
                node.HashValue = node.CalculatedHashValue = variables[uniformInt(random)].Hash;
            }
            node.Value = normalReal(random);
        };

        auto root = grammar.SampleRandomSymbol(random, minArity, maxArity);
        init(root);

        targetLen = targetLen - 1; // because we already added 1 root node
        size_t openSlots = root.Arity;
        stk.push({ root, root.Arity, 1, targetLen }); // node, slot, depth, available length

        auto childLen = 0ul;
        auto runningLength = root.Arity + 1; // 1 root node
        while (!stk.empty()) {
            auto [node, slot, depth, length] = stk.top();
            stk.pop();

            //fmt::print("{}: current length: {}\n", slot, runningLength);

            if (slot == 0) {
                nodes.push_back(node); // this node's children have been filled
                continue;
            }
            stk.push({ node, slot - 1, depth, length });

            childLen = slot == node.Arity ? length % node.Arity : childLen;
            childLen += length / node.Arity - 1;
            if (childLen > 0 && childLen % 2 == 0) childLen--;

            maxArity = depth == maxDepth - 1u ? 0u : std::min(grammarMaxArity, std::min(childLen, targetLen - openSlots));
            minArity = std::min(grammarMinArity, maxArity);
            auto child = grammar.SampleRandomSymbol(random, minArity, maxArity);
            init(child);

            runningLength += child.Arity;
            
            targetLen = targetLen - 1;
            openSlots = openSlots + child.Arity - 1;

            stk.push({ child, child.Arity, depth + 1, childLen });

        }
        auto tree = Tree(nodes).UpdateNodes();
        return tree;
    }

private:
    mutable T dist;
    size_t maxDepth;
    size_t maxLength;

    inline size_t SampleFromDistribution(operon::rand_t& random) const
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
