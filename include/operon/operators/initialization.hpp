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
    using SizeDistributionParamType = typename T::param_type;

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

        auto childLenLeft = 0u;
        while (!stk.empty()) {
            auto [node, slot, depth, length] = stk.top();
            stk.pop();

            if (slot == 0) {
                nodes.push_back(node); // this node's children have been filled
                continue;
            }
            stk.push({ node, slot - 1, depth, length });

            auto childLen = length / node.Arity - 1 + childLenLeft;

            if (slot == node.Arity)
                childLen += length % node.Arity;

            maxArity = depth == maxDepth - 1u ? 0u : std::min(grammarMaxArity, std::min(childLen, targetLen - openSlots));
            minArity = std::min(grammarMinArity, maxArity);
            auto child = grammar.SampleRandomSymbol(random, minArity, maxArity);
            init(child);

            childLenLeft = child.IsLeaf() ? childLen : 0u;

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
