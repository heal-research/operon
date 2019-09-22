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
        : dist(distribution)
        , maxDepth(depth)
        , maxLength(length)
    {
    }
    Tree operator()(operon::rand_t& random, const Grammar& grammar, const gsl::span<const Variable> variables) const override
    {
        std::vector<Node> nodes;
        std::stack<std::tuple<Node, size_t, size_t>> stk;

        // always start with a function node
        auto root = grammar.SampleRandomSymbol(random, /* min arity */ 1, /* max arity */ 2);
        stk.push({ root, root.Arity, 1 }); // node, slot, depth, available length

        std::uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1);
        std::normal_distribution<double> normalReal(0, 1);

        size_t freeSpace = SampleFromDistribution(random);
        size_t minLength = root.Arity;
        freeSpace = std::clamp(freeSpace, minLength, maxLength - 1);
        size_t openSlots = root.Arity;

        while (!stk.empty()) {
            auto [node, slot, depth] = stk.top();
            stk.pop();

            if (slot == 0) {
                nodes.push_back(node); // this node's children have been filled
                continue;
            }
            stk.push({ node, slot - 1, depth });

            size_t minArity = 1u;
            size_t maxArity = depth == maxDepth - 1u ? 0u : freeSpace - openSlots;
            auto child = grammar.SampleRandomSymbol(random, std::min(minArity, maxArity), maxArity);
            freeSpace = freeSpace - 1;
            openSlots = openSlots + child.Arity - 1;

            if (child.IsVariable()) {
                child.HashValue = child.CalculatedHashValue = variables[uniformInt(random)].Hash;
            }
            if (child.IsLeaf()) {
                child.Value = normalReal(random);
            }
            stk.push({ child, child.Arity, depth + 1 });
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
