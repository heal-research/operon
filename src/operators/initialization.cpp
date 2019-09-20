#include <algorithm>
#include <execution>
#include <stack>

#include "core/dataset.hpp"
#include "core/grammar.hpp"
#include "operators/initialization.hpp"

namespace Operon {
Tree GrowTreeCreator::operator()(operon::rand_t& random, const Grammar& grammar, const gsl::span<const Variable> variables) const
{
    std::vector<Node> nodes;

    std::stack<std::tuple<Node, int, size_t>> stk;

    // always start with a function node
    auto root = grammar.SampleRandomFunction(random);
    stk.push({ root, root.Arity, 1 }); // node, slot, depth, available length

    std::uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1);
    std::normal_distribution<double> normalReal(0, 1);

    int freeSpace = maxLength - (1 + root.Arity); // we have 1 root node + at least root.Arity children

    while (!stk.empty()) {
        auto [node, slot, depth] = stk.top();
        stk.pop();

        if (slot == 0) {
            nodes.push_back(node); // this node's children have been filled
            continue;
        }
        stk.push({ node, slot - 1, depth });

        auto child = freeSpace < 2 || depth == maxDepth - 1 // a function node can have arity 2 so we need at least (2+1) space
            ? grammar.SampleRandomTerminal(random)
            : grammar.SampleRandomSymbol(random);

        if (child.IsVariable()) {
            child.HashValue = child.CalculatedHashValue = variables[uniformInt(random)].Hash;
        }
        if (child.IsLeaf()) {
            child.Value = normalReal(random);
        }
        stk.push({ child, child.Arity, depth + 1 });
        freeSpace -= (1 + node.Arity);
    }
    auto tree = Tree(nodes).UpdateNodes();
    return tree;
}

Tree FullTreeCreator::operator()(operon::rand_t& random, const Grammar& grammar, const gsl::span<const Variable> variables) const
{
    std::vector<Node> nodes;

    std::stack<std::tuple<Node, int, size_t>> stk;

    auto root = grammar.SampleRandomFunction(random);
    stk.push({ root, root.Arity, 1 });

    std::uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1);
    std::normal_distribution<double> normalReal(0, 1);

    while (!stk.empty()) {
        auto [node, slot, depth] = stk.top();
        stk.pop();

        if (slot == 0) {
            nodes.push_back(node);
            continue;
        }
        stk.push({ node, slot - 1, depth });

        auto child = depth < maxDepth - 1
            ? grammar.SampleRandomFunction(random)
            : grammar.SampleRandomTerminal(random);

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
};
}
