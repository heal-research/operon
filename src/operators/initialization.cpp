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
    std::stack<std::tuple<Node, size_t, size_t>> stk;

    // always start with a function node
    auto root = grammar.SampleRandomSymbol(random, /* min arity */ 1, /* max arity */ 2);
    stk.push({ root, root.Arity, 1 }); // node, slot, depth, available length

    std::uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1);
    std::normal_distribution<double> normalReal(0, 1);

    size_t freeSpace = std::uniform_int_distribution<size_t>(root.Arity, maxLength - 1)(random);  
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

Tree FullTreeCreator::operator()(operon::rand_t& random, const Grammar& grammar, const gsl::span<const Variable> variables) const
{
    std::vector<Node> nodes;
    std::stack<std::tuple<Node, size_t, size_t>> stk;

    // always start with a function node
    auto root = grammar.SampleRandomSymbol(random, /* min arity */ 1, /* max arity */ 2);
    stk.push({ root, root.Arity, 1 }); // node, slot, depth, available length

    std::uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1);
    std::normal_distribution<double> normalReal(0, 1);

    size_t freeSpace = std::uniform_int_distribution<size_t>(root.Arity, maxLength - 1)(random);  
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
};
}
