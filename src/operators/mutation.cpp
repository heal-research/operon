#include "operators/mutation.hpp"

namespace Operon
{
    Tree OnePointMutation::operator()(RandomDevice& random, const Tree& tree) const 
    {
        auto& nodes = tree.Nodes();

        auto leafCount = std::count_if(nodes.begin(), nodes.end(), [](const Node& node) { return node.IsLeaf(); });
        std::uniform_int_distribution<gsl::index> uniformInt(1, leafCount);
        auto index = uniformInt(random);

        size_t i = 0;
        for (; i < nodes.size(); ++i)
        {
            if (nodes[i].IsLeaf() && --index == 0) break;
        }

        std::normal_distribution<double> normalReal(0, 1);
        auto child = tree;
        child[i].Value += normalReal(random);

        return child;
    }

    Tree MultiPointMutation::operator()(RandomDevice& random, const Tree& tree) const 
    {
        auto child = tree;
        std::normal_distribution<double> normalReal(0, 1);
        for(auto& node : child.Nodes())
        {
            node.Value += normalReal(random);
        }
        return child;
    }
}

