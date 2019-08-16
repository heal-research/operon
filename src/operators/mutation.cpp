#include "operators/mutation.hpp"

namespace Operon
{
    Tree OnePointMutation::operator()(RandomDevice& random, const Tree& tree) const 
    {
        std::vector<size_t> leafIndices;
        for (size_t i = 0; i < tree.Length(); ++i)
        {
            if (tree[i].IsLeaf())
            {
                leafIndices.push_back(i);
            }
        }

        auto child = tree;
        std::uniform_int_distribution<size_t> uniformInt(0, leafIndices.size() - 1);
        auto& node = child[leafIndices[uniformInt(random)]];

        std::normal_distribution<double> normalReal(0, 1);
        node.Value += normalReal(random);

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

