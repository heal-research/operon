#include "mutation.hpp"

using namespace std;

namespace Operon
{
    Tree OnePointMutation::operator()(RandomDevice& random, const Tree& tree) const 
    {
        vector<size_t> leafIndices;
        for (size_t i = 0; i < tree.Length(); ++i)
        {
            if (tree[i].IsLeaf)
            {
                leafIndices.push_back(i);
            }
        }

        auto child = tree;
        uniform_int_distribution<size_t> uniformInt(0, leafIndices.size() - 1);
        auto& node = child[leafIndices[uniformInt(random)]];

        normal_distribution<double> normalReal(0, 1);
        node.Value += normalReal(random);

        return child;
    }
}

