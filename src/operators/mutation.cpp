#include "operators/mutation.hpp"

namespace Operon
{
    void OnePointMutation::operator()(operon::rand_t& random, Tree& tree) const
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
        tree[i].Value += normalReal(random);
    }

    void MultiPointMutation::operator()(operon::rand_t& random, Tree& tree) const
    {
        std::normal_distribution<double> normalReal(0, 1);
        for(auto& node : tree.Nodes())
        {
            node.Value += normalReal(random);
        }
    }
}

