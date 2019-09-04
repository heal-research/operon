#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "core/operator.hpp"

namespace Operon 
{
    template<bool InPlace>
    struct OnePointMutation : public MutatorBase<InPlace>
    {
        void Mutate(operon::rand_t& random, Tree& tree) const
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
    };

    template<bool InPlace>
    struct MultiPointMutation : public MutatorBase<InPlace>
    {
        void Mutate(operon::rand_t& random, Tree& tree)
        {
            std::normal_distribution<double> normalReal(0, 1);
            for (auto& node : tree.Nodes())
            {
                if (node.IsLeaf())
                {
                    node.Value += normalReal(random);
                }
            }
        }
    };
}

#endif

