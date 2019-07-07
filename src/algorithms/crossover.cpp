#include "operators.hpp"

using namespace std;

namespace Operon
{
    size_t CutRandom(Rand& random, const Tree& tree, double internalProb)
    {
        uniform_real_distribution<double> uniformReal(0, 1);
        // create a vector of indices and shuffle it to ensure fair sampling
        vector<size_t> indices(tree.Length());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), random);

        auto chooseInternal = uniformReal(random) < internalProb;
        for (auto i : indices)
        {
            if (chooseInternal != tree[i].IsLeaf)
            {
                return i; 
            }
        }
        return indices.back();
    }

    optional<size_t> SelectRandomBranch(Rand& random, const Tree& tree, double internalProb, size_t maxLength, size_t maxDepth)
    {
        uniform_real_distribution<double> uniformReal(0, 1);
        // create a vector of indices and shuffle it to ensure fair sampling
        vector<size_t> indices(tree.Length());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), random);

        auto chooseInternal = uniformReal(random) < internalProb;
        for(auto i : indices)
        {
            if (tree[i].Length + 1U > maxLength || tree.Depth(i) > maxDepth)
            {
                continue;
            }

            if (chooseInternal != tree[i].IsLeaf)
            {
                return make_optional(i);
            }
        }
        return nullopt;
    }

    Tree Cross(Rand& random, const Tree& lhs, const Tree& rhs, double internalProb, int maxLength, int maxDepth)
    {
        auto& left = lhs.Nodes();
        auto& right = rhs.Nodes();

        auto i = CutRandom(random, lhs, internalProb);

        auto maxBranchDepth = maxDepth - lhs.Level(i);
        auto maxBranchLength = maxLength - (lhs.Length() - (left[i].Length + 1));

        if (auto result = SelectRandomBranch(random, rhs, internalProb, maxBranchLength, maxBranchDepth); result.has_value())
        {
            auto j = result.value();
            vector<Node> nodes;
            copy_n(left.begin(), i - left[i].Length, back_inserter(nodes)); 
            copy_n(right.begin() + j - right[j].Length, right[j].Length + 1, back_inserter(nodes));
            copy_n(left.begin() + i + 1, left.size() - (i + 1), back_inserter(nodes)); 

            return Tree(nodes);
        }

        return lhs;
    }
}

