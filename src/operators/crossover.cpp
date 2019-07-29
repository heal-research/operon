#include "crossover.hpp"

using namespace std;

namespace Operon
{
    optional<size_t> SubtreeCrossover::SelectRandomBranch(RandomDevice& random, const Tree& tree, double internalProb, size_t maxBranchDepth, size_t maxBranchLength) const
    {
        std::uniform_real_distribution<double> uniformReal(0, 1);
        // create a vector of indices and shuffle it to ensure fair sampling
        vector<size_t> indices(tree.Length());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), random);

        auto chooseInternal = uniformReal(random) < internalProb;
        for(auto i : indices)
        {
            if (tree[i].Length + 1U > maxBranchLength || tree.Depth(i) > maxBranchDepth)
            {
                continue;
            }

            if (chooseInternal != tree[i].IsLeaf)
            {
                return make_optional(i);
            }
        }
        return std::nullopt;
    }

    size_t SubtreeCrossover::CutRandom(RandomDevice& random, const Tree& tree, double internalProb) const
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

    Tree SubtreeCrossover::operator()(RandomDevice& random, const Tree& lhs, const Tree& rhs) const
    {
        auto i = CutRandom(random, lhs, internalProbability);

        auto maxBranchDepth  = maxDepth - lhs.Level(i);
        auto maxBranchLength = maxLength - (lhs.Length() - (lhs[i].Length + 1));

        assert(maxBranchDepth < maxDepth);
        assert(maxBranchLength < maxLength);

        auto& left           = lhs.Nodes();
        auto& right          = rhs.Nodes();
        if (auto result = SelectRandomBranch(random, rhs, internalProbability, maxBranchDepth, maxBranchLength); result.has_value())
        {
            auto j = result.value();
            std::vector<Node> nodes;
            copy_n(left.begin(), i - left[i].Length, back_inserter(nodes)); 
            copy_n(right.begin() + j - right[j].Length, right[j].Length + 1, back_inserter(nodes));
            copy_n(left.begin() + i + 1, left.size() - (i + 1), back_inserter(nodes)); 

            return Tree(nodes).UpdateNodes();
        }

        return lhs;
    }
}

