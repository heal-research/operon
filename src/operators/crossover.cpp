#include "operators/crossover.hpp"

namespace Operon
{
    std::optional<size_t> SubtreeCrossover::SelectRandomBranch(operon::rand_t& random, const Tree& tree, double internalProb, size_t maxBranchDepth, size_t maxBranchLength) const
    {
        if (maxBranchDepth == 0 || maxBranchLength == 0)
        {
            return std::nullopt;
        }
        std::uniform_real_distribution<double> uniformReal(0, 1);
        // create a vector of indices and shuffle it to ensure fair sampling
        std::vector<gsl::index> indices(tree.Length());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), random);

        for(auto i : indices)
        {
            auto& node = tree[i];
            auto length = node.IsLeaf() ? 1UL : node.Length + 1UL;
            if (length > maxBranchLength || tree.Depth(i) > maxBranchDepth)
            {
                continue;
            }

            if ((uniformReal(random) < internalProb) != tree[i].IsLeaf())
            {
                return std::make_optional(i);
            }
        }
        return std::nullopt;
    }

    size_t SubtreeCrossover::CutRandom(operon::rand_t& random, const Tree& tree, double internalProb) const
    {
        std::uniform_real_distribution<double> uniformReal(0, 1);
        // create a vector of indices and shuffle it to ensure fair sampling
        std::vector<size_t> indices(tree.Length());
        iota(indices.begin(), indices.end(), 0);
        shuffle(indices.begin(), indices.end(), random);

        for (auto i : indices)
        {
            if ((uniformReal(random) < internalProb) != tree[i].IsLeaf())
            {
                return i; 
            }
        }
        return indices.back();
    }

    Tree SubtreeCrossover::operator()(operon::rand_t& random, const Tree& lhs, const Tree& rhs) const
    {
        auto i = CutRandom(random, lhs, internalProbability);

        long maxBranchDepth  = maxDepth - lhs.Level(i);
        long partialTreeLength = (lhs.Length() - (lhs[i].Length + 1));
        long maxBranchLength = maxLength - partialTreeLength;

        if (auto result = SelectRandomBranch(random, rhs, internalProbability, maxBranchDepth, maxBranchLength); result.has_value())
        {
            auto j = result.value();
            std::vector<Node> nodes;
            auto& left           = lhs.Nodes();
            auto& right          = rhs.Nodes();
            copy_n(left.begin(),                        i - left[i].Length,    back_inserter(nodes));
            copy_n(right.begin() + j - right[j].Length, right[j].Length + 1,   back_inserter(nodes));
            copy_n(left.begin() + i + 1,                left.size() - (i + 1), back_inserter(nodes));

            auto child = Tree(nodes).UpdateNodes();
            return child;
        }

        return lhs;
    }
}

