#include "operators/crossover.hpp"

namespace Operon
{
    static std::optional<size_t> SelectRandomBranch(operon::rand_t& random, const Tree& tree, double internalProb, size_t maxBranchDepth, size_t maxBranchLength)
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

        auto choice = uniformReal(random) < internalProb;

        for(auto i : indices)
        {
            auto& node = tree[i];
            auto length = node.IsLeaf() ? 1UL : node.Length + 1UL;
            if (length > maxBranchLength || node.Depth > maxBranchDepth)
            {
                continue;
            }

            if (choice != node.IsLeaf())
            {
                return std::make_optional(i);
            }
        }
        return std::nullopt;
    }

    Tree SubtreeCrossover::operator()(operon::rand_t& random, const Tree& lhs, const Tree& rhs) const
    {
        if (auto cut = SelectRandomBranch(random, lhs, internalProbability, maxDepth, maxLength); cut.has_value())
        {
            auto i = cut.value();

            long maxBranchDepth    = maxDepth - lhs.Level(i);
            long partialTreeLength = (lhs.Length() - (lhs[i].Length + 1));
            long maxBranchLength   = maxLength - partialTreeLength;

            if (auto result = SelectRandomBranch(random, rhs, internalProbability, maxBranchDepth, maxBranchLength); result.has_value())
            {
                auto j      = result.value();
                auto& left  = lhs.Nodes();
                auto& right = rhs.Nodes();
                std::vector<Node> nodes;
                nodes.reserve(right[j].Length - left[i].Length + left.size());
                copy_n(left.begin(),                        i - left[i].Length,    back_inserter(nodes));
                copy_n(right.begin() + j - right[j].Length, right[j].Length + 1,   back_inserter(nodes));
                copy_n(left.begin() + i + 1,                left.size() - (i + 1), back_inserter(nodes));

                auto child = Tree(nodes).UpdateNodes();
                return child;
            }
        }

        return lhs;
    }
}

