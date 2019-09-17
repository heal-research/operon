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
        auto choice = uniformReal(random) < internalProb;
        const auto& nodes = tree.Nodes();
        // create a vector of indices and shuffle it to ensure fair sampling
        std::vector<gsl::index> indices(nodes.size());
        size_t leafs = 0;
        for (size_t i = 0; i < indices.size(); ++i)
        {
            indices[i] = i;
            if (nodes[i].IsLeaf()) ++leafs;
        }

        std::sort(indices.begin(), indices.end(), [&](auto i, auto j) { return nodes[i].Arity < nodes[j].Arity; } );

        if (choice)
        {
            std::shuffle(indices.begin() + leafs, indices.end(), random);
            for(size_t i = leafs; i < indices.size(); ++i)
            {
                auto        idx  = indices[i];
                const auto& node = nodes[idx];

                if (node.Length + 1u > maxBranchLength || node.Depth > maxBranchDepth)
                {
                    continue;
                }

                return std::make_optional(idx);
            }
        }
        std::uniform_int_distribution<size_t> uniformInt(0, leafs-1);
        return std::make_optional(indices[uniformInt(random)]);
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

