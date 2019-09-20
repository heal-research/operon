#include "operators/crossover.hpp"

namespace Operon {
static gsl::index SelectRandomBranch(operon::rand_t& random, const Tree& tree, double internalProb, size_t maxBranchDepth, size_t maxBranchLength)
{
    std::uniform_real_distribution<double> uniformReal(0, 1);
    auto selectInternalNode = uniformReal(random) < internalProb;
    const auto& nodes = tree.Nodes();
    // create a vector of indices where leafs are in the front and internal nodes in the back
    std::vector<gsl::index> indices(nodes.size());
    size_t head = 0;
    size_t tail = nodes.size() - 1;
    for (size_t i = 0; i < nodes.size(); ++i) {
        const auto& node = nodes[i];
        auto idx = node.IsLeaf() ? head++ : tail--;
        indices[idx] = i;
    }
    // if we want an internal node, we shuffle the corresponding part of the indices vector
    // then we walk over it and return the first index that satisfies the constraints
    if (selectInternalNode) {
        std::shuffle(indices.begin() + head, indices.end(), random);
        for (size_t i = head; i < indices.size(); ++i) {
            auto idx = indices[i];
            const auto& node = nodes[idx];

            if (node.Length + 1u > maxBranchLength || node.Depth > maxBranchDepth) {
                continue;
            }

            return idx;
        }
    }
    // if we couldn't find a suitable internal node or just wanted a leaf, fallback here
    std::uniform_int_distribution<size_t> uniformInt(0, head - 1);
    auto idx = uniformInt(random);
    return indices[idx];
}

Tree SubtreeCrossover::operator()(operon::rand_t& random, const Tree& lhs, const Tree& rhs) const
{
    if (auto i = SelectRandomBranch(random, lhs, internalProbability, maxDepth, maxLength)) {
        long maxBranchDepth = maxDepth - lhs.Level(i);
        long partialTreeLength = (lhs.Length() - (lhs[i].Length + 1));
        long maxBranchLength = maxLength - partialTreeLength;

        if (auto j = SelectRandomBranch(random, rhs, internalProbability, maxBranchDepth, maxBranchLength)) {
            auto& left = lhs.Nodes();
            auto& right = rhs.Nodes();
            std::vector<Node> nodes;
            nodes.reserve(right[j].Length - left[i].Length + left.size());
            copy_n(left.begin(), i - left[i].Length, back_inserter(nodes));
            copy_n(right.begin() + j - right[j].Length, right[j].Length + 1, back_inserter(nodes));
            copy_n(left.begin() + i + 1, left.size() - (i + 1), back_inserter(nodes));

            auto child = Tree(nodes).UpdateNodes();
            return child;
        }
    }
    return lhs;
}
}
