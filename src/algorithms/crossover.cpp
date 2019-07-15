#include "operators.hpp"

using namespace std;

namespace Operon
{

    class SubtreeCrossover : public CrossoverBase 
    {
        public:
            SubtreeCrossover(double p, size_t d, size_t l) : internalProbability(p), maxDepth(d), maxLength(l) { } 

            auto operator()(Rand& random, const Tree& lhs, const Tree& rhs) const -> Tree override
            {

                auto i = CutRandom(random, lhs, internalProbability);

                auto maxBranchDepth  = maxDepth - lhs.Level(i);
                auto maxBranchLength = maxLength - (lhs.Length() - (lhs[i].Length + 1));

                auto& left           = lhs.Nodes();
                auto& right          = rhs.Nodes();
                if (auto result = SelectRandomBranch(random, rhs, internalProbability, maxBranchLength, maxBranchDepth); result.has_value())
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

            size_t CutRandom(Rand& random, const Tree& tree, double internalProb) const
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

            optional<size_t> SelectRandomBranch(Rand& random, const Tree& tree, double internalProb, size_t maxLength, size_t maxDepth) const
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

        private:
            double internalProbability;
            size_t maxDepth;
            size_t maxLength;
    };
}

