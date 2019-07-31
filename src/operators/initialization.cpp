#include <algorithm>
#include <execution>

#include "core/grammar.hpp"
#include "core/dataset.hpp"
#include "operators/initialization.hpp"

namespace Operon
{
    using namespace std;

    Node SampleProportional(RandomDevice& random, const vector<pair<NodeType, double>>& partials) 
    {
        uniform_real_distribution<double> uniformReal(0, partials.back().second - numeric_limits<double>::epsilon());
        auto r    = uniformReal(random);
        auto it   = find_if(partials.begin(), partials.end(), [=](auto& p) { return p.second > r; });
        auto node = Node(it->first);
        return node;
    }

    void Grow(RandomDevice& random, const Grammar& grammar, const vector<Variable>& variables, vector<Node>& nodes, const vector<pair<NodeType, double>>& partials, size_t maxBranchLength, size_t maxBranchDepth) 
    {
        if (maxBranchDepth == 0 || maxBranchLength <= 1)
        {
            // only allowed to grow leaf nodes
            auto pc   = grammar.GetFrequency(NodeType::Constant);
            auto pv   = grammar.GetFrequency(NodeType::Variable);
            uniform_real_distribution<double> uniformReal(0, pc + pv);
            auto node = uniformReal(random) < pc ? Node(NodeType::Constant) : Node(NodeType::Variable);
            nodes.push_back(node);
        }
        else 
        {
            auto node = SampleProportional(random, partials);
            nodes.push_back(node);
            for (size_t i = 0; i < node.Arity; ++i)
            {
                auto maxChildLength = maxBranchLength / node.Arity;
                Grow(random, grammar, variables, nodes, partials, maxChildLength, maxBranchDepth - 1);
            }
        }
    }

    Tree GrowTreeCreator::operator()(RandomDevice& random, const Grammar& grammar, const vector<Variable>& variables) const
    {
        auto allowed = grammar.AllowedSymbols();
        vector<pair<NodeType, double>> partials(allowed.size());
        std::inclusive_scan(std::execution::seq, allowed.begin(), allowed.end(), partials.begin(), [](const auto& lhs, const auto& rhs) { return std::make_pair(rhs.first, lhs.second + rhs.second); });
        vector<Node> nodes;
        auto root = SampleProportional(random, partials);
        nodes.push_back(root);

        for (int i = 0; i < root.Arity; ++i)
        {
            auto maxBranchLength = (maxLength - 1) / root.Arity;
            Grow(random, grammar, variables, nodes, partials, maxBranchLength, maxDepth - 1);
        }
        uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1); 
        normal_distribution<double> normalReal(0, 1);
        for(auto& node : nodes)
        {
            if (node.IsVariable())
            {
                node.HashValue = node.CalculatedHashValue = variables[uniformInt(random)].Hash;
            }
            if (node.IsLeaf())
            {
                node.Value = normalReal(random);
            }
        }
        reverse(nodes.begin(), nodes.end());
        auto tree = Tree(nodes);
        return tree.UpdateNodes();
    }
}
