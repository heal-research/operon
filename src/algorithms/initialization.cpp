#include "operators.hpp"
#include "grammar.hpp"
#include "dataset.hpp"

namespace Operon 
{

    using T = std::pair<NodeType, double>;
    class GrowTreeCreator : public OperatorBase<Tree>
    {
        public:
            Tree operator()(Rand& random, const Grammar& grammar, const std::vector<Variable>& variables)
            {
                auto allowed = grammar.AllowedSymbols();
                auto partials = std::vector<T>(allowed.size());
                for (size_t i = 1; i < partials.size(); ++i)
                {
                    partials[i].second += partials[i-1].second;
                }
                std::vector<Node> nodes;
                auto node = SampleProportional(random, partials);
                nodes.push_back(node);

                for (int i = 0; i < node.Arity; ++i)
                {
                    Grow(random, grammar, variables, nodes, partials, maxLength, maxDepth - 1);
                }
                std::reverse(nodes.begin(), nodes.end());
                auto tree = Tree(nodes);
                return tree.UpdateNodes();
            }

        private:
            void Grow(Rand& random, const Grammar& grammar, const std::vector<Variable>& variables, std::vector<Node>& nodes, const std::vector<T>& partials, size_t maxLength, size_t maxDepth)
            {
                if (maxDepth == 0)
                {
                    // only allowed to grow leaf nodes
                    auto pc   = grammar.GetFrequency(NodeType::Constant);
                    auto pv   = grammar.GetFrequency(NodeType::Variable);
                    std::uniform_real_distribution<double> uniformReal(0, pc + pv);
                    auto node = Node(NodeType::Constant);
                    if (uniformReal(random) > pc)
                    {
                        node = Node(NodeType::Variable);
                        // currently each variable is considered equally probable
                        std::uniform_int_distribution<size_t> uniformInt(0, variables.size() - 1); 
                        node.HashValue = variables[uniformInt(random)].Hash;
                    }
                    nodes.push_back(node);
                }
                else 
                {
                    auto node = SampleProportional(random, partials);
                    nodes.push_back(node);
                    for (size_t i = 0; i < node.Arity; ++i)
                    {
                        Grow(random, grammar, variables, nodes, partials, maxLength, maxDepth - 1);
                    }
                }
            }

            Node SampleProportional(Rand& random, const std::vector<std::pair<NodeType, double>>& partials)
            {
                std::uniform_real_distribution<double> uniformReal(0, partials.back().second - std::numeric_limits<double>::epsilon());
                auto r    = uniformReal(random);
                auto it   = std::find_if(partials.begin(), partials.end(), [=](auto& p) { return p.second > r; });
                auto node = Node(it->first);
                return node;
            }

            size_t maxDepth;
            size_t maxLength;
    };

}
