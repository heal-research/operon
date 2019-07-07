#ifndef GRAMMAR_HPP
#define GRAMMAR_HPP

#include <unordered_map>
#include <algorithm>
#include <execution>
#include <stack>

#include "tree.hpp"
#include "jsf.hpp"

namespace Operon 
{
    using Rand = Random::JsfRand<64>;

    class Grammar
    {
        public:
            bool IsEnabled(NodeType type) const { return config & type; }
            void SetEnabled(NodeType type, bool enabled) { config = enabled ? (config | type) : (config & ~type); }
            void SetFrequency(NodeType type, double frequency) { symbolFrequencies[type] = frequency; };
            double GetFrequency(NodeType type) const { return symbolFrequencies.find(type)->second; }

            static const uint16_t Arithmetic   = NodeType::Constant | NodeType::Variable | NodeType::Add  | NodeType::Sub | NodeType::Mul | NodeType::Div;
            static const uint16_t TypeCoherent = Arithmetic         | NodeType::Exp      | NodeType::Log  | NodeType::Sin | NodeType::Cos;
            static const uint16_t Full         = TypeCoherent       | NodeType::Tan      | NodeType::Sqrt | NodeType::Cbrt;

            std::vector<std::pair<NodeType, double>> AllowedSymbols() const { 
                std::vector<std::pair<NodeType, double>> allowed;
                for (auto& t : symbolFrequencies)
                {
                    if (IsEnabled(t.first))
                    {
                        allowed.push_back(t);
                    }
                }
                return allowed;
            };

            std::vector<Tree> Initialize(Rand& random, size_t size, size_t maxLength, size_t maxDepth)
            {
                std::vector<Tree> population(size);
                auto allowed = AllowedSymbols();
                auto partials = allowed;
                for (size_t i = 1; i < partials.size(); ++i)
                {
                    partials[i].second += partials[i-1].second;
                }

                std::generate(population.begin(), population.end(), [&]() { return GrowTree(random, partials, maxLength, maxDepth); });
                return population;
            }

        private:
            Tree GrowTree(Rand& random, const std::vector<std::pair<NodeType, double>>& partials, size_t maxLength, size_t maxDepth)
            {
                using T = std::pair<NodeType, double>;
                std::vector<Node> nodes;

                auto node = SampleProportional(random, partials);
                nodes.push_back(node);

                for (int i = 0; i < node.Arity; ++i)
                {
                    Grow(random, nodes, partials, maxLength, maxDepth - 1);
                }

                auto tree = Tree(nodes);
                //return tree.UpdateNodes();
                return tree;
            }

            void Grow(Rand& random, std::vector<Node>& nodes, const std::vector<std::pair<NodeType, double>> partials, size_t maxLength, size_t maxDepth)
            {
                if (maxDepth == 0)
                {
                    // only allowed to grow leaf nodes
                    auto pConst = GetFrequency(NodeType::Constant);
                    auto pVariable = GetFrequency(NodeType::Variable);
                    std::uniform_real_distribution<double> uniformReal(0, pConst + pVariable);
                    auto node = uniformReal(random) < pConst ? Node(NodeType::Constant) : Node(NodeType::Variable);                    
                    nodes.push_back(node);
                }
                else 
                {
                    auto node = SampleProportional(random, partials);
                    nodes.push_back(node);
                    //fmt::print("{} arity: {}\n", node.Name(), node.Arity);
                    for (size_t i = 0; i < node.Arity; ++i)
                    {
                        Grow(random, nodes, partials, maxLength, maxDepth - 1);
                    }
                }
            }

            Node SampleProportional(Rand& random, const std::vector<std::pair<NodeType, double>>& partials)
            {
                std::uniform_real_distribution<double> uniformReal(0, partials.back().second - std::numeric_limits<double>::epsilon());
                auto r = uniformReal(random);
                auto it = std::find_if(partials.begin(), partials.end(), [=](auto& p) { return p.second > r; });
                auto node = Node(it->first);
                return node;
            }

            uint16_t config = Grammar::Arithmetic;
            std::unordered_map<NodeType, double> symbolFrequencies = {
                { NodeType::Add,      1.0 },
                { NodeType::Mul,      1.0 },
                { NodeType::Sub,      1.0 },
                { NodeType::Div,      1.0 },
                { NodeType::Exp,      1.0 },
                { NodeType::Log,      1.0 },
                { NodeType::Sin,      1.0 },
                { NodeType::Cos,      1.0 },
                { NodeType::Tan,      1.0 },
                { NodeType::Sqrt,     1.0 },
                { NodeType::Cbrt,     1.0 },
                { NodeType::Constant, 1.0 },
                { NodeType::Variable, 1.0 },
            };
    };

}
#endif

