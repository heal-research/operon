#ifndef GRAMMAR_HPP
#define GRAMMAR_HPP

#include <unordered_map>
#include <algorithm>
#include <stack>

#include "tree.hpp"
#include "jsf.hpp"

namespace Operon 
{
    using Rand = Random::JsfRand<64>;

    class Grammar
    {
        public:
            bool   IsEnabled(NodeType type) const { return static_cast<bool>(config & type); }
            void   SetEnabled(NodeType type, bool enabled) { config = enabled ? (config | type) : (config & ~type); }
            void   SetFrequency(NodeType type, double frequency) { symbolFrequencies[type] = frequency; };
            double GetFrequency(NodeType type) const { return symbolFrequencies.find(type)->second; }

            static const NodeType Arithmetic   = NodeType::Constant | NodeType::Variable | NodeType::Add  | NodeType::Sub | NodeType::Mul | NodeType::Div;
            static const NodeType TypeCoherent = Arithmetic         | NodeType::Exp      | NodeType::Log  | NodeType::Sin | NodeType::Cos;
            static const NodeType Full         = TypeCoherent       | NodeType::Tan      | NodeType::Sqrt | NodeType::Cbrt;

            std::vector<std::pair<NodeType, double>> AllowedSymbols() const 
            { 
                std::vector<std::pair<NodeType, double>> allowed;
                std::copy_if(symbolFrequencies.begin(), symbolFrequencies.end(), std::back_inserter(allowed), [&](auto& t){ return IsEnabled(t.first); } );
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

//                std::generate(population.begin(), population.end(), [&]() { return GrowTree(random, partials, maxLength, maxDepth); });
                return population;
            }

        private:
            NodeType config = Grammar::Arithmetic;
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

