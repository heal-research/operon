#ifndef GRAMMAR_HPP
#define GRAMMAR_HPP

#include <unordered_map>
#include <algorithm>

#include "tree.hpp"
#include "jsf.hpp"

namespace Operon 
{
    using GrammarConfig = NodeType;

    class Grammar
    {
        public:
            bool          IsEnabled(NodeType type) const { return static_cast<bool>(config & type); }
            void          SetEnabled(NodeType type, bool enabled) { config = enabled ? (config | type) : (config & ~type); }
            void          SetFrequency(NodeType type, double frequency) { symbolFrequencies[type] = frequency; };
            double        GetFrequency(NodeType type) const { return symbolFrequencies.find(type)->second; }
            GrammarConfig GetConfig() const { return config; }
            void          SetConfig(GrammarConfig cfg) { config = cfg; }

            static const GrammarConfig Arithmetic   = NodeType::Constant | NodeType::Variable | NodeType::Add  | NodeType::Sub | NodeType::Mul | NodeType::Div;
            static const GrammarConfig TypeCoherent = Arithmetic         | NodeType::Exp      | NodeType::Log  | NodeType::Sin | NodeType::Cos | NodeType::Square;
            static const GrammarConfig Full         = TypeCoherent       | NodeType::Tan      | NodeType::Sqrt | NodeType::Cbrt;

            std::vector<std::pair<NodeType, double>> AllowedSymbols() const 
            { 
                std::vector<std::pair<NodeType, double>> allowed;
                std::copy_if(symbolFrequencies.begin(), symbolFrequencies.end(), std::back_inserter(allowed), [&](auto& t){ return IsEnabled(t.first); } );
                return allowed;
            };

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
                { NodeType::Square,   1.0 },
                { NodeType::Constant, 1.0 },
                { NodeType::Variable, 1.0 },
            };
    };

}
#endif

