#ifndef GRAMMAR_HPP
#define GRAMMAR_HPP

#include <unordered_map>
#include <algorithm>
#include <execution>

#include "tree.hpp"
#include "jsf.hpp"

namespace Operon 
{
    using GrammarConfig = NodeType;

    class Grammar
    {
        public:
            Grammar() 
            {
                config = Grammar::Arithmetic;
                UpdatePartials();
            }

            bool          IsEnabled(NodeType type) const                { return static_cast<bool>(config & type);               }
            void          SetEnabled(NodeType type, bool enabled)       { config = enabled ? (config | type) : (config & ~type); }
            void          SetFrequency(NodeType type, double frequency) { symbolFrequencies[type] = frequency; UpdatePartials(); }
            double        GetFrequency(NodeType type) const             { return symbolFrequencies.find(type)->second;           }
            GrammarConfig GetConfig() const                             { return config;                                         }
            void          SetConfig(GrammarConfig cfg)                  { config = cfg; UpdatePartials();                        }

            static const GrammarConfig Arithmetic   = NodeType::Constant | NodeType::Variable | NodeType::Add  | NodeType::Sub | NodeType::Mul | NodeType::Div;
            static const GrammarConfig TypeCoherent = Arithmetic         | NodeType::Exp      | NodeType::Log  | NodeType::Sin | NodeType::Cos | NodeType::Square;
            static const GrammarConfig Full         = TypeCoherent       | NodeType::Tan      | NodeType::Sqrt | NodeType::Cbrt;

            std::vector<std::pair<NodeType, double>> AllowedSymbols() const 
            { 
                std::vector<std::pair<NodeType, double>> allowed;
                std::copy_if(symbolFrequencies.begin(), symbolFrequencies.end(), std::back_inserter(allowed), [&](auto& t){ return IsEnabled(t.first); } );
                return allowed;
            };

            size_t MinimumFunctionArity() const 
            {
                auto minArity = std::numeric_limits<size_t>::max();
                for(auto& [key, val] : symbolFrequencies)
                {
                    if (!IsEnabled(key) || key > NodeType::Square || val < 1e-6)
                    {
                        continue;
                    }
                    auto arity = key < NodeType::Log ? 2U : 1U;
                    if (minArity > arity) minArity = arity;
                }
                return minArity;
            }

            Node SampleRandomFunction(operon::rand_t& random) const 
            {
                return Sample(random, functionPartials);
            }

            Node SampleRandomTerminal(operon::rand_t& random) const
            {
                return Sample(random, terminalPartials);
            }

            Node SampleRandomSymbol(operon::rand_t& random) const
            {
                return Sample(random, symbolPartials);
            }

            static Node Sample(operon::rand_t& random, const std::vector<std::pair<NodeType, double>>& partials)
            {
                std::uniform_real_distribution<double> uniformReal(0, partials.back().second - std::numeric_limits<double>::epsilon());
                auto r    = uniformReal(random);
                auto it   = find_if(partials.begin(), partials.end(), [=](const auto& p) { return p.second > r; });
                return Node(it->first);
            }

        private:
            NodeType config = Grammar::Arithmetic;
            std::unordered_map<NodeType, double> symbolFrequencies = {
                { NodeType::Add,      1.0 },
                { NodeType::Mul,      1.0 },
                { NodeType::Sub,      1.0 },
                { NodeType::Div,      1.0 },
                { NodeType::Log,      1.0 },
                { NodeType::Exp,      1.0 },
                { NodeType::Sin,      1.0 },
                { NodeType::Cos,      1.0 },
                { NodeType::Tan,      1.0 },
                { NodeType::Sqrt,     1.0 },
                { NodeType::Cbrt,     1.0 },
                { NodeType::Square,   1.0 },
                { NodeType::Constant, 1.0 },
                { NodeType::Variable, 1.0 },
            };
            std::vector<std::pair<NodeType, double>> functionPartials;
            std::vector<std::pair<NodeType, double>> terminalPartials;
            std::vector<std::pair<NodeType, double>> symbolPartials;

            void UpdatePartials()
            {
                using P = std::pair<NodeType, double>;
                std::vector<P> funcs;
                std::copy_if(symbolFrequencies.begin(), symbolFrequencies.end(), std::back_inserter(funcs), [&](auto p) { return IsEnabled(p.first) && p.first < NodeType::Constant; });

                std::vector<P> terms;
                std::copy_if(symbolFrequencies.begin(), symbolFrequencies.end(), std::back_inserter(terms), [&](auto p) { return IsEnabled(p.first) && p.first > NodeType::Square; });

                functionPartials.clear();
                std::inclusive_scan(std::execution::seq, funcs.begin(), funcs.end(), std::back_inserter(functionPartials), [](const auto& lhs, const auto& rhs) { return std::make_pair(rhs.first, lhs.second + rhs.second); });

                terminalPartials.clear();
                std::inclusive_scan(std::execution::seq, terms.begin(), terms.end(), std::back_inserter(terminalPartials), [](const auto& lhs, const auto& rhs) { return std::make_pair(rhs.first, lhs.second + rhs.second); });

                symbolPartials.clear();
                auto allowed = AllowedSymbols();
                std::inclusive_scan(std::execution::seq, allowed.begin(), allowed.end(), std::back_inserter(symbolPartials), [](const auto& lhs, const auto& rhs) { return std::make_pair(rhs.first, lhs.second + rhs.second); });
            }
    };

}
#endif

