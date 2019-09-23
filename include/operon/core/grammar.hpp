#ifndef GRAMMAR_HPP
#define GRAMMAR_HPP

#include <algorithm>
#include <bitset>
#include <execution>
#include <unordered_map>

#include "jsf.hpp"
#include "tree.hpp"

namespace Operon {
using GrammarConfig = NodeType;

class Grammar {
public:
    Grammar()
    {
        config = Grammar::Arithmetic;

        for (size_t i = 0; i < Operon::NodeTypes::Count; ++i) {
            if (!IsEnabled(static_cast<NodeType>(1u << i))) {
                frequencies[i] = 0;
            }
        }
    }

    bool IsEnabled(NodeType type) const { return static_cast<bool>(config & type); }
    void Enable(NodeType type, size_t freq)
    {
        config |= type;
        frequencies[NodeTypes::GetIndex(type)] = freq;
    }
    void Disable(NodeType type)
    {
        config &= ~type;
        frequencies[NodeTypes::GetIndex(type)] = 0;
    }
    GrammarConfig GetConfig() const { return config; }
    void SetConfig(GrammarConfig cfg)
    {
        config = cfg;
        for (auto i = 0u; i < frequencies.size(); ++i) {
            auto type = static_cast<NodeType>(1u << i);
            if (IsEnabled(type)) {
                if (frequencies[i] == 0) {
                    frequencies[i] = 1;
                }
            } else {
                frequencies[i] = 0;
            }
        }
    }
    size_t GetFrequency(NodeType type) const { return frequencies[NodeTypes::GetIndex(type)]; }

    static const GrammarConfig Arithmetic = NodeType::Constant | NodeType::Variable | NodeType::Add | NodeType::Sub | NodeType::Mul | NodeType::Div;
    static const GrammarConfig TypeCoherent = Arithmetic | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Square;
    static const GrammarConfig Full = TypeCoherent | NodeType::Tan | NodeType::Sqrt | NodeType::Cbrt;

    std::vector<std::pair<NodeType, size_t>> EnabledSymbols() const
    {
        std::vector<std::pair<NodeType, size_t>> allowed;
        for (size_t i = 0; i < frequencies.size(); ++i) {
            if (auto f = frequencies[i]; f > 0) {
                allowed.push_back({ static_cast<NodeType>(1u << i), f });
            }
        }
        return allowed;
    };

    std::pair<size_t, size_t> FunctionArityLimits() const
    {
        size_t minArity = std::numeric_limits<size_t>::max();
        size_t maxArity = std::numeric_limits<size_t>::min();
        for (size_t i = 0; i < frequencies.size() - 2; ++i) {
            if (frequencies[i] == 0) {
                continue;
            }
            size_t arity = i < 4 ? 2 : 1;
            minArity = std::min(minArity, arity);
            maxArity = std::max(maxArity, arity);
        }
        return { minArity, maxArity };
    }

    Node SampleRandomSymbol(operon::rand_t& random, size_t minArity = 0, size_t maxArity = 2) const
    {
        decltype(frequencies)::const_iterator head = frequencies.end();
        decltype(frequencies)::const_iterator tail = frequencies.end();

        if (minArity == 0) {
            tail = frequencies.end();
        } else if (minArity == 1) {
            tail = frequencies.end() - 2;
        } else {
            tail = frequencies.begin() + 4;
        }

        if (maxArity == 0) {
            head = frequencies.end() - 2;
        } else if (maxArity == 1) {
            head = frequencies.begin() + 4;
        } else {
            head = frequencies.begin();
        }

        auto d = std::distance(frequencies.begin(), head);
        //fmt::print("minArity = {}, maxArity = {}, d = {}\n", d, minArity, maxArity);
        auto i = std::discrete_distribution<size_t>(head, tail)(random) + d;
        return Node(static_cast<NodeType>(1u << i));
    }

private:
    NodeType config = Grammar::Arithmetic;
    std::array<size_t, Operon::NodeTypes::Count> frequencies = { 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1 };
};

}
#endif
