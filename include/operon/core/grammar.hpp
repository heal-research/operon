/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Copyright (C) 2019 Bogdan Burlacu 
 * 
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU Affero General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 * 
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Affero General Public License for more details.
 * 
 * You should have received a copy of the GNU Affero General Public License
 * along with this program.  If not, see <https://www.gnu.org/licenses/>.
 * SOFTWARE.
 */

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

        frequencies.fill(0);
        for (size_t i = 0; i < Operon::NodeTypes::Count; ++i) {
            if (IsEnabled(static_cast<NodeType>(1u << i))) {
                frequencies[i] = 1;
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

        Expects(maxArity <= 2);
        Expects(minArity <= maxArity);

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

        if (std::all_of(head, tail, [](size_t v) { return v == 0; })) {
            if (minArity == 0 && maxArity == 2) {
                throw new std::runtime_error(fmt::format("Could not sample any symbol as all frequencies are set to zero"));
            }
            return SampleRandomSymbol(random, minArity - 1, maxArity);
        }
        auto d = std::distance(frequencies.begin(), head);
        auto i = std::discrete_distribution<size_t>(head, tail)(random) + d;
        auto node = Node(static_cast<NodeType>(1u << i));
        Ensures(IsEnabled(node.Type));

        return node;
    }

private:
    NodeType config = Grammar::Arithmetic;
    std::array<size_t, Operon::NodeTypes::Count> frequencies;
};

}
#endif
