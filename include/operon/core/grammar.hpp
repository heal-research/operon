/* This file is part of:
 * Operon - Large Scale Genetic Programming Framework
 *
 * Licensed under the ISC License <https://opensource.org/licenses/ISC> 
 * Copyright (C) 2019 Bogdan Burlacu 
 *
 * Permission to use, copy, modify, and/or distribute this software for any
 * purpose with or without fee is hereby granted, provided that the above
 * copyright notice and this permission notice appear in all copies.
 *
 * THE SOFTWARE IS PROVIDED "AS IS" AND THE AUTHOR DISCLAIMS ALL WARRANTIES WITH
 * REGARD TO THIS SOFTWARE INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY AND
 * FITNESS. IN NO EVENT SHALL THE AUTHOR BE LIABLE FOR ANY SPECIAL, DIRECT,
 * INDIRECT, OR CONSEQUENTIAL DAMAGES OR ANY DAMAGES WHATSOEVER RESULTING FROM
 * LOSS OF USE, DATA OR PROFITS, WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR
 * OTHER TORTIOUS ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR
 * PERFORMANCE OF THIS SOFTWARE. 
 */

#ifndef GRAMMAR_HPP
#define GRAMMAR_HPP

#include <algorithm>
#include <bitset>
#include <execution>
#include <unordered_map>

#include "core/common.hpp"
#include "core/tree.hpp"

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

    Node SampleRandomSymbol(Operon::Random& random, size_t minArity = 0, size_t maxArity = 2) const
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
