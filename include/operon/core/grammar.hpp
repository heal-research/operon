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
            auto type = static_cast<NodeType>(1u << i);
            frequencies[i] = 1;
            auto arity = Node(type).Arity;
            arityLimits[i] = { arity, arity };
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

    // return a vector of enabled symbols
    std::vector<NodeType> EnabledSymbols() const
    {
        std::vector<NodeType> allowed;
        for (size_t i = 0; i < frequencies.size(); ++i) {
            if (auto f = frequencies[i]; f > 0) {
                allowed.push_back(static_cast<NodeType>(1u << i));
            }
        }
        return allowed;
    };

    std::pair<size_t, size_t> FunctionArityLimits() const
    {
        size_t minArity = std::numeric_limits<size_t>::max();
        size_t maxArity = std::numeric_limits<size_t>::min();

        for (size_t i = 0; i < arityLimits.size(); ++i) {
            auto type = static_cast<NodeType>(1u << i);

            if (type == NodeType::Constant || type == NodeType::Variable)
                continue;

            if (IsEnabled(type)) {
                auto [amin, amax] = arityLimits[i];

                minArity = std::min(minArity, amin);
                maxArity = std::max(maxArity, amax);
            }
        }
        EXPECT(minArity <= maxArity);

        return { minArity, maxArity };
    }

    Node SampleRandomSymbol(Operon::RandomGenerator& random, size_t minArity, size_t maxArity) const;

    void SetMinimumArity(NodeType type, size_t minArity) {
        EXPECT(minArity <= GetMaximumArity(type));
        std::get<0>(arityLimits[NodeTypes::GetIndex(type)]) = minArity; 
    }

    size_t GetMinimumArity(NodeType type) const noexcept {
        return std::get<0>(arityLimits[NodeTypes::GetIndex(type)]);
    }

    void SetMaximumArity(NodeType type, size_t maxArity) {
        EXPECT(maxArity >= GetMinimumArity(type));
        std::get<1>(arityLimits[NodeTypes::GetIndex(type)]) = maxArity; 
        ENSURE(std::get<1>(arityLimits[NodeTypes::GetIndex(type)]) == maxArity);
    }

    size_t GetMaximumArity(NodeType type) const noexcept {
        return std::get<1>(arityLimits[NodeTypes::GetIndex(type)]);
    }

private:
    NodeType config = Grammar::Arithmetic;
    std::array<size_t, Operon::NodeTypes::Count> frequencies;
    std::array<std::tuple<size_t, size_t>, Operon::NodeTypes::Count> arityLimits;
};

}
#endif
