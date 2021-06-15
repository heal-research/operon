// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PSET_HPP 
#define OPERON_PSET_HPP

#include <algorithm>
#include <bitset>
#include <unordered_map>

#include "core/contracts.hpp"
#include "core/tree.hpp"

namespace Operon {
using PrimitiveSetConfig = NodeType;

class PrimitiveSet {
public:
    PrimitiveSet()
    {
        config = PrimitiveSet::Arithmetic;

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
    PrimitiveSetConfig GetConfig() const { return config; }
    void SetConfig(PrimitiveSetConfig cfg)
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

    static const PrimitiveSetConfig Arithmetic = NodeType::Constant | NodeType::Variable | NodeType::Add | NodeType::Sub | NodeType::Mul | NodeType::Div;
    static const PrimitiveSetConfig TypeCoherent = Arithmetic | NodeType::Pow | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Square;
    static const PrimitiveSetConfig Full = TypeCoherent | NodeType::Aq | NodeType::Tan | NodeType::Tanh | NodeType::Sqrt | NodeType::Cbrt;

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

    std::tuple<size_t, size_t> GetMinMaxArity(NodeType type) const noexcept {
        return arityLimits[NodeTypes::GetIndex(type)];
    }

    void SetMinMaxArity(NodeType type, size_t minArity, size_t maxArity) noexcept {
        EXPECT(maxArity >= minArity);
        arityLimits[NodeTypes::GetIndex(type)] = { minArity, maxArity };
    }

private:
    NodeType config = PrimitiveSet::Arithmetic;
    std::array<size_t, Operon::NodeTypes::Count> frequencies;
    std::array<std::tuple<size_t, size_t>, Operon::NodeTypes::Count> arityLimits;
};

}
#endif
