// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_PSET_HPP
#define OPERON_PSET_HPP

#include <algorithm>

#include "core/contracts.hpp"
#include "core/tree.hpp"

#include "robin_hood.h"

namespace Operon {
using PrimitiveSetConfig = NodeType;

namespace {
    enum { NODE = 0,
        FREQUENCY = 1,
        MIN_ARITY = 2,
        MAX_ARITY = 3 };
}

class PrimitiveSet {
private:
    using Primitive = std::tuple<
        Node,
        size_t, // 1: frequency
        size_t, // 2: min arity
        size_t  // 3: max arity
        >;
    robin_hood::unordered_flat_map<Operon::Hash, Primitive> _pset;

    Primitive const& GetPrimitive(Operon::Hash hash) const {
        auto it = _pset.find(hash);
        if (it == _pset.end()) {
            throw std::runtime_error(fmt::format("Unknown node hash {}\n", hash));
        }
        return it->second;
    }

    Primitive& GetPrimitive(Operon::Hash hash) {
        return const_cast<Primitive&>(const_cast<PrimitiveSet const*>(this)->GetPrimitive(hash));
    }

public:
    static const PrimitiveSetConfig Arithmetic = NodeType::Constant | NodeType::Variable | NodeType::Add | NodeType::Sub | NodeType::Mul | NodeType::Div;
    static const PrimitiveSetConfig TypeCoherent = Arithmetic | NodeType::Pow | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Square;
    static const PrimitiveSetConfig Full = TypeCoherent | NodeType::Aq | NodeType::Fmin | NodeType::Fmax | NodeType::Tan |       
                                           NodeType::Abs | NodeType::Ceil | NodeType::Floor |
                                           NodeType::Erf | NodeType::Erfc |
                                           NodeType::Log1p |
                                           NodeType::Asin | NodeType::Acos | NodeType::Atan |
                                           NodeType::Asin | NodeType::Acos | NodeType::Atan |
                                           NodeType::Sinh | NodeType::Cosh | NodeType::Tanh |
                                           NodeType::Sqrt | NodeType::Cbrt;

    PrimitiveSet() {}

    PrimitiveSet(PrimitiveSetConfig config)
    {
        SetConfig(config);
    }

    decltype(_pset) const& Primitives() const { return _pset; }

    bool AddPrimitive(Operon::Node node, size_t frequency, size_t min_arity, size_t max_arity)
    {
        auto [_, ok] = _pset.insert({ node.HashValue, Primitive { node, frequency, min_arity, max_arity } });
        return ok;
    }
    void RemovePrimitive(Operon::Node node) { _pset.erase(node.HashValue); }

    void RemovePrimitive(Operon::Hash hash) { _pset.erase(hash); }

    void SetConfig(PrimitiveSetConfig config)
    {
        _pset.clear();
        for (size_t i = 0; i < Operon::NodeTypes::Count; ++i) {
            NodeType t = static_cast<Operon::NodeType>(1u << i);
            Operon::Node n(t);

            if ((1u << i) & (uint32_t)config) {
                _pset[n.HashValue] = { n, 1, n.Arity, n.Arity };
            }
        }
    }

    std::vector<Node> EnabledPrimitives() const {
        std::vector<Node> nodes;
        for (auto const& [k, v] : _pset) {
            auto [node, freq, min_arity, max_arity] = v;
            if (node.IsEnabled && freq > 0) {
                nodes.push_back(node);
            }
        }
        return nodes;
    }

    PrimitiveSetConfig GetConfig() const
    {
        PrimitiveSetConfig conf { static_cast<PrimitiveSetConfig>(0) };
        for (auto [k, v] : _pset) {
            auto const& [node, freq, min_arity, max_arity] = v;
            if (node.IsEnabled && freq > 0) {
                conf |= node.Type;
            }
        }
        return conf;
    }

    size_t GetFrequency(Operon::Hash hash) const
    {
        auto const& p = GetPrimitive(hash);
        return std::get<FREQUENCY>(p);
    }

    void SetFrequency(Operon::Hash hash, size_t frequency)
    {
        auto& p = GetPrimitive(hash);
        std::get<FREQUENCY>(p) = frequency;
    }

    bool Contains(Operon::Hash hash) const { return _pset.contains(hash); }

    bool IsEnabled(Operon::Hash hash) const
    {
        auto const& p = GetPrimitive(hash);
        return std::get<NODE>(p).IsEnabled;
    }

    void SetEnabled(Operon::Hash hash, bool enabled)
    {
        auto& p = GetPrimitive(hash);
        std::get<NODE>(p).IsEnabled = enabled;
    }

    void Enable(Operon::Hash hash)
    {
        SetEnabled(hash, true);
    }

    void Disable(Operon::Hash hash)
    {
        SetEnabled(hash, false);
    }

    std::pair<size_t, size_t> FunctionArityLimits() const
    {
        auto min_arity = std::numeric_limits<size_t>::max();
        auto max_arity = std::numeric_limits<size_t>::min();
        for (auto const& [key, val] : _pset) {
            min_arity = std::min(min_arity, std::get<MIN_ARITY>(val));
            max_arity = std::max(max_arity, std::get<MAX_ARITY>(val));
        }
        return { min_arity, max_arity };
    }

    Operon::Node SampleRandomSymbol(Operon::RandomGenerator& random, size_t minArity, size_t maxArity) const;

    void SetMinimumArity(Operon::Hash hash, size_t minArity)
    {
        EXPECT(minArity <= GetMaximumArity(hash));
        auto& p = GetPrimitive(hash);
        std::get<MIN_ARITY>(p) = minArity;
    }

    size_t GetMinimumArity(Operon::Hash hash) const
    {
        auto const& p = GetPrimitive(hash);
        return std::get<MIN_ARITY>(p);
    }

    void SetMaximumArity(Operon::Hash hash, size_t maxArity)
    {
        EXPECT(maxArity >= GetMinimumArity(hash));
        auto& p = GetPrimitive(hash);
        std::get<MAX_ARITY>(p) = maxArity;
    }

    size_t GetMaximumArity(Operon::Hash hash) const
    {
        auto const& p = GetPrimitive(hash);
        return std::get<MAX_ARITY>(p);
    }

    std::tuple<size_t, size_t> GetMinMaxArity(Operon::Hash hash) const
    {
        auto const& p = GetPrimitive(hash);
        return { std::get<MIN_ARITY>(p), std::get<MAX_ARITY>(p) };
    }

    void SetMinMaxArity(Operon::Hash hash, size_t minArity, size_t maxArity)
    {
        EXPECT(maxArity >= minArity);
        auto& p = GetPrimitive(hash);
        std::get<MIN_ARITY>(p) = minArity;
        std::get<MAX_ARITY>(p) = maxArity;
    }

    // convenience overloads
    void SetFrequency(Operon::Node node, size_t frequency) { SetFrequency(node.HashValue, frequency); }
    size_t GetFrequency(Operon::Node node) const { return GetFrequency(node.HashValue); }

    bool Contains(Operon::Node node) const { return Contains(node.HashValue); }
    bool IsEnabled(Operon::Node node) const { return IsEnabled(node.HashValue); }

    void SetEnabled(Operon::Node node, bool enabled) { SetEnabled(node.HashValue, enabled); }
    void Enable(Operon::Node node) { SetEnabled(node, true); }
    void Disable(Operon::Node node) { SetEnabled(node, false); }

    void SetMinimumArity(Operon::Node node, size_t minArity)
    {
        SetMinimumArity(node.HashValue, minArity);
    }
    size_t GetMinimumArity(Operon::Node node) const { return GetMinimumArity(node.HashValue); }

    void SetMaximumArity(Operon::Node node, size_t minArity)
    {
        SetMaximumArity(node.HashValue, minArity);
    }
    size_t GetMaximumArity(Operon::Node node) const { return GetMaximumArity(node.HashValue); }

    std::tuple<size_t, size_t> GetMinMaxArity(Operon::Node node) const
    {
        return GetMinMaxArity(node.HashValue);
    }
    void SetMinMaxArity(Operon::Node node, size_t minArity, size_t maxArity)
    {
        SetMinMaxArity(node.HashValue, minArity, maxArity);
    }
};
} // namespace Operon
#endif
