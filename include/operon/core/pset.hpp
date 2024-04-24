// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_PSET_HPP
#define OPERON_PSET_HPP

#include "contracts.hpp"
#include "node.hpp"

namespace Operon {

class PrimitiveSet {
    using Primitive = std::tuple<
        Node,
        size_t, // 1: frequency
        size_t, // 2: min arity
        size_t  // 3: max arity
        >;
    enum { NODE = 0, FREQUENCY = 1, MINARITY = 2, MAXARITY = 3}; // for accessing tuple elements more easily

    Operon::Map<Operon::Hash, Primitive> pset_;

    [[nodiscard]] auto OPERON_EXPORT GetPrimitive(Operon::Hash hash) const -> Primitive const&;

    auto GetPrimitive(Operon::Hash hash) -> Primitive& {
        return const_cast<Primitive&>(const_cast<PrimitiveSet const*>(this)->GetPrimitive(hash)); // NOLINT
    }

public:
    static constexpr PrimitiveSetConfig Arithmetic = NodeType::Constant | NodeType::Variable | NodeType::Add | NodeType::Sub | NodeType::Mul | NodeType::Div;
    static constexpr PrimitiveSetConfig TypeCoherent = Arithmetic | NodeType::Pow | NodeType::Exp | NodeType::Log | NodeType::Sin | NodeType::Cos | NodeType::Square;
    static constexpr PrimitiveSetConfig Full = TypeCoherent | NodeType::Aq | NodeType::Tan | NodeType::Tanh | NodeType::Sqrt | NodeType::Cbrt;

    PrimitiveSet() = default;

    explicit PrimitiveSet(PrimitiveSetConfig config)
    {
        SetConfig(config);
    }

    [[nodiscard]] auto Primitives() const -> decltype(pset_) const& { return pset_; }

    auto AddPrimitive(Operon::Node node, size_t frequency, size_t minArity, size_t maxArity) -> bool
    {
        auto [_, ok] = pset_.insert({ node.HashValue, Primitive { node, frequency, minArity, maxArity } });
        return ok;
    }
    void RemovePrimitive(Operon::Node node) { pset_.erase(node.HashValue); }

    void RemovePrimitive(Operon::Hash hash) { pset_.erase(hash); }

    void SetConfig(PrimitiveSetConfig config)
    {
        pset_.clear();
        for (size_t i = 0; i < Operon::NodeTypes::Count; ++i) {
            auto t = static_cast<Operon::NodeType>(1U << i);
            Operon::Node n(t);

            if (((1U << i) & static_cast<uint32_t>(config)) != 0U) {
                pset_[n.HashValue] = { n, 1, n.Arity, n.Arity };
            }
        }
    }

    [[nodiscard]] auto EnabledPrimitives() const -> std::vector<Node> {
        std::vector<Node> nodes;
        for (auto const& [k, v] : pset_) {
            auto [node, freq, min_arity, max_arity] = v;
            if (node.IsEnabled && freq > 0) {
                nodes.push_back(node);
            }
        }
        return nodes;
    }

    [[nodiscard]] auto Config() const -> PrimitiveSetConfig
    {
        PrimitiveSetConfig conf { static_cast<PrimitiveSetConfig>(0) };
        for (auto [k, v] : pset_) {
            auto const& [node, freq, min_arity, max_arity] = v;
            if (node.IsEnabled && freq > 0) {
                conf |= node.Type;
            }
        }
        return conf;
    }

    [[nodiscard]] auto Frequency(Operon::Hash hash) const -> size_t
    {
        auto const& p = GetPrimitive(hash);
        return std::get<FREQUENCY>(p);
    }

    void SetFrequency(Operon::Hash hash, size_t frequency)
    {
        auto& p = GetPrimitive(hash);
        std::get<FREQUENCY>(p) = frequency;
    }

    [[nodiscard]] auto Contains(Operon::Hash hash) const -> bool { return pset_.contains(hash); }

    [[nodiscard]] auto IsEnabled(Operon::Hash hash) const -> bool
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
        SetEnabled(hash, /*enabled=*/true);
    }

    void Disable(Operon::Hash hash)
    {
        SetEnabled(hash, /*enabled=*/false);
    }

    [[nodiscard]] auto FunctionArityLimits() const -> std::pair<size_t, size_t>
    {
        auto minArity = std::numeric_limits<size_t>::max();
        auto maxArity = std::numeric_limits<size_t>::min();
        for (auto const& [key, val] : pset_) {
            if (std::get<NODE>(val).IsLeaf()) { continue; }
            minArity = std::min(minArity, std::get<MINARITY>(val));
            maxArity = std::max(maxArity, std::get<MAXARITY>(val));
        }
        return { minArity, maxArity };
    }

    OPERON_EXPORT auto SampleRandomSymbol(Operon::RandomGenerator& random, size_t minArity, size_t maxArity) const -> Operon::Node;

    void SetMinimumArity(Operon::Hash hash, size_t minArity)
    {
        EXPECT(minArity <= MaximumArity(hash));
        auto& p = GetPrimitive(hash);
        std::get<MINARITY>(p) = minArity;
    }

    [[nodiscard]] auto MinimumArity(Operon::Hash hash) const -> size_t
    {
        auto const& p = GetPrimitive(hash);
        return std::get<MINARITY>(p);
    }

    void SetMaximumArity(Operon::Hash hash, size_t maxArity)
    {
        EXPECT(maxArity >= MinimumArity(hash));
        auto& p = GetPrimitive(hash);
        std::get<MAXARITY>(p) = maxArity;
    }

    [[nodiscard]] auto MaximumArity(Operon::Hash hash) const -> size_t
    {
        auto const& p = GetPrimitive(hash);
        return std::get<MAXARITY>(p);
    }

    [[nodiscard]] auto MinMaxArity(Operon::Hash hash) const -> std::tuple<size_t, size_t>
    {
        auto const& p = GetPrimitive(hash);
        return { std::get<MINARITY>(p), std::get<MAXARITY>(p) };
    }

    void SetMinMaxArity(Operon::Hash hash, size_t minArity, size_t maxArity)
    {
        EXPECT(maxArity >= minArity);
        auto& p = GetPrimitive(hash);
        std::get<MINARITY>(p) = minArity;
        std::get<MAXARITY>(p) = maxArity;
    }

    // convenience overloads
    void SetFrequency(Operon::Node node, size_t frequency) { SetFrequency(node.HashValue, frequency); }
    [[nodiscard]] auto Frequency(Operon::Node node) const -> size_t { return Frequency(node.HashValue); }

    [[nodiscard]] auto Contains(Operon::Node node) const -> bool { return Contains(node.HashValue); }
    [[nodiscard]] auto IsEnabled(Operon::Node node) const -> bool { return IsEnabled(node.HashValue); }

    void SetEnabled(Operon::Node node, bool enabled) { SetEnabled(node.HashValue, enabled); }
    void Enable(Operon::Node node) { SetEnabled(node, /*enabled=*/true); }
    void Disable(Operon::Node node) { SetEnabled(node, /*enabled=*/false); }

    void SetMinimumArity(Operon::Node node, size_t minArity)
    {
        SetMinimumArity(node.HashValue, minArity);
    }
    [[nodiscard]] auto MinimumArity(Operon::Node node) const -> size_t { return MinimumArity(node.HashValue); }

    void SetMaximumArity(Operon::Node node, size_t maxArity)
    {
        SetMaximumArity(node.HashValue, maxArity);
    }
    [[nodiscard]] auto MaximumArity(Operon::Node node) const -> size_t { return MaximumArity(node.HashValue); }

    [[nodiscard]] auto MinMaxArity(Operon::Node node) const -> std::tuple<size_t, size_t>
    {
        return MinMaxArity(node.HashValue);
    }
    void SetMinMaxArity(Operon::Node node, size_t minArity, size_t maxArity)
    {
        SetMinMaxArity(node.HashValue, minArity, maxArity);
    }
};
} // namespace Operon
#endif
