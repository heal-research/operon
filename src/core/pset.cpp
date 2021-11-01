// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "core/pset.hpp"

#include <numeric>

namespace Operon {
    Node PrimitiveSet::SampleRandomSymbol(Operon::RandomGenerator& random, size_t minArity, size_t maxArity) const
    {
        EXPECT(minArity <= maxArity);

        //std::array<NodeType, NodeTypes::Count> candidates;
        std::vector<Node> candidates; candidates.reserve(_pset.size());
        size_t idx = 0;
        for (auto const& [k, v] : _pset) {
            auto [node, freq, min_arity, max_arity] = v;
            if (!(node.IsEnabled && freq > 0)) continue;
            if (minArity > max_arity || maxArity < min_arity) continue;
            candidates[idx++] = node;
        }

        // throw an error if arity requirements are unreasonable (TODO: maybe here return optional)
        ENSURE(idx > 0);

        auto sum = std::transform_reduce(candidates.begin(), candidates.begin() + idx, 0.0, std::plus{}, [&](auto& n) { return GetFrequency(n.HashValue); });

        auto r = std::uniform_real_distribution<double>(0., sum)(random);
        auto c = 0.0;

        //Node node(NodeType::Constant);
        Node node;
        for (size_t i = 0; i < idx; ++i) {
            node = candidates[i];
            c += (double)GetFrequency(node.HashValue);

            if (c > r) {
                auto amin = std::max(minArity, GetMinimumArity(node.HashValue));
                auto amax = std::min(maxArity, GetMaximumArity(node.HashValue));
                auto arity = std::uniform_int_distribution<size_t>(amin, amax)(random);
                node.Arity = static_cast<uint16_t>(arity);
                break;
            }
        }

        ENSURE(IsEnabled(node.HashValue));
        ENSURE(GetFrequency(node.HashValue) > 0);

        return node;
    }
}
