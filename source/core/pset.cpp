// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <numeric>

#include "operon/core/pset.hpp"

namespace Operon {
    auto PrimitiveSet::SampleRandomSymbol(Operon::RandomGenerator& random, size_t minArity, size_t maxArity) const -> Node
    {
        EXPECT(minArity <= maxArity);
        EXPECT(!pset_.empty());

        std::vector<Node> candidates; candidates.reserve(pset_.size());
        for (auto const& [k, v] : pset_) {
            auto [node, freq, min_arity, max_arity] = v;
            if (!(node.IsEnabled && freq > 0)) { continue; }
            if (minArity > max_arity || maxArity < min_arity) { continue; }
            candidates.push_back(node);
        }

        // throw an error if arity requirements are unreasonable (TODO: maybe here return optional)
        if (candidates.empty()) {
            // arity requirements unreasonable
            throw std::runtime_error(fmt::format("PrimitiveSet::SampleRandomSymbol: unable to find suitable symbol with arity between {} and {}\n", minArity, maxArity));
        }

        auto sum = std::transform_reduce(candidates.begin(), candidates.end(), 0.0, std::plus{}, [&](auto& n) { return Frequency(n.HashValue); });

        auto r = std::uniform_real_distribution<double>(0., sum)(random);
        auto c = 0.0;

        Node node(NodeType::Constant);
        for (auto const& candidate : candidates) {
            node = candidate;
            c += static_cast<double>(Frequency(node.HashValue));

            if (c > r) {
                auto amin = std::max(minArity, MinimumArity(node.HashValue));
                auto amax = std::min(maxArity, MaximumArity(node.HashValue));
                auto arity = std::uniform_int_distribution<size_t>(amin, amax)(random);
                node.Arity = static_cast<uint16_t>(arity);
                break;
            }
        }

        ENSURE(IsEnabled(node.HashValue));
        ENSURE(Frequency(node.HashValue) > 0);

        return node;
    }
} // namespace Operon
