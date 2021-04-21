// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#include "core/pset.hpp"

#include <numeric>
#include <execution>

namespace Operon {
    Node PrimitiveSet::SampleRandomSymbol(Operon::RandomGenerator& random, size_t minArity, size_t maxArity) const
    {
        EXPECT(minArity <= maxArity);

        std::array<NodeType, NodeTypes::Count> candidates;
        size_t idx = 0;

        for (size_t i = 0; i < NodeTypes::Count; ++i) {
            auto type = static_cast<NodeType>(1u << i);

            // skip symbols that are not enabled or have frequency set to zero
            if (!IsEnabled(type) || GetFrequency(type) == 0)
                continue;

            // get the min and max arities for this symbol
            auto [aMin, aMax] = GetMinMaxArity(type);

            // skip symbols that don't fit arity requirements
            if (minArity > aMax || maxArity < aMin)
                continue;

            candidates[idx++] = type;
        }

        // throw an error if arity requirements are unreasonable (TODO: maybe here return optional)
        ENSURE(idx > 0);

        auto sum = std::transform_reduce(candidates.begin(), candidates.begin() + idx, 0.0, std::plus{}, [&](auto& t) { return GetFrequency(t); });

        auto r = std::uniform_real_distribution<double>(0., sum)(random);
        auto c = 0.0;

        Node node(NodeType::Constant);
        for (size_t i = 0; i < idx; ++i) {
            auto type = candidates[i];
            c += (double)GetFrequency(type);

            if (c > r) {
                node = Node(type);

                auto amin = std::max(minArity, GetMinimumArity(type));
                auto amax = std::min(maxArity, GetMaximumArity(type));

                auto arity = std::uniform_int_distribution<size_t>(amin, amax)(random);
                node.Arity = gsl::narrow_cast<uint16_t>(arity);

                break;
            }
        }

        ENSURE(IsEnabled(node.Type));
        ENSURE(GetFrequency(node.Type) > 0);

        return node;
    }
}
