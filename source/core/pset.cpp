// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <fmt/core.h>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "operon/core/pset.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/types.hpp"

namespace Operon {
    [[nodiscard]] auto PrimitiveSet::GetPrimitive(Operon::Hash hash) const -> Primitive const& {
        auto it = pset_.find(hash);
        if (it == pset_.end()) {
            throw std::runtime_error(fmt::format("Unknown node hash {}\n", hash));
        }
        return it->second;
    }

    auto PrimitiveSet::SampleRandomSymbol(Operon::RandomGenerator& random, size_t minArity, size_t maxArity) const -> Node
    {
        EXPECT(minArity <= maxArity);
        EXPECT(!pset_.empty());

        std::vector<Primitive> candidates;
        candidates.reserve(pset_.size());

        auto sum{0UL};
        for (auto const& [k, v] : pset_) {
            auto const& [node, freq, min_arity, max_arity] = v;
            if (!(node.IsEnabled && freq > 0)) { continue; }
            if (minArity > max_arity || maxArity < min_arity) { continue; }
            sum += freq;
            candidates.push_back(v);
        }

        if (candidates.empty()) {
            // arity requirements unreasonable
            throw std::runtime_error(fmt::format("PrimitiveSet::SampleRandomSymbol: unable to find suitable symbol with arity between {} and {}\n", minArity, maxArity));
        }

        Operon::Node result { Operon::NodeType::Constant };

        auto c { std::uniform_real_distribution<Operon::Scalar>(0, sum)(random) };
        auto s { 0UL };
        for (auto const& [node, freq, min_arity, max_arity] : candidates) {
            s += freq;
            if (c < s) {
                auto amin = std::max(minArity, MinimumArity(node.HashValue));
                auto amax = std::min(maxArity, MaximumArity(node.HashValue));
                auto arity = std::uniform_int_distribution<size_t>(amin, amax)(random);
                result = node;
                result.Arity = static_cast<uint16_t>(arity);
                break;
            }
        }

        ENSURE(IsEnabled(result.HashValue));
        ENSURE(Frequency(result.HashValue) > 0);

        return result;
    }
} // namespace Operon
