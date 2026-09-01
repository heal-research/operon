// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <algorithm>
#include <cstddef>
#include <ranges>
#include <cstdint>
#include <fmt/format.h>
#include <random>
#include <stdexcept>
#include <tuple>
#include <utility>
#include <vector>

#include "operon/core/pset.hpp"
#include "operon/core/contracts.hpp"
#include "operon/core/node.hpp"
#include "operon/core/standard_library.hpp"
#include "operon/core/types.hpp"

namespace Operon {

PrimitiveSet::PrimitiveSet(PrimitiveSet const& other)
    : pset_(other.pset_)
{
}

PrimitiveSet::PrimitiveSet(PrimitiveSet&& other) noexcept
    : pset_(std::move(other.pset_))
{
}

auto PrimitiveSet::operator=(PrimitiveSet const& other) -> PrimitiveSet&
{
    pset_ = other.pset_;
    reachable_.store(nullptr, std::memory_order_relaxed);
    reachableDirty_.store(true, std::memory_order_release);
    return *this;
}

auto PrimitiveSet::operator=(PrimitiveSet&& other) noexcept -> PrimitiveSet&
{
    pset_ = std::move(other.pset_);
    reachable_.store(nullptr, std::memory_order_relaxed);
    reachableDirty_.store(true, std::memory_order_release);
    return *this;
}

void PrimitiveSet::SetConfig(PrimitiveSetConfig config)
{
    pset_.clear();

    for (size_t i = 0; i < Operon::BuiltinOpCount; ++i) {
        if (!config.Test(i)) { continue; }
        auto const op = static_cast<Operon::BuiltinOp>(i);
        auto const [minArity, maxArity] = StandardLibrary::ArityLimits(op);
        auto n = Operon::Node::Function(static_cast<Operon::Hash>(op), minArity);
        pset_[n.HashValue] = { n, 1, minArity, maxArity };
    }

    for (auto type : { Operon::NodeType::Constant, Operon::NodeType::Variable, Operon::NodeType::Ref }) {
        if (!config.Test(Operon::BuiltinOpCount + Operon::NodeTypes::GetIndex(type))) { continue; }
        Operon::Node n(type);
        pset_[n.HashValue] = { n, 1, 0, 0 };
    }

    InvalidateReachability();
}

auto PrimitiveSet::ReachableLengths(size_t maxLength) const -> std::shared_ptr<std::vector<bool> const>
{
    auto reachable = reachable_.load(std::memory_order_acquire);
    if (!reachableDirty_.load(std::memory_order_acquire) && reachable != nullptr && reachable->size() >= maxLength) { return reachable; }

    std::lock_guard lock(reachableMutex_);
    reachable = reachable_.load(std::memory_order_relaxed);
    if (!reachableDirty_.load(std::memory_order_relaxed) && reachable != nullptr && reachable->size() >= maxLength) { return reachable; }

    auto rebuilt = std::make_shared<std::vector<bool>>(maxLength, false);
    if (maxLength != 0) {
        (*rebuilt)[0] = true;
        for (size_t i = 1; i < maxLength; ++i) {
            for (auto const& [_, primitive] : pset_) {
                auto const& [node, frequency, minArity, maxArity] = primitive;
                if (node.IsLeaf() || !node.IsEnabled || frequency == 0) { continue; }
                for (size_t arity = minArity; arity <= std::min(maxArity, i); ++arity) {
                    if ((*rebuilt)[i - arity]) {
                        (*rebuilt)[i] = true;
                        break;
                    }
                }
                if ((*rebuilt)[i]) { break; }
            }
        }
    }
    reachable_.store(std::shared_ptr<std::vector<bool> const>(std::move(rebuilt)), std::memory_order_release);
    reachableDirty_.store(false, std::memory_order_release);
    return reachable_.load(std::memory_order_acquire);
}

auto PrimitiveSet::AchievableLength(size_t targetLen) const -> size_t
{
    if (targetLen <= 1) { return 1; }
    auto const reachable = ReachableLengths(targetLen);
    for (auto length = targetLen; length > 0; --length) {
        if ((*reachable)[length - 1]) { return length; }
    }
    return 1;
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
            if (!node.IsEnabled || freq <= 0) { continue; }
            if (minArity > max_arity || maxArity < min_arity) { continue; }
            sum += freq;
            candidates.push_back(v);
        }

        if (candidates.empty()) {
            // arity requirements unreasonable
            throw std::runtime_error(fmt::format("PrimitiveSet::SampleRandomSymbol: unable to find suitable symbol with arity between {} and {}\n", minArity, maxArity));
        }

        Operon::Node result { Operon::NodeType::Constant };

        auto c { std::uniform_real_distribution<Operon::Scalar>(0, static_cast<Operon::Scalar>(sum))(random) };
        auto s { 0UL };
        for (auto const& [node, freq, min_arity, max_arity] : candidates) {
            s += freq;
            if (c < static_cast<Operon::Scalar>(s)) {
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
