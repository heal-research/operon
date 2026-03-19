// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#include <algorithm>
#include <ranges>
#include <vector>

#include "operon/core/pset.hpp"
#include "operon/operators/creator.hpp"

namespace Operon {

CreatorBase::CreatorBase(gsl::not_null<PrimitiveSet const*> pset, std::vector<Operon::Hash> variables, size_t maxLength)
    : pset_(pset)
    , variables_(std::move(variables))
    , maxLength_(maxLength)
{
    BuildAchievable();
}

auto CreatorBase::SetPrimitiveSet(gsl::not_null<PrimitiveSet const*> pset) -> void
{
    pset_ = pset;
    BuildAchievable();
}

auto CreatorBase::BuildAchievable() -> void
{
    snap_.clear();
    if (maxLength_ == 0) { return; }

    // Collect all distinct arities from enabled non-leaf primitives.
    std::vector<size_t> arities;
    for (auto const& [key, val] : pset_->Primitives()) {
        auto const& [node, freq, minAr, maxAr] = val;
        if (node.IsLeaf() || !node.IsEnabled || freq == 0) { continue; }
        for (auto a = minAr; a <= maxAr; ++a) {
            if (std::ranges::find(arities, a) == arities.end()) {
                arities.push_back(a);
            }
        }
    }

    // dp[i] == true iff a tree of length (i+1) is achievable.
    // Length 1 (single leaf) is always achievable.
    // Length n > 1 is achievable iff (n-1) is a sum of available arities.
    std::vector<bool> dp(maxLength_, false);
    dp[0] = true;
    for (size_t i = 1; i < maxLength_; ++i) {
        for (auto a : arities) {
            if (i >= a && dp[i - a]) { dp[i] = true; break; }
        }
    }

    // Build the snap-down table: snap_[i] = largest achievable length <= i+1.
    // A single forward pass carries the last seen achievable length forward,
    // so AchievableLength(n) reduces to a single array lookup: snap_[n-1].
    snap_.resize(maxLength_);
    snap_[0] = 1; // length 1 is always achievable
    for (size_t i = 1; i < maxLength_; ++i) {
        snap_[i] = dp[i] ? i + 1 : snap_[i - 1];
    }
}

auto CreatorBase::AchievableLength(size_t targetLen) const -> size_t
{
    if (targetLen <= 1) { return 1; }
    if (targetLen <= snap_.size()) { return snap_[targetLen - 1]; }
    // Fallback for lengths beyond the precomputed table (rare).
    return pset_->AchievableLength(targetLen);
}

} // namespace Operon
