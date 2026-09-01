// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#include <ranges>
#include <vector>

#include "operon/core/concepts.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/operators/creator.hpp"

namespace Operon {

// See core/concepts.hpp for why these are asserted here rather than
// constraining a template. Asserted from this .cpp rather than creator.hpp:
// Concepts::Creator's return-type check needs Tree complete, and
// creator.hpp only forward-declares it. If BalancedTreeCreator/
// GrowTreeCreator/ProbabilisticTreeCreator's definitions ever move out of
// this translation unit, move these asserts (and the tree.hpp include) with them.
static_assert(Concepts::Creator<BalancedTreeCreator>);
static_assert(Concepts::Creator<GrowTreeCreator>);
static_assert(Concepts::Creator<ProbabilisticTreeCreator>);

CreatorBase::CreatorBase(gsl::not_null<PrimitiveSet const*> pset, std::vector<Operon::Hash> variables, size_t /*maxLength*/)
    : pset_(pset)
    , variables_(std::move(variables))
{
}

auto CreatorBase::SetPrimitiveSet(gsl::not_null<PrimitiveSet const*> pset) -> void
{
    pset_ = pset;
}

auto CreatorBase::AchievableLength(size_t targetLen) const -> size_t
{
    return pset_->AchievableLength(targetLen);
}


} // namespace Operon
