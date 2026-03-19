// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_CREATOR_HPP
#define OPERON_CREATOR_HPP

#include <gsl/pointers>
#include <utility>

#include "operon/core/operator.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

class Tree;
class PrimitiveSet;

// the creator builds a new tree using the existing pset and allowed inputs
struct OPERON_EXPORT CreatorBase : public OperatorBase<Tree, size_t, size_t, size_t> {
    // maxLength: upper bound for the precomputed achievability table.
    // Pass the same value as the GP run's maximum tree length to avoid
    // per-call DP allocation.  Pass 0 to disable precomputation and fall
    // back to a stateless per-call DP in PrimitiveSet::AchievableLength.
    CreatorBase(gsl::not_null<PrimitiveSet const*> pset, std::vector<Operon::Hash> variables, size_t maxLength);

    [[nodiscard]] auto GetPrimitiveSet() const -> PrimitiveSet const* { return pset_.get(); }
    void SetPrimitiveSet(gsl::not_null<PrimitiveSet const*> pset);

    [[nodiscard]] auto GetVariables() const -> Operon::Span<Operon::Hash const> { return variables_; }
    auto SetVariables(Operon::Span<Operon::Hash const> variables) { variables_ = std::vector<Operon::Hash>(variables.begin(), variables.end()); }

protected:
    // Returns the largest tree length <= targetLen achievable with the current
    // pset. Uses the precomputed table when targetLen <= maxLength_ (O(targetLen)
    // scan, no allocation). Falls back to pset->AchievableLength otherwise.
    [[nodiscard]] auto AchievableLength(size_t targetLen) const -> size_t;

private:
    auto BuildAchievable() -> void;

    gsl::not_null<PrimitiveSet const*> pset_;
    std::vector<Operon::Hash>          variables_;
    std::vector<size_t>                snap_;      // snap_[i] = largest achievable length <= i+1
    size_t                             maxLength_; // size of snap_ table (0 = disabled)
};

// this tree creator expands bread-wise using a "horizon" of open expansion slots
// at the end the breadth sequence of nodes is converted to a postfix sequence
// if the depth is not limiting, the target length is guaranteed to be reached
class OPERON_EXPORT BalancedTreeCreator final : public CreatorBase {
public:
    BalancedTreeCreator(gsl::not_null<PrimitiveSet const*> pset, std::vector<Operon::Hash> variables, double bias, size_t maxLength)
        : CreatorBase(pset, std::move(variables), maxLength)
        , irregularityBias_(bias)
    {
    }

    auto operator()(Operon::RandomGenerator& random, size_t targetLen, size_t minDepth, size_t maxDepth) const -> Tree override;

    void SetBias(double bias) { irregularityBias_ = bias; }
    [[nodiscard]] auto GetBias() const -> double { return irregularityBias_; }

private:
    double irregularityBias_;
};

class OPERON_EXPORT GrowTreeCreator final : public CreatorBase {
    public:
        GrowTreeCreator(gsl::not_null<PrimitiveSet const*> pset, std::vector<Operon::Hash> variables, size_t maxLength)
            : CreatorBase(pset, std::move(variables), maxLength)
        { }

    auto operator()(Operon::RandomGenerator& random, size_t targetLen, size_t minDepth, size_t maxDepth) const -> Tree override;
};

class OPERON_EXPORT ProbabilisticTreeCreator final : public CreatorBase {
public:
    ProbabilisticTreeCreator(gsl::not_null<PrimitiveSet const*> pset, std::vector<Operon::Hash> variables, double bias, size_t maxLength)
        : CreatorBase(pset, std::move(variables), maxLength)
        , irregularityBias_(bias)
    {
    }

    auto operator()(Operon::RandomGenerator& random, size_t targetLen, size_t minDepth, size_t maxDepth) const -> Tree override;

    void SetBias(double bias) { irregularityBias_ = bias; }
    [[nodiscard]] auto GetBias() const -> double { return irregularityBias_; }

private:
    double irregularityBias_;
};

} // namespace Operon

#endif
