// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

#ifndef OPERON_CREATOR_HPP
#define OPERON_CREATOR_HPP

#include <gsl/pointers>
#include <utility>
#include <vector>

#include "operon/core/operator.hpp"
#include "operon/operon_export.hpp"

namespace Operon {

class Tree;
class PrimitiveSet;

// Builds a tree from (targetLen, minDepth, maxDepth). Depth parameters are
// creator-specific: Grow honors both; PTC2 enforces maxDepth but ignores
// minDepth and may undershoot targetLen; BTC preserves target length and
// ignores both depth values.
struct OPERON_EXPORT CreatorBase : public OperatorBase<Tree, size_t, size_t, size_t> {
    // maxLength is retained for source compatibility; reachability is cached by
    // the PrimitiveSet shared with all creators.
    CreatorBase(gsl::not_null<PrimitiveSet const*> pset, std::vector<Operon::Hash> variables, size_t maxLength);

    [[nodiscard]] auto GetPrimitiveSet() const -> PrimitiveSet const* { return pset_.get(); }
    void SetPrimitiveSet(gsl::not_null<PrimitiveSet const*> pset);

    [[nodiscard]] auto GetVariables() const -> Operon::Span<Operon::Hash const> { return variables_; }
    auto SetVariables(Operon::Span<Operon::Hash const> variables) { variables_ = std::vector<Operon::Hash>(variables.begin(), variables.end()); }

protected:
    // Returns the largest tree length <= targetLen reachable with the current
    // PrimitiveSet configuration.
    [[nodiscard]] auto AchievableLength(size_t targetLen) const -> size_t;

private:
    gsl::not_null<PrimitiveSet const*> pset_;
    std::vector<Operon::Hash>          variables_;
};

// This tree creator expands breadth-wise using a "horizon" of open expansion slots.
// It always returns its snapped target length. It ignores minDepth and maxDepth:
// enforcing either can make that length impossible at high bias. Use PTC2 for a
// hard maximum depth; PTC2 can then return a shorter tree.
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
