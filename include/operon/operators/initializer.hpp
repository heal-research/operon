// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INITIALIZER_HPP
#define OPERON_INITIALIZER_HPP

#include "operon/core/tree.hpp"
#include "operon/operators/creator.hpp"

namespace Operon {

struct CoefficientInitializerBase : public OperatorBase<void, Tree&> {
};

struct TreeInitializerBase : public OperatorBase<Tree> {
};

template <typename Dist>
struct OPERON_EXPORT CoefficientInitializer : public CoefficientInitializerBase {
    using NodeCheckCallback = std::function<bool(Operon::Node)>;

    explicit CoefficientInitializer(NodeCheckCallback callback)
        : callback_(std::move(callback))
    {
    }

    CoefficientInitializer()
        : CoefficientInitializer([](auto const& node) { return node.IsLeaf(); })
    {
    }

    auto operator()(Operon::RandomGenerator& random, Operon::Tree& tree) const -> void override
    {
        for (auto& node : tree.Nodes()) {
            if (callback_(node)) {
                node.Value = Dist(params_)(random);
            }
        }
    }

    template <typename... Args>
    auto ParameterizeDistribution(Args... args) const -> void
    {
        params_ = typename Dist::param_type { std::forward<Args&&>(args)... };
    }

private:
    mutable typename Dist::param_type params_;
    NodeCheckCallback callback_;
};

template <typename Dist>
struct OPERON_EXPORT TreeInitializer : public TreeInitializerBase {
    explicit TreeInitializer(Operon::CreatorBase& creator)
        : creator_(creator)
    {
    }

    auto operator()(Operon::RandomGenerator& random) const -> Operon::Tree override
    {
        auto targetLen = std::max(size_t { 1 }, static_cast<size_t>(std::round(Dist(params_)(random))));
        return creator_(random, targetLen, minDepth_, maxDepth_); // initialize tree
    }

    template <typename... Args>
    auto ParameterizeDistribution(Args... args) const -> void
    {
        params_ = typename Dist::param_type { std::forward<Args&&>(args)... };
    }

    void SetMinDepth(size_t minDepth) { minDepth_ = minDepth; }
    auto MinDepth() const -> size_t { return minDepth_; }

    void SetMaxDepth(size_t maxDepth) { maxDepth_ = maxDepth; }
    auto MaxDepth() const -> size_t { return maxDepth_; }

    void SetCreator(CreatorBase const& creator) { creator_ = creator; }
    [[nodiscard]] auto Creator() const -> CreatorBase const& { return creator_.get(); }

    static constexpr size_t DefaultMaxDepth { 1000 }; // we don't want a depth restriction to limit the achievable shapes/lengths

private:
    mutable typename Dist::param_type params_;
    std::reference_wrapper<Operon::CreatorBase const> creator_;
    size_t minDepth_ { 1 };
    size_t maxDepth_ { DefaultMaxDepth };
};

// wraps a creator and generates trees from a given size distribution
using UniformCoefficientInitializer = CoefficientInitializer<std::uniform_real_distribution<Operon::Scalar>>;
using NormalCoefficientInitializer = CoefficientInitializer<std::normal_distribution<Operon::Scalar>>;

using UniformTreeInitializer = TreeInitializer<std::uniform_int_distribution<size_t>>;
using NormalTreeInitializer = TreeInitializer<std::normal_distribution<>>;
} // namespace Operon
#endif
