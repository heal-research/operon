// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_MUTATION_HPP
#define OPERON_MUTATION_HPP

#include <utility>

#include "operon/core/operator.hpp"
#include "operon/core/pset.hpp"
#include "operon/core/tree.hpp"
#include "operon/core/variable.hpp"

namespace Operon {

struct CoefficientInitializerBase;
struct CreatorBase;

// the mutator can work in place or return a copy (child)
struct MutatorBase : public OperatorBase<Tree, Tree> {
};

template<typename Dist>
struct OPERON_EXPORT OnePointMutation : public MutatorBase {
    auto operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree override
    {
        auto& nodes = tree.Nodes();
        // sample a random node with an optimized value
        auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) { return n.Optimize; });
        if (it < nodes.end()) {
            it->Value += Dist(params_)(random);
        }

        return tree;
    }

    template <typename... Args>
    auto ParameterizeDistribution(Args... args) const -> void
    {
        params_ = typename Dist::param_type { std::forward<Args&&>(args)... };
    }

    private:
    mutable typename Dist::param_type params_;
};

template<typename Dist>
struct OPERON_EXPORT MultiPointMutation : public MutatorBase {
    auto operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree override
    {
        for (auto& node : tree.Nodes()) {
            if (node.Optimize) {
                node.Value += Dist(params_)(random);
            }
        }
        return tree;
    }

    template <typename... Args>
    auto ParameterizeDistribution(Args... args) const -> void
    {
        params_ = typename Dist::param_type { std::forward<Args&&>(args)... };
    }

    private:
    mutable typename Dist::param_type params_;
};

struct OPERON_EXPORT DiscretePointMutation : public MutatorBase {
    auto operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree override;

    auto Add(Operon::Scalar value, Operon::Scalar weight = 1.0) -> void {
        values_.push_back(value);
        weights_.push_back(weight);
    }

    private:
        std::vector<Operon::Scalar> weights_;
        std::vector<Operon::Scalar> values_;
};

struct OPERON_EXPORT MultiMutation : public MutatorBase {
    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

    void Add(const MutatorBase& op, double prob)
    {
        operators_.push_back(std::ref(op));
        probabilities_.push_back(prob);
    }

    [[nodiscard]] auto Count() const -> size_t { return operators_.size(); }

private:
    std::vector<std::reference_wrapper<const MutatorBase>> operators_;
    std::vector<double> probabilities_;
};

struct OPERON_EXPORT ChangeVariableMutation : public MutatorBase {
    explicit ChangeVariableMutation(Operon::Span<Operon::Hash const> variables)
        : variables_(variables.begin(), variables.end())
    {
    }

    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

private:
    std::vector<Operon::Hash> variables_;
};

struct OPERON_EXPORT ChangeFunctionMutation : public MutatorBase {
    explicit ChangeFunctionMutation(PrimitiveSet ps)
        : pset_(std::move(ps))
    {
    }

    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

private:
    PrimitiveSet pset_;
};

struct OPERON_EXPORT RemoveSubtreeMutation final : public MutatorBase {
    explicit RemoveSubtreeMutation(PrimitiveSet ps) : pset_(std::move(ps)) { }

    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

private:
    PrimitiveSet pset_;
};

struct OPERON_EXPORT InsertSubtreeMutation final : public MutatorBase {
    InsertSubtreeMutation(CreatorBase& creator, CoefficientInitializerBase& coeffInit, size_t maxDepth, size_t maxLength)
        : creator_(creator)
        , coefficientInitializer_(coeffInit)
        , maxDepth_(maxDepth)
        , maxLength_(maxLength)
    {
    }

    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

private:
    std::reference_wrapper<CreatorBase> creator_;
    std::reference_wrapper<CoefficientInitializerBase> coefficientInitializer_;
    size_t maxDepth_;
    size_t maxLength_;
};

struct OPERON_EXPORT ReplaceSubtreeMutation : public MutatorBase {
    ReplaceSubtreeMutation(CreatorBase& creator, CoefficientInitializerBase& coeffInit, size_t maxDepth, size_t maxLength)
        : creator_(creator) 
        , coefficientInitializer_(coeffInit)
        , maxDepth_(maxDepth)
        , maxLength_(maxLength)
    {
    }

    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

private:
    std::reference_wrapper<CreatorBase> creator_;
    std::reference_wrapper<CoefficientInitializerBase> coefficientInitializer_;
    size_t maxDepth_;
    size_t maxLength_;
};

struct OPERON_EXPORT ShuffleSubtreesMutation : public MutatorBase {
    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;
};
} // namespace Operon

#endif
