// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_MUTATION_HPP
#define OPERON_MUTATION_HPP

#include <utility>

#include "operon/core/operator.hpp"
#include "operon/core/pset.hpp"
//#include "operon/operators/creator.hpp"
#include "operon/core/tree.hpp"
//#include "operon/core/pset.hpp"

namespace Operon {

struct CreatorBase;
struct Variable;

// the mutator can work in place or return a copy (child)
struct MutatorBase : public OperatorBase<Tree, Tree> {
};

template<typename Dist>
struct OPERON_EXPORT OnePointMutation : public MutatorBase {
    auto operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree override
    {
        auto& nodes = tree.Nodes();
        // sample a random leaf
        auto it = Operon::Random::Sample(random, nodes.begin(), nodes.end(), [](auto const& n) { return n.IsLeaf(); });
        EXPECT(it < nodes.end());
        it->Value += dist_(random);

        return tree;
    }

    template <typename... Args>
    auto ParameterizeDistribution(Args... args) const -> void
    {
        typename Dist::param_type params { static_cast<typename Dist::result_type>(args)... };
        dist_.param(params);
    }

    private:
    mutable Dist dist_;
};

template<typename Dist>
struct OPERON_EXPORT MultiPointMutation : public MutatorBase {
    auto operator()(Operon::RandomGenerator& random, Tree tree) const -> Tree override
    {
        for (auto& node : tree.Nodes()) {
            if (node.IsLeaf()) {
                node.Value += dist_(random);
            }
        }
        return tree;
    }

    template <typename... Args>
    auto ParameterizeDistribution(Args... args) const -> void
    {
        typename Dist::param_type params { static_cast<typename Dist::result_type>(args)... };
        dist_.param(params);
    }

    private:
    mutable Dist dist_;
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
    explicit ChangeVariableMutation(const Operon::Span<const Variable> vars)
        : variables(vars)
    {
    }

    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

private:
    const Operon::Span<Variable const> variables;
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
    InsertSubtreeMutation(CreatorBase& creator, size_t maxDepth, size_t maxLength, PrimitiveSet pset)
        : creator_(creator)
        , maxDepth_(maxDepth)
        , maxLength_(maxLength)
        , pset_(std::move(pset))
    {
    }

    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

private:
    std::reference_wrapper<CreatorBase> creator_;
    size_t maxDepth_;
    size_t maxLength_;
    PrimitiveSet pset_;
};

struct OPERON_EXPORT ReplaceSubtreeMutation : public MutatorBase {
    ReplaceSubtreeMutation(CreatorBase& creator, size_t maxDepth, size_t maxLength) 
        : creator_(creator) 
        , maxDepth_(maxDepth)
        , maxLength_(maxLength)
    {
    }

    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;

private:
    std::reference_wrapper<CreatorBase> creator_;
    size_t maxDepth_;
    size_t maxLength_;
};

struct OPERON_EXPORT ShuffleSubtreesMutation : public MutatorBase {
    auto operator()(Operon::RandomGenerator& /*random*/, Tree /*args*/) const -> Tree override;
};
} // namespace Operon

#endif
