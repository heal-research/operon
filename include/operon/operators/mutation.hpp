// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "core/operator.hpp"

namespace Operon {
struct OnePointMutation : public MutatorBase {
    Tree operator()(Operon::RandomGenerator&, Tree) const override;
};

struct MultiPointMutation : public MutatorBase {
    Tree operator()(Operon::RandomGenerator&, Tree) const override;
};

struct MultiMutation : public MutatorBase {
    Tree operator()(Operon::RandomGenerator&, Tree) const override;

    void Add(const MutatorBase& op, double prob)
    {
        operators.push_back(std::ref(op));
        probabilities.push_back(prob);
    }

    size_t Count() const { return operators.size(); }

private:
    std::vector<std::reference_wrapper<const MutatorBase>> operators;
    std::vector<double> probabilities;
};

struct ChangeVariableMutation : public MutatorBase {
    ChangeVariableMutation(const gsl::span<const Variable> vars)
        : variables(vars)
    {
    }

    Tree operator()(Operon::RandomGenerator&, Tree) const override;

private:
    const gsl::span<const Variable> variables;
};

struct ChangeFunctionMutation : public MutatorBase {
    ChangeFunctionMutation(PrimitiveSet ps)
        : pset(ps)
    {
    }

    Tree operator()(Operon::RandomGenerator&, Tree) const override;

private:
    PrimitiveSet pset;
};

struct RemoveSubtreeMutation final : public MutatorBase {
    RemoveSubtreeMutation(PrimitiveSet ps) : pset(ps) { }

    Tree operator()(Operon::RandomGenerator&, Tree) const override;

private:
    PrimitiveSet pset;
};

struct InsertSubtreeMutation : public MutatorBase {
    InsertSubtreeMutation(CreatorBase& creator, size_t maxDepth, size_t maxLength) 
        : creator_(creator)
        , maxDepth_(maxDepth)
        , maxLength_(maxLength)
    {
    }

    Tree operator()(Operon::RandomGenerator&, Tree) const override;

private:
    std::reference_wrapper<CreatorBase> creator_;
    size_t maxDepth_;
    size_t maxLength_;
};

struct ReplaceSubtreeMutation : public MutatorBase {
    ReplaceSubtreeMutation(CreatorBase& creator, size_t maxDepth, size_t maxLength) 
        : creator_(creator) 
        , maxDepth_(maxDepth)
        , maxLength_(maxLength)
    {
    }

    Tree operator()(Operon::RandomGenerator&, Tree) const override;

private:
    std::reference_wrapper<CreatorBase> creator_;
    size_t maxDepth_;
    size_t maxLength_;
};

struct ShuffleSubtreesMutation : public MutatorBase {
    Tree operator()(Operon::RandomGenerator&, Tree) const override;
};

}

#endif
