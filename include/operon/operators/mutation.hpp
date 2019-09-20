#ifndef MUTATION_HPP
#define MUTATION_HPP

#include "core/operator.hpp"

namespace Operon {
struct OnePointMutation : public MutatorBase {
    Tree operator()(operon::rand_t& random, Tree tree) const override;
};

struct MultiPointMutation : public MutatorBase {
    Tree operator()(operon::rand_t& random, Tree tree) const override;
};

struct MultiMutation : public MutatorBase {
    Tree operator()(operon::rand_t& random, Tree tree) const override;

    void Add(const MutatorBase& op, double prob)
    {
        operators.push_back(std::ref(op));
        partials.push_back(partials.empty() ? prob : prob + partials.back());
    }

private:
    static constexpr double eps = std::numeric_limits<double>::epsilon();
    std::vector<std::reference_wrapper<const MutatorBase>> operators;
    std::vector<double> partials;
};

struct ChangeVariableMutation : public MutatorBase {
    ChangeVariableMutation(const gsl::span<const Variable> vars)
        : variables(vars)
    {
    }

    Tree operator()(operon::rand_t& random, Tree tree) const override;

private:
    const gsl::span<const Variable> variables;
};
}

#endif
