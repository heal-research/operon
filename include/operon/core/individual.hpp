#ifndef OPERON_INDIVIDUAL_HPP
#define OPERON_INDIVIDUAL_HPP

#include <cstddef>
#include "core/tree.hpp"
#include "core/types.hpp"
#include <gsl/util>

namespace Operon {

struct Individual {
    Tree Genotype;
    std::vector<Operon::Scalar> Fitness;

    Operon::Scalar& operator[](size_t i) noexcept { return Fitness[i]; }
    Operon::Scalar operator[](size_t i) const noexcept { return Fitness[i]; }

    Individual()
        : Individual(1)
    {
    }
    Individual(size_t fitDim)
        : Fitness(fitDim, 0.0)
    {
    }
};

struct Comparison {
    virtual bool operator()(Individual const&, Individual const&) const = 0;
    virtual ~Comparison() {}
};

struct SingleObjectiveComparison final : public Comparison {
    SingleObjectiveComparison(size_t idx)
        : objectiveIndex(idx)
    {
    }
    SingleObjectiveComparison()
        : SingleObjectiveComparison(0)
    {
    }

    bool operator()(Individual const& lhs, Individual const& rhs) const override
    {
        return lhs[objectiveIndex] < rhs[objectiveIndex];
    }

private:
    size_t objectiveIndex;
};

using ComparisonCallback = std::function<bool(Individual const&, Individual const&)>;

} // namespace

#endif
