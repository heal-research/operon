// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2023 Heal Research

#ifndef OPERON_INDIVIDUAL_HPP
#define OPERON_INDIVIDUAL_HPP

#include "comparison.hpp"
#include "tree.hpp" 
#include "types.hpp" 
#include <cstddef>
#include <functional>

namespace Operon {

struct LexicographicalComparison; // fwd def

struct Individual {
    Tree Genotype;
    Operon::Vector<Operon::Scalar> Fitness;
    size_t Rank{}; // domination rank; used by NSGA2
    Operon::Scalar Distance{}; // crowding distance; used by NSGA2

    inline auto operator[](size_t const i) noexcept -> Operon::Scalar& { return Fitness[i]; }
    inline auto operator[](size_t const i) const noexcept -> Operon::Scalar { return Fitness[i]; }

    [[nodiscard]] inline auto Size() const noexcept -> size_t { return Fitness.size(); }

    Individual()
        : Individual(1)
    {
    }
    explicit Individual(size_t nObj)
        : Fitness(nObj, std::numeric_limits<Operon::Scalar>::max())
    {
    }
};

struct SingleObjectiveComparison {
    explicit SingleObjectiveComparison(size_t idx)
        : obj_(idx)
    {
    }
    SingleObjectiveComparison()
        : SingleObjectiveComparison(0)
    {
    }

    auto operator()(Individual const& lhs, Individual const& rhs, Operon::Scalar eps = 0) const -> bool
    {
        return Operon::Less{}(lhs[obj_], rhs[obj_], eps);
    }

    [[nodiscard]] auto GetObjectiveIndex() const -> size_t { return obj_; }
    void SetObjectiveIndex(size_t obj) { obj_ = obj; }

private:
    size_t obj_; // objective index
};

struct LexicographicalComparison {
    auto operator()(Individual const& lhs, Individual const& rhs, Operon::Scalar eps = 0) const -> bool
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        auto const& fit1 = lhs.Fitness;
        auto const& fit2 = rhs.Fitness;
        return Less{}(fit1.begin(), fit1.end(), fit2.begin(), fit2.end(), eps);
    }
};

// TODO: use a collection of SingleObjectiveComparison functors
// returns true if lhs dominates rhs
struct ParetoComparison {
    // assumes minimization in every dimension
    auto operator()(Individual const& lhs, Individual const& rhs, Operon::Scalar eps = 0) const -> bool
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        auto const& fit1 = lhs.Fitness;
        auto const& fit2 = rhs.Fitness;
        return ParetoDominance{}(fit1.begin(), fit1.end(), fit2.begin(), fit2.end(), eps) == Dominance::Left;
    }
};

struct CrowdedComparison {
    auto operator()(Individual const& lhs, Individual const& rhs, Operon::Scalar eps = 0) const -> bool
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        if (lhs.Rank != rhs.Rank) { return lhs.Rank < rhs.Rank; }
        return Operon::Less{}(rhs.Distance, lhs.Distance, eps);
    }
};

using ComparisonCallback = std::function<bool(Individual const&, Individual const&)>;

} // namespace Operon

#endif
