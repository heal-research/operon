// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2025 Heal Research
// SPDX-FileCopyrightText: Copyright 2025-present Bogdan Burlacu and contributors

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

    template<typename Self>
    auto operator[](this Self& self, size_t const i) noexcept -> decltype(auto) { return (self.Fitness[i]); }

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

// Feasibility-first comparator: a feasible individual always precedes an
// infeasible one, regardless of fitness; individuals with equal
// feasibility defer to `Fallback` (default: single-objective on
// objective 0). This is the constrained-dominance alternative to
// ShapeConstrainedEvaluator's own worst-value-substitution gate: the
// gate reproduces Kronberger et al. 2021's Algorithm 1 exactly (a
// rejected individual's fitness is replaced, not given an extra
// objective), this is for callers who instead want feasibility treated
// as a first-class ordering criterion. The two are independent and can
// be combined (see operon_gp.cpp, which does).
//
// `IsFeasible` is intentionally a predicate over the genotype, not a
// Fitness-vector read: EvaluatorBase::Evaluate() takes `Individual
// const&` and returns only a fitness vector, so a wrapping evaluator
// like ShapeConstrainedEvaluator has no channel to tag an Individual as
// infeasible other than the fitness values themselves -- and a
// worst-value fitness substitution isn't reliably distinguishable from a
// genuinely bad score. This struct itself does no caching -- it's a
// thin, trivial wrapper matching the other comparators in this header.
// Repeated predicate calls are cheap only when the caller supplies a
// caching predicate (for example ShapeConstrainedEvaluator::Feasible(),
// whose cache is populated by Prepare() and Evaluate()); non-caching
// predicates may recompute on each comparison.
struct FeasibilityFirstComparison {
    using FeasibilityPredicate = std::function<bool(Tree const&)>;

    explicit FeasibilityFirstComparison(FeasibilityPredicate isFeasible, ComparisonCallback fallback = SingleObjectiveComparison{})
        : isFeasible_(std::move(isFeasible))
        , fallback_(std::move(fallback))
    {
    }

    auto operator()(Individual const& lhs, Individual const& rhs, Operon::Scalar /*eps*/ = 0) const -> bool
    {
        auto const lf = isFeasible_(lhs.Genotype);
        auto const rf = isFeasible_(rhs.Genotype);
        if (lf != rf) { return lf; } // feasible (true) precedes infeasible (false)
        return fallback_(lhs, rhs);
    }

private:
    FeasibilityPredicate isFeasible_;
    ComparisonCallback fallback_;
};

} // namespace Operon

#endif
