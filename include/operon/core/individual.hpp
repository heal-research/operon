// SPDX-License-Identifier: MIT
// SPDX-FileCopyrightText: Copyright 2019-2021 Heal Research

#ifndef OPERON_INDIVIDUAL_HPP
#define OPERON_INDIVIDUAL_HPP

#include "core/tree.hpp"
#include "core/types.hpp"
#include <cstddef>

namespace Operon {

enum class Dominance : int { Left = 0,
    Equal = 1,
    Right = 2,
    None = 3 };

namespace detail {
    template <typename T>
    inline Dominance Compare(T const* lhs, T const* rhs, size_t n) noexcept
    {
        Eigen::Map<Eigen::Array<T, -1, 1> const> x(lhs, n), y(rhs, n);
        auto better = (x < y).any();
        auto worse = (x > y).any();
        if (better) { return worse ? Dominance::None : Dominance::Left; }
        if (worse) { return better ? Dominance::None : Dominance::Right; }
        return Dominance::Equal;
    }
}

struct LexicographicalComparison; // fwd def

struct Individual {
    Tree Genotype;
    Operon::Vector<Operon::Scalar> Fitness;
    size_t Rank; // domination rank; used by NSGA2
    Operon::Scalar Distance; // crowding distance; used by NSGA2

    inline Operon::Scalar& operator[](size_t const i) noexcept { return Fitness[i]; }
    inline Operon::Scalar operator[](size_t const i) const noexcept { return Fitness[i]; }

    inline size_t Size() const noexcept { return Fitness.size(); }

    Individual()
        : Individual(1)
    {
    }
    explicit Individual(size_t nObj)
        : Fitness(nObj, 0.0)
    {
    }

    inline bool operator==(Individual const& other) const noexcept
    {
        return std::equal(Fitness.begin(), Fitness.end(), other.Fitness.begin());
    }

    inline bool operator!=(Individual const& other) const noexcept
    {
        return !(*this == other);
    }

    inline bool LexicographicalCompare(Individual const& other) const noexcept;

    inline Dominance ParetoCompare(Individual const& other) const noexcept
    {
        return detail::Compare(Fitness.data(), other.Fitness.data(), Fitness.size());
    }
};

struct Comparison {
    virtual bool operator()(Individual const&, Individual const&) const = 0;
    virtual ~Comparison() noexcept = default;
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

struct LexicographicalComparison : public Comparison {
    bool operator()(Individual const& lhs, Individual const& rhs) const override
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        return std::lexicographical_compare(lhs.Fitness.cbegin(), lhs.Fitness.cend(), rhs.Fitness.cbegin(), rhs.Fitness.cend());
    }
};

// TODO: use a collection of SingleObjectiveComparison functors
// returns true if lhs dominates rhs
struct ParetoComparison : public Comparison {
    // assumes minimization in every dimension
    bool operator()(Individual const& lhs, Individual const& rhs) const override
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        bool better { false }, worse { false };

        for (size_t i = 0; i < std::size(lhs.Fitness); ++i) {
            better |= lhs[i] < rhs[i];
            worse |= lhs[i] > rhs[i];
        }

        return better && !worse;
    }
};

struct CrowdedComparison : public Comparison {

    bool operator()(Individual const& lhs, Individual const& rhs) const override
    {
        EXPECT(std::size(lhs.Fitness) == std::size(rhs.Fitness));
        return std::tie(lhs.Rank, rhs.Distance) < std::tie(rhs.Rank, lhs.Distance);
    }
};

bool Individual::LexicographicalCompare(Individual const& other) const noexcept
{
    return LexicographicalComparison{}(*this, other);
}

using ComparisonCallback = std::function<bool(Individual const&, Individual const&)>;

} // namespace

#endif
